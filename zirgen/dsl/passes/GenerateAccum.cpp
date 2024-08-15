// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

namespace zirgen {
namespace dsl {

class AccumBuilder {
public:
  AccumBuilder(OpBuilder& builder, Value accumLayout, Value randomnessLayout)
      : builder(builder), accumLayout(accumLayout) {
    // Preemptively lookup all verifier randomness values and store them in a
    // dictionary. Unneeded values will be pruned later by folding.
    zeroDistance =
        builder.create<arith::ConstantOp>(randomnessLayout.getLoc(), builder.getIndexAttr(0));
    ValType extValType = Zhlt::getValExtType(builder.getContext());
    auto randomnessLayoutType = randomnessLayout.getType().cast<LayoutType>();
    for (auto field : randomnessLayoutType.getFields().drop_back()) {
      auto [entry, _] = verifierRandomness.insert({field.name, DenseMap<StringRef, Value>()});
      Value typeLookup =
          builder.create<LookupOp>(randomnessLayout.getLoc(), randomnessLayout, field.name);
      for (auto subfield : field.type.cast<LayoutType>().getFields()) {
        Value randomness = builder.create<LookupOp>(typeLookup.getLoc(), typeLookup, subfield.name);
        randomness = builder.create<ZStruct::LoadOp>(
            randomness.getLoc(), extValType, randomness, zeroDistance);
        entry->second.insert({subfield.name, randomness});
      }
    }

    offset = builder.create<LookupOp>(randomnessLayout.getLoc(), randomnessLayout, "$offset");
    offset = builder.create<ZStruct::LoadOp>(offset.getLoc(), extValType, offset, zeroDistance);

    Location loc = accumLayout.getLoc();
    auto accumLayoutType = accumLayout.getType().cast<LayoutArrayType>();

    // Read last accumulator from previous row
    IntegerAttr distanceAttr = builder.getIndexAttr(1);
    Value distance = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), distanceAttr);
    Value lastLayout = getAccumColumnLayout(accumLayoutType.getSize() - 1);
    this->oldT = builder.create<ZStruct::LoadOp>(loc, extValType, lastLayout, distance);
    this->t = oldT;

    this->accCount = 0;
    this->accCol = 0;
  }

  void build(Value majorLayout) { t = doAccum(t, majorLayout); }

  // Write and constrain the ultimate accumulator value for this cycle to the
  // last column in the accum buffer.
  void finalize() {
    Location loc = accumLayout.getLoc();
    auto accumLayoutType = accumLayout.getType().cast<LayoutArrayType>();
    ValType extValType = Zhlt::getValExtType(builder.getContext());

    // If we have a non-multiple of 3 number of arguments, be sure to write
    // the last accumulator value into a register
    if (accCount != 0) {
      t = storeTemporarySum(t);
    }

    // If necessary, copy the ultimate accumulator value to the last column so
    // it is always in the same place for the next cycle. We also need to
    // constrain this value to match the last used accum column. Note that if
    // accCol == 0 at this point, this major contains no arguments at all, but
    // we still need to write the last accum column for this cycle.
    if (accCol == 0 || accCol != accumLayoutType.getSize()) {
      Value lastLayout = getAccumColumnLayout(accumLayoutType.getSize() - 1);
      builder.create<StoreOp>(loc, lastLayout, t);
      Value newT =
          builder.create<LoadOp>(lastLayout.getLoc(), extValType, lastLayout, zeroDistance);
      Value diff = builder.create<SubOp>(newT.getLoc(), newT, oldT);
      builder.create<EqualZeroOp>(diff.getLoc(), diff);
    }
  }

private:
  // Condense all non-count columns for this argument component down to a
  // single value: v[i] = r_a * a[i] + r_b * b[i] + ...
  Value condenseArgument(LayoutType type, Value layout) {
    ValType extValType = Zhlt::getValExtType(builder.getContext());
    SmallVector<uint64_t> zero(extValType.getFieldK(), 0);
    Value v = builder.create<ConstOp>(layout.getLoc(), extValType, zero);
    for (auto field : type.getFields().drop_front()) {
      Value r = verifierRandomness[type.getId()][field.name];
      Value colLayout = builder.create<LookupOp>(layout.getLoc(), layout, field.name);
      colLayout = builder.create<LookupOp>(colLayout.getLoc(), colLayout, "@super");
      Value colValue =
          builder.create<ZStruct::LoadOp>(colLayout.getLoc(), extValType, colLayout, zeroDistance);
      Value sumTerm = builder.create<MulOp>(colValue.getLoc(), r, colValue);
      v = builder.create<AddOp>(sumTerm.getLoc(), v, sumTerm);
    }
    return v;
  }

  void addConstraint(Value newT) {
    Value deltaT = builder.create<SubOp>(newT.getLoc(), newT, oldT);
    Value constraint = builder.create<MulOp>(deltaT.getLoc(), deltaT, constraintLhs);
    for (size_t i = 0; i < accCount; i++) {
      constraint = builder.create<SubOp>(constraint.getLoc(), constraint, constraintRhsTerms[i]);
    }
    builder.create<EqualZeroOp>(constraint.getLoc(), constraint);
  }

  // Writes t to the next accum column, and returns a value loaded from it.
  Value storeTemporarySum(Value t) {
    assert(accCount != 0 && "writing accum column without accumulating any new arguments");
    ValType extValType = Zhlt::getValExtType(builder.getContext());
    Value tLayout = getAccumColumnLayout(accCol);
    builder.create<StoreOp>(t.getLoc(), tLayout, t);
    Value newT = builder.create<ZStruct::LoadOp>(t.getLoc(), extValType, tLayout, zeroDistance);
    addConstraint(newT);

    oldT = newT;
    accCount = 0;
    accCol++;
    return newT;
  }

  // Build terms for the constraint on the accum registers.
  void buildConstraintTerms(Value vPlusOffset, Value c) {
    // If we accum 3 arguments, then:
    // constraintLhs = (v_1 + x) (v_2 + x) (v_3 + x)
    // constraintRhsTerms[0] = c_1 (v2 + x) (v_3 + x)
    // constraintRhsTerms[1] = (v1 + x) c_2 (v_3 + x)
    // constraintRhsTerms[2] = (v1 + x) (v_2 + x) c_3
    //
    // If we accum fewer than 3 arguments, then each of these values will have
    // only the product terms corresponding to the accumulated arguments. We
    // will still construct extra products for the unused term(s), but these
    // will be removed by folding at the end of the pass.
    if (accCount == 0) {
      constraintLhs = vPlusOffset;
      constraintRhsTerms[0] = c;
      constraintRhsTerms[1] = vPlusOffset;
      constraintRhsTerms[2] = vPlusOffset;
    } else {
      constraintLhs = builder.create<MulOp>(vPlusOffset.getLoc(), constraintLhs, vPlusOffset);
      constraintRhsTerms[accCount] =
          builder.create<MulOp>(c.getLoc(), constraintRhsTerms[accCount], c);
      for (size_t i = 0; i < 3; i++) {
        if (i != accCount) {
          constraintRhsTerms[i] =
              builder.create<MulOp>(vPlusOffset.getLoc(), constraintRhsTerms[i], vPlusOffset);
        }
      }
    }
  }

  // Accumulates the given argument into a temporary accumulator value t:
  // t' = t + c[i] / (v[i] + x)
  Value accumulateArgument(Value t, LayoutType type, Value layout) {
    ValType extValType = Zhlt::getValExtType(builder.getContext());
    StringAttr countName = type.getFields()[0].name;
    Value cLayout = builder.create<LookupOp>(layout.getLoc(), layout, countName);
    cLayout = builder.create<LookupOp>(cLayout.getLoc(), cLayout, "@super");
    Value c = builder.create<ZStruct::LoadOp>(cLayout.getLoc(), extValType, cLayout, zeroDistance);
    Value v = condenseArgument(type, layout);
    Value vPlusOffset = builder.create<AddOp>(v.getLoc(), v, offset);
    Value denominator = builder.create<InvOp>(v.getLoc(), vPlusOffset);
    Value delta = builder.create<MulOp>(c.getLoc(), c, denominator);
    Value tNew = builder.create<AddOp>(t.getLoc(), t, delta);

    buildConstraintTerms(vPlusOffset, c);

    accCount++;
    if (accCount == 3) {
      storeTemporarySum(tNew);
    }
    return tNew;
  }

  Value doAccum(Value t, Value layout) {
    TypeSwitch<Type>(layout.getType())
        .Case<LayoutArrayType>([&](auto layoutArrayType) {
          // TODO: we could consider generating this as a loop, but a loop
          // iteration might not accumulate a multiple of 3 arguments which
          // would disrupt the register allocation and constraint generation.
          for (size_t i = 0; i < layoutArrayType.getSize(); i++) {
            IntegerAttr indexAttr = builder.getIndexAttr(i);
            Value indexValue = builder.create<arith::ConstantOp>(
                layout.getLoc(), builder.getIndexType(), indexAttr);
            Value sublayout = builder.create<SubscriptOp>(layout.getLoc(), layout, indexValue);
            t = doAccum(t, sublayout);
          }
        })
        .Case<LayoutType>([&](auto layoutType) {
          if (layoutType.getKind() == LayoutKind::Argument) {
            t = accumulateArgument(t, layoutType, layout);
            return;
          }
          if (layoutType.getKind() == LayoutKind::Mux) {
            // Arguments are already hoisted out of muxes
            return;
          }
          for (auto field : layoutType.getFields()) {
            Value sublayout = builder.create<LookupOp>(layout.getLoc(), layout, field.name);
            t = doAccum(t, sublayout);
          }
        });
    return t;
  }

  Value getAccumColumnLayout(size_t index) {
    Location loc = accumLayout.getLoc();
    IntegerAttr indexAttr = builder.getIndexAttr(index);
    Value indexValue = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), indexAttr);
    return builder.create<SubscriptOp>(loc, accumLayout, indexValue);
  }

  OpBuilder& builder;

  // The layout of the accum buffer
  Value accumLayout;

  // A zero distance value for LoadOp
  Value zeroDistance;

  // A map of argument types and member names to global verifier randomness
  DenseMap<StringRef, DenseMap<StringRef, Value>> verifierRandomness;

  // The global "offset" value used for SumCheck summation
  Value offset;

  // The last value of the grand sum written to the witness and constrainted
  Value oldT;

  // The running grand sum for the current cycle
  Value t;

  // The term (t' - t)(v_1 + x)(v_2 + x)(v_3 + x) for LogUp sum constraints
  Value constraintLhs;

  // The terms c_i * product (v_j + x) for j != i for LogUp sum constraints
  std::array<Value, 3> constraintRhsTerms;

  // The offset into the accum buffer the next argument should be accumulated
  size_t accCol = 0;

  // The number of arguments that have been accumulated into the column at
  // offset accCol. This should range from 0 to 2.
  size_t accCount = 0;
};

struct GenerateAccumPass : public GenerateAccumBase<GenerateAccumPass> {
  void runOnOperation() override {
    getOperation().walk([&](ComponentOp component) {
      llvm::StringRef baseName = component.getName();
      if (baseName.ends_with("$accum") || (baseName != "Top" && !baseName.starts_with("test$")))
        return;

      buildAccumStep(component);
    });
  }

  void buildAccumStep(ComponentOp component) {
    auto layoutType = component.getLayoutType();
    if (!layoutType)
      return;

    llvm::MapVector<Type, size_t> argumentCounts = Zhlt::muxArgumentCounts(layoutType);
    if (argumentCounts.empty())
      return;

    auto ctx = component->getContext();
    auto loc = component->getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(getOperation().getBody());

    // Create Accum component, which takes the Top layout as a parameter and the
    // accum buffer as its layout.
    SmallVector<Type> accumParams = {component.getLayoutType()};
    size_t accumColumns = getAccumColumnCount(argumentCounts);
    auto accumLayoutType = LayoutArrayType::get(ctx, Zhlt::getExtRefType(ctx), accumColumns);
    std::string accumName = (component.getName() + "$accum").str();
    auto accum = builder.create<Zhlt::ComponentOp>(
        loc, accumName, Zhlt::getComponentType(ctx), accumParams, accumLayoutType);
    SymbolTable::setSymbolVisibility(accum, SymbolTable::Visibility::Public);
    builder.setInsertionPointToStart(accum.addEntryBlock());

    // Create globals for verifier randomness
    LayoutType mixLayoutType = getRandomnessLayoutType(argumentCounts);
    auto randomnessLayout =
        builder.create<Zhlt::GetGlobalLayoutOp>(loc, mixLayoutType, "mix", "randomness");

    // Generate IR to compute the grand sum
    AccumBuilder accumBuilder(builder, accum.getLayout(), randomnessLayout);
    accumBuilder.build(accum.getConstructParam()[0]);
    accumBuilder.finalize();

    Value super = builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), ValueRange{});
    builder.create<Zhlt::ReturnOp>(loc, super);
  }

private:
  LayoutType getRandomnessLayoutType(llvm::MapVector<Type, size_t>& argumentCounts) {
    // Allocate verifier randomness in the mix buffer. We need one column for
    // each non-count column of each argument type, and one more for the offset
    // used by the LogUp grand sum.
    ModuleOp mod = getOperation();
    MLIRContext* ctx = &getContext();
    RefType extType = Zhlt::getExtRefType(ctx);

    SmallVector<ZStruct::FieldInfo> members;
    mod.walk([&](ComponentOp component) {
      auto type = dyn_cast_or_null<LayoutType>(component.getLayoutType());
      if (!type || type.getKind() != LayoutKind::Argument)
        return;

      SmallVector<ZStruct::FieldInfo> submembers;
      for (auto field : type.getFields().drop_front()) {
        assert(field.type == Zhlt::getNondetRegLayoutType(ctx));
        submembers.push_back({field.name, extType});
      }
      auto fieldName = StringAttr::get(ctx, type.getId());
      auto fieldTypeName = StringAttr::get(ctx, "arg$" + type.getId());
      members.push_back({fieldName, LayoutType::get(ctx, fieldTypeName, submembers)});
    });

    // Include one extra global column for the LogUp offset
    members.push_back({StringAttr::get(ctx, "$offset"), extType});

    return LayoutType::get(ctx, "$accum", members);
  }

  // Compute number of accumulator columns. Each argument instance is compressed
  // to a single extension field element using verifier randomness, and then
  // three arguments can be accumulated using a degree 4 constraint (which is
  // the highest we can manage inside the major mux). therefore, we need one
  // accum column for every three arguments, rounding up.
  size_t getAccumColumnCount(llvm::MapVector<Type, size_t>& argumentCounts) {
    size_t argumentSubcomponents = 0;
    for (auto argument : argumentCounts) {
      argumentSubcomponents += argument.second;
    }
    return (argumentSubcomponents + 2) / 3;
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateAccumPass() {
  return std::make_unique<GenerateAccumPass>();
}

} // namespace dsl
} // namespace zirgen
