// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Utilities/KeyPath.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

namespace zirgen {
namespace dsl {

struct RandomnessMap {
  RandomnessMap() = default;
  RandomnessMap(OpBuilder& builder, Value pivot, Value zeroDistance) : pivot(pivot) {
    ValType extValType = Zhlt::getValExtType(builder.getContext());

    if (pivot.getType() == Zhlt::getExtRefType(pivot.getContext())) {
      this->pivot = builder.create<LoadOp>(pivot.getLoc(), extValType, pivot, zeroDistance);
    } else if (auto pivotType = dyn_cast<LayoutType>(pivot.getType())) {
      for (auto field : pivotType.getFields()) {
        if (field.name == "$offset")
          continue;

        Value member = builder.create<LookupOp>(pivot.getLoc(), pivot, field.name);
        map.insert({field.name, RandomnessMap(builder, member, zeroDistance)});
      }
    } else if (auto pivotType = dyn_cast<LayoutArrayType>(pivot.getType())) {
      for (size_t i = 0; i < pivotType.getSize(); i++) {
        Value index = builder.create<arith::ConstantOp>(pivot.getLoc(), builder.getIndexAttr(i));
        Value element = builder.create<SubscriptOp>(pivot.getLoc(), pivot, index);
        map.insert({i, RandomnessMap(builder, element, zeroDistance)});
      }
    } else {
      llvm::errs() << "unhandled case: " << pivot.getType();
      abort();
    }
  }

  Value pivot;
  std::map<Key, RandomnessMap> map;
};

class AccumBuilder {
public:
  AccumBuilder(OpBuilder& builder, Value accumLayout, Value randomnessLayout)
      : builder(builder), accumLayout(accumLayout) {
    // Preemptively lookup all verifier randomness values and store them in a
    // dictionary. Unneeded values will be pruned later by folding.
    zeroDistance =
        builder.create<arith::ConstantOp>(randomnessLayout.getLoc(), builder.getIndexAttr(0));
    ValType extValType = Zhlt::getValExtType(builder.getContext());

    verifierRandomness = RandomnessMap(builder, randomnessLayout, zeroDistance);

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
  // Since an argument can have a non-trivial layout, define the condensation
  // inductively on the layout structure. The resulting sum can be flattened to
  // an expression like the one above.
  Value condenseArgument(Value layout, const RandomnessMap& randomness) {
    MLIRContext* ctx = builder.getContext();
    ValType extValType = Zhlt::getValExtType(ctx);

    if (layout.getType() == Zhlt::getNondetRegLayoutType(ctx)) {
      // For a single register, compute r_a * a[i]
      assert(randomness.pivot.getType() == extValType);
      Value r = randomness.pivot;
      Value colLayout = builder.create<LookupOp>(layout.getLoc(), layout, "@super");
      Value colValue =
          builder.create<ZStruct::LoadOp>(colLayout.getLoc(), extValType, colLayout, zeroDistance);
      return builder.create<MulOp>(colValue.getLoc(), r, colValue);
    } else if (auto layoutType = dyn_cast<LayoutType>(layout.getType())) {
      // for a LayoutType, sum condensations of non-count fields
      auto fields = layoutType.getFields();
      if (layoutType.getKind() == LayoutKind::Argument)
        fields = fields.drop_front();

      SmallVector<uint64_t> zero(extValType.getFieldK(), 0);
      Value v = builder.create<ConstOp>(layout.getLoc(), extValType, zero);
      for (auto field : fields) {
        Value sublayout = builder.create<LookupOp>(layout.getLoc(), layout, field.name);
        Value sumTerm = condenseArgument(sublayout, randomness.map.at(field.name.getValue()));
        v = builder.create<AddOp>(sumTerm.getLoc(), v, sumTerm);
      }
      return v;
    } else if (auto layoutType = dyn_cast<LayoutArrayType>(layout.getType())) {
      // For a LayoutArrayType, sum condensations at all subscripts
      SmallVector<uint64_t> zero(extValType.getFieldK(), 0);
      Value v = builder.create<ConstOp>(layout.getLoc(), extValType, zero);
      for (size_t i = 0; i < layoutType.getSize(); i++) {
        Value index = builder.create<arith::ConstantOp>(layout.getLoc(), builder.getIndexAttr(i));
        Value sublayout = builder.create<SubscriptOp>(layout.getLoc(), layout, index);
        Value sumTerm = condenseArgument(sublayout, randomness.map.at(i));
        v = builder.create<AddOp>(sumTerm.getLoc(), v, sumTerm);
      }
      return v;
    } else {
      llvm_unreachable("unhandled layout when condensing argument");
      return nullptr;
    }
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
    MLIRContext* ctx = builder.getContext();
    ValType extValType = Zhlt::getValExtType(builder.getContext());
    StringAttr countName = type.getFields()[0].name;
    Value cLayout = builder.create<LookupOp>(layout.getLoc(), layout, countName);
    cLayout = Zhlt::coerceTo(cLayout, Zhlt::getRefType(ctx), builder);
    Value c = builder.create<ZStruct::LoadOp>(cLayout.getLoc(), extValType, cLayout, zeroDistance);
    Value v = condenseArgument(layout, verifierRandomness.map.at(type.getId()));
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
  RandomnessMap verifierRandomness;

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
      // Generate accum code for entry points like Top and tests, but don't get
      // stuck in a loop generating more accum code from the new accum code.
      llvm::StringRef baseName = component.getName();
      if (baseName.ends_with("$accum") || !Zhlt::isEntryPoint(component))
        return;

      auto layoutType = component.getLayoutType();
      if (!layoutType)
        return;

      KeyPath keyPath, cur;
      LayoutType majorType;
      int count = findMajorMux(majorType, keyPath, layoutType, cur);
      if (count == 0) {
        return;
      }
      if (count != 1) {
        llvm::errs() << "Unable to find unique major mux for " << baseName << "\n";
        return signalPassFailure();
      }
      buildAccumStep(component, keyPath, majorType);
    });
  }

  int findMajorMux(LayoutType& outType, KeyPath& outPath, Type in, KeyPath& cur) {
    int tot = 0;
    if (auto array = dyn_cast<LayoutArrayType>(in)) {
      cur.push_back(size_t(0));
      tot = findMajorMux(outType, outPath, array.getElement(), cur);
      cur.pop_back();
      tot *= array.getSize();
      return std::min(tot, 2);
    }
    if (auto layout = dyn_cast<LayoutType>(in)) {
      switch (layout.getKind()) {
      case LayoutKind::Argument:
        return 2;
      case LayoutKind::Mux:
        return 0;
      case LayoutKind::MajorMux:
        outType = layout;
        outPath = cur;
        return 1;
      case LayoutKind::Normal:
        for (auto& field : layout.getFields()) {
          cur.push_back(field.name);
          tot += findMajorMux(outType, outPath, field.type, cur);
          cur.pop_back();
        }
        return tot;
      }
    }
    return 0;
  }

  void buildAccumStep(ComponentOp component, KeyPath keyPath, LayoutType& majorType) {
    // Get the worst case column count
    size_t columns = 0;
    for (auto& field : majorType.getFields()) {
      if (field.name.getValue().starts_with("arm")) {
        llvm::MapVector<Type, size_t> argCounts;
        extractArguments(argCounts, field.type);
        columns = std::max(columns, getAccumColumnCount(argCounts));
      }
    }

    auto ctx = component->getContext();
    auto loc = component->getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(getOperation().getBody());

    // Create Accum component, which takes the Top layout as a parameter and the
    // accum buffer as its layout.
    SmallVector<Type> accumParams = {component.getLayoutType()};
    auto accumLayoutType = LayoutArrayType::get(ctx, Zhlt::getExtRefType(ctx), columns);
    std::string accumName = (component.getName() + "$accum").str();
    auto accum = builder.create<Zhlt::ComponentOp>(
        loc, accumName, Zhlt::getComponentType(ctx), accumParams, accumLayoutType);
    SymbolTable::setSymbolVisibility(accum, SymbolTable::Visibility::Public);
    builder.setInsertionPointToStart(accum.addEntryBlock());

    // Create globals for verifier randomness
    LayoutType mixLayoutType = getRandomnessLayoutType();
    auto randomnessLayout =
        builder.create<Zhlt::GetGlobalLayoutOp>(loc, mixLayoutType, "mix", "randomness");

    // Walk down the key path to the major mux layout component
    Value cur = accum.getConstructParam()[0];
    for (const Key& key : keyPath) {
      if (auto* strKey = std::get_if<mlir::StringRef>(&key)) {
        cur = builder.create<LookupOp>(loc, cur, *strKey);
      } else {
        Value index =
            builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(std::get<size_t>(key)));
        cur = builder.create<SubscriptOp>(loc, cur, index);
      }
    }

    // Load from the list of saved selectors
    Value selectorLayoutArray = builder.create<LookupOp>(loc, cur, "@selector");
    size_t armCount = selectorLayoutArray.getType().dyn_cast<LayoutArrayType>().getSize();
    SmallVector<Value> selectors;
    Value zeroDistance = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    ValType valType = Zhlt::getValType(builder.getContext());
    for (size_t i = 0; i < armCount; i++) {
      Value index = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(i));
      Value nondetReg = builder.create<SubscriptOp>(loc, selectorLayoutArray, index);
      Value ref = builder.create<LookupOp>(loc, nondetReg, "@super");
      Value val = builder.create<LoadOp>(loc, valType, ref, zeroDistance);
      selectors.push_back(val);
    }

    // Build a switch op over the (saved) selectors
    Type componentType = Zhlt::getComponentType(builder.getContext());
    auto switchOp =
        builder.create<ZStruct::SwitchOp>(loc, componentType, selectors, /*numArms=*/armCount);

    // Now, generate each arm
    for (size_t i = 0; i < armCount; i++) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      Block* block = builder.createBlock(&switchOp.getArms()[i]);
      builder.setInsertionPointToEnd(block);
      std::string armName = "arm" + std::to_string(i);
      bool hasArm = std::any_of(majorType.getFields().begin(),
                                majorType.getFields().end(),
                                [&](auto field) { return field.name == armName; });
      if (hasArm) {
        Value armLayout = builder.create<LookupOp>(loc, cur, armName);
        AccumBuilder accumBuilder(builder, accum.getLayout(), randomnessLayout);
        accumBuilder.build(armLayout);
        accumBuilder.finalize();
      } else {
        // Read accumulator + forward
        ValType extValType = Zhlt::getValExtType(builder.getContext());
        Value distance =
            builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
        Value indexValue = builder.create<arith::ConstantOp>(
            loc, builder.getIndexType(), builder.getIndexAttr(accumLayoutType.getSize() - 1));
        Value lastLayout = builder.create<SubscriptOp>(loc, accum.getLayout(), indexValue);
        Value prevVal = builder.create<LoadOp>(loc, extValType, lastLayout, distance);
        builder.create<StoreOp>(loc, lastLayout, prevVal);
      }

      Value empty = builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), ValueRange{});
      builder.create<ZStruct::YieldOp>(loc, empty);
    }

    // Make a null return
    Value empty = builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), ValueRange{});
    builder.create<Zhlt::ReturnOp>(loc, empty);
  }

private:
  LayoutType getRandomnessLayoutTypeFor(LayoutType type, bool isRoot) {
    MLIRContext* ctx = &getContext();
    RefType extType = Zhlt::getExtRefType(ctx);

    // Add global randomness for all non-count members of type
    auto fields = type.getFields();
    if (isRoot)
      fields = fields.drop_front();

    SmallVector<ZStruct::FieldInfo> members;
    for (auto field : fields) {
      if (field.type == Zhlt::getNondetRegLayoutType(ctx)) {
        members.push_back({field.name, extType});
      } else if (auto fieldType = dyn_cast<LayoutType>(field.type)) {
        members.push_back({field.name, getRandomnessLayoutTypeFor(fieldType, false)});
      } else if (auto fieldType = dyn_cast<LayoutArrayType>(field.type)) {
        LayoutType elementType =
            getRandomnessLayoutTypeFor(cast<LayoutType>(fieldType.getElement()), false);
        members.push_back(
            {field.name, LayoutArrayType::get(ctx, elementType, fieldType.getSize())});
      } else {
        llvm_unreachable("unhandled case");
      }
    }
    auto fieldTypeName = StringAttr::get(ctx, "arg$" + type.getId());
    return LayoutType::get(ctx, fieldTypeName, members);
  }

  LayoutType getRandomnessLayoutType() {
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

      auto fieldName = StringAttr::get(ctx, type.getId());
      members.push_back({fieldName, getRandomnessLayoutTypeFor(type, true)});
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
