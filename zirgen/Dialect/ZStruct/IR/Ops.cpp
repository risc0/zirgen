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

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZStruct/IR/Ops.cpp.inc"

using namespace mlir;
using namespace zirgen::codegen;
using namespace zirgen::Zll;

namespace zirgen::ZStruct {

LogicalResult LookupOp::verify() {
  // Output val type must match named member type
  llvm::StringRef member = getMember();
  Type baseType = getBase().getType();
  auto elements =
      TypeSwitch<Type, std::optional<ArrayRef<ZStruct::FieldInfo>>>(baseType)
          .Case<StructType, LayoutType, UnionType>([&](auto ty) { return ty.getFields(); })
          .Default([&](auto) {
            emitError() << "Bad type";
            return std::nullopt;
          });
  if (!elements) {
    return failure();
  }
  Type outType = getOut().getType();
  for (auto& field : *elements) {
    if (member != field.name)
      continue;

    if (outType != field.type) {
      emitError() << "Field type " << field.type << " but out type " << outType << "\n";
      return failure();
    }
    return success();
  }
  emitError() << "Cannot find field " << member << " in " << baseType << "\n";
  return failure();
}

LogicalResult SubscriptOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location>,
                                            Adaptor adaptor,
                                            llvm::SmallVectorImpl<Type>& out) {
  return TypeSwitch<Type, LogicalResult>(adaptor.getBase().getType())
      .Case<ArrayType, LayoutArrayType>([&](auto arrayType) {
        out.push_back(arrayType.getElement());
        return success();
      })
      .Default([](auto) { return failure(); });
}

namespace {

Attribute derefConst(Operation* op, Attribute attr) {
  if (!llvm::isa_and_present<SymbolRefAttr>(attr))
    return attr;

  auto constOp =
      SymbolTable::lookupNearestSymbolFrom<GlobalConstOp>(op, llvm::cast<SymbolRefAttr>(attr));
  if (!constOp) {
    op->emitError() << "Can't find symbol " << attr << "\n";
    return {};
  }

  return constOp.getConstant();
}

Attribute resolve(Operation* op, Attribute attr) {
  while (isa<SymbolRefAttr>(attr) || isa<BoundLayoutAttr>(attr)) {
    attr = derefConst(op, attr);
    if (auto bound = dyn_cast<BoundLayoutAttr>(attr))
      attr = bound.getLayout();
  }
  return attr;
}

/// Compare two layout attributes, resolving symbol references and looking through bound layouts
bool deepCmp(Attribute lhs, Attribute rhs, Operation* op) {
  lhs = resolve(op, lhs);
  rhs = resolve(op, rhs);

  assert(isa<RefAttr>(lhs) || isa<StructAttr>(lhs) || isa<ArrayAttr>(lhs));
  assert(isa<RefAttr>(rhs) || isa<StructAttr>(rhs) || isa<ArrayAttr>(rhs));
  if (auto lRef = dyn_cast<RefAttr>(lhs); auto rRef = dyn_cast<RefAttr>(rhs)) {
    return lRef == rRef;
  } else if (auto lStr = dyn_cast<StructAttr>(lhs); auto rStr = dyn_cast<StructAttr>(rhs)) {
    for (auto [lField, rField] : llvm::zip(lStr.getFields(), rStr.getFields())) {
      if (lField.getName() != rField.getName() ||
          !deepCmp(lField.getValue(), rField.getValue(), op))
        return false;
    }
    return true;
  } else if (auto lArr = dyn_cast<ArrayAttr>(lhs); auto rArr = dyn_cast<ArrayAttr>(rhs)) {
    for (auto [lElem, rElem] : llvm::zip(lArr, rArr)) {
      if (!deepCmp(lElem, rElem, op))
        return false;
    }
    return true;
  } else {
    return false;
  }
}

} // namespace

OpFoldResult SubscriptOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getIndex())
    return nullptr;

  size_t index;
  if (auto indexAttr = dyn_cast<IntegerAttr>(adaptor.getIndex()))
    index = getIndexVal(indexAttr);
  else if (auto valAttr = dyn_cast<PolynomialAttr>(adaptor.getIndex())) {
    // TODO: Do we want to put a consistant "constant integer" type
    // here instead of having a mix of `IntegerAttr' and `PolynomialAttr'?
    index = valAttr[0];
  } else {
    return nullptr;
  }

  if (auto arrayOp = getBase().getDefiningOp<ArrayOp>()) {
    if (index < arrayOp.getElements().size()) {
      return arrayOp.getElements()[index];
    }
  }

  if (Attribute base = derefConst(*this, adaptor.getBase())) {
    if (auto arrayAttr = llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getBase())) {
      if (index < arrayAttr.getValue().size()) {
        return arrayAttr.getValue()[index];
      }
    }

    if (auto bufferAttr = dyn_cast_if_present<BoundLayoutAttr>(adaptor.getBase())) {
      if (auto base =
              dyn_cast_if_present<mlir::ArrayAttr>(derefConst(*this, bufferAttr.getLayout()))) {
        if (index < base.getValue().size()) {
          auto elem = base.getValue()[index];
          return BoundLayoutAttr::get(bufferAttr.getBuffer(), elem);
        }
      }
    }
  }

  return {};
}

mlir::IntegerAttr SubscriptOp::getIndexAsAttr() {
  Operation* indexOp = getIndex().getDefiningOp();
  SmallVector<OpFoldResult, 4> foldResults;
  if (succeeded(indexOp->fold(foldResults))) {
    assert(foldResults.size() == 1);
    if (auto attr = dyn_cast<Attribute>(foldResults[0])) {
      return cast<mlir::IntegerAttr>(attr);
    }
  }
  return nullptr;
}

size_t SubscriptOp::getIndexUpperBound() {
  Operation* indexOp = getIndex().getDefiningOp();
  if (!indexOp)
    return 0;

  // If the index is a constant, determine its exact value by folding
  SmallVector<OpFoldResult, 4> foldResults;
  if (succeeded(indexOp->fold(foldResults))) {
    assert(foldResults.size() == 1);
    if (auto attr = llvm::dyn_cast<Attribute>(foldResults[0]))
      return extractIntAttr(attr);
    else
      return 0;
  }

  // If the index is a loop induction variable, as in a for or reduce construct,
  // then determine its upper bound from the loop
  if (auto loop = dyn_cast<LoopLikeOpInterface>(indexOp)) {
    if (auto attr = loop.getSingleUpperBound()->dyn_cast<Attribute>()) {
      return extractIntAttr(attr);
    }
  }

  assert(false && "failed to deduce upper bound on a SubscriptOp's index");
  return SIZE_MAX;
}

void SubscriptOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  EmitPart indexPart = getIndex();

  // TODO: Fix types so we can guarantee a non-polynomial type in the index
  if (llvm::isa<ValType>(getIndex().getType())) {
    Attribute constVal;
    if (matchPattern(getIndex(), m_Constant(&constVal)))
      indexPart = llvm::cast<PolynomialAttr>(constVal)[0];
    else if (cg.getLanguageKind() == LanguageKind::Rust)
      indexPart = [this, &cg]() { cg << "u64::from(" << getIndex() << ") as usize"; };
    else
      indexPart = [this, &cg]() { cg << "to_size_t(" << getIndex() << ")"; };
  }

  if (isLayoutType(getBase().getType())) {
    cg.emitInvokeMacro(cg.getStringAttr("layoutSubscript"), {getBase(), indexPart});
  } else {
    cg << getBase() << "[" << indexPart << "]";
  }
}

LogicalResult SubscriptOp::verify() {
  return TypeSwitch<Type, LogicalResult>(getBase().getType())
      .Case<ArrayType, LayoutArrayType>([&](auto baseType) -> LogicalResult {
        if (getOut().getType() != baseType.getElement()) {
          return emitError() << "Output val type must match array element type";
        }
        // if (getIndexUpperBound() >= baseType.getSize()) {
        //   return emitError() << "Index must fall within bounds";
        // }
        return success();
      })
      .Default([](auto) { return failure(); });
}

LogicalResult LoadOp::verify() {
  auto inElemType = cast<RefType>(getRef().getType()).getElement();
  auto outElemType = getOut().getType();
  if (inElemType == outElemType)
    return success();

  auto inValType = llvm::dyn_cast<ValType>(inElemType);
  auto outValType = llvm::dyn_cast<ValType>(inElemType);
  if (!inValType || !outValType || inValType.getField() != outValType.getField()) {
    return emitError() << "Input ref element type must match output value type";
  }

  if (inValType.getExtended()) {
    assert(!outValType.getExtended());
    return emitError() << "Input ref element type is not coercible to output value type";
  }
  return success();
}

void LoadOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  bool resultIsExt = bool(getType().getExtended());
  bool refIsExt = bool(getRef().getType().getElement().getExtended());
  auto macroName = cg.getStringAttr(refIsExt ? "load_ext" : (resultIsExt ? "load_as_ext" : "load"));
  OpBuilder builder(getContext());
  cg.emitInvokeMacro(macroName, {getRef(), getDistance()});
}

LogicalResult StoreOp::verify() {
  if (cast<RefType>(getRef().getType()).getElement() != getVal().getType()) {
    return emitError() << "Source value type must match destination ref element type";
  }
  return success();
}

void StoreOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  if (getVal().getType().getFieldK() > 1)
    cg.emitInvokeMacro(cg.getStringAttr("storeExt"), {getRef(), getVal()});
  else
    cg.emitInvokeMacro(cg.getStringAttr("store"), {getRef(), getVal()});
}

LogicalResult PackOp::verify() {
  auto fields = getOut().getType().getFields();
  if (fields.size() != getNumOperands()) {
    return emitError() << "Expected " << fields.size() << " arguments to generate a "
                       << getOut().getType();
  }

  for (auto [f, arg] : llvm::zip_first(fields, getMembers())) {
    auto argType = arg.getType();
    if (f.type != argType) {
      return emitError() << "Trying to construct member " << f.name << " from " << argType
                         << " instead of " << f.type;
    }
  }

  return success();
}

OpFoldResult PackOp::fold(FoldAdaptor adaptor) {
  auto fields = getType().getFields();
  if (fields.size() != adaptor.getMembers().size())
    return {};

  SmallVector<NamedAttribute> fieldVals;
  for (auto [field, arg] : llvm::zip(getType().getFields(), adaptor.getMembers())) {
    if (!arg)
      return {};

    fieldVals.emplace_back(field.name, arg);
  }

  return StructAttr::get(getContext(), DictionaryAttr::get(getContext(), fieldVals), getType());
}

namespace {

struct LookupLayoutPattern : public OpRewritePattern<LookupOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LookupOp op, PatternRewriter& rewriter) const override {
    auto bindLayout = op.getBase().getDefiningOp<BindLayoutOp>();
    if (!bindLayout)
      return rewriter.notifyMatchFailure(op, "layout not from BindLayoutOp");

    auto layoutAttr = llvm::dyn_cast<StructAttr>(bindLayout.getLayout());
    if (!layoutAttr)
      return rewriter.notifyMatchFailure(op, "Layout not StructAttr");

    auto newLayout = layoutAttr.getFields().get(op.getMember());
    if (!newLayout)
      return rewriter.notifyMatchFailure(op, "Missing layout member");

    rewriter.replaceOpWithNewOp<BindLayoutOp>(op, op.getType(), newLayout, bindLayout.getBuffer());

    return success();
  }
};

struct SubscriptLayoutPattern : public OpRewritePattern<SubscriptOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubscriptOp op, PatternRewriter& rewriter) const override {
    Attribute indexValue;
    if (!matchPattern(op.getIndex(), m_Constant(&indexValue)))
      return rewriter.notifyMatchFailure(op, "index not a constant");
    size_t idx = extractIntAttr(indexValue);

    auto bindLayout = op.getBase().getDefiningOp<BindLayoutOp>();
    if (!bindLayout)
      return rewriter.notifyMatchFailure(op, "layout not from BindLayoutOp");

    auto layoutAttr = llvm::dyn_cast<ArrayAttr>(bindLayout.getLayout());
    if (!layoutAttr)
      return rewriter.notifyMatchFailure(op, "Layout not ArrayAttr");

    if (idx > layoutAttr.size())
      return rewriter.notifyMatchFailure(op, "index out of range");
    auto newLayout = layoutAttr[idx];
    rewriter.replaceOpWithNewOp<BindLayoutOp>(op, op.getType(), newLayout, bindLayout.getBuffer());

    return success();
  }
};

struct LoadLayoutPattern : public OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LoadOp op, PatternRewriter& rewriter) const override {
    Attribute distanceValue;
    if (!matchPattern(op.getDistance(), m_Constant(&distanceValue)))
      return rewriter.notifyMatchFailure(op, "distance not a constant");
    size_t distance = extractIntAttr(distanceValue);

    auto bindLayout = op.getRef().getDefiningOp<BindLayoutOp>();
    if (!bindLayout)
      return rewriter.notifyMatchFailure(op, "layout not from BindLayoutOp");

    auto layoutAttr = llvm::dyn_cast<RefAttr>(bindLayout.getLayout());
    if (!layoutAttr)
      return rewriter.notifyMatchFailure(op, "Layout not RefAttr");
    size_t offset = layoutAttr.getIndex();

    auto bufType = llvm::dyn_cast<BufferType>(bindLayout.getBuffer().getType());
    if (!bufType)
      return rewriter.notifyMatchFailure(op, "binding not to a buffer");

    auto valType = llvm::cast<ValType>(op.getType());

    // zll.get doesn't support reading extension field elements,
    // so read each part one at a time and multiply them out.
    Value result;
    Value shiftOnce;

    for (size_t idx : llvm::reverse(llvm::seq(0u, valType.getFieldK()))) {
      Value val;
      if (bufType.getKind() == BufferKind::Global) {
        if (distance)
          return rewriter.notifyMatchFailure(op, "Cannot take back on global");
        else
          val = rewriter.create<Zll::GetGlobalOp>(
              op.getLoc(), bufType.getElement(), bindLayout.getBuffer(), offset + idx);
      } else {
        auto getOp = rewriter.create<Zll::GetOp>(op.getLoc(),
                                                 bufType.getElement(),
                                                 bindLayout.getBuffer(),
                                                 offset + idx,
                                                 distance,
                                                 /*optional tap=*/IntegerAttr{});
        val = getOp;
        if (op->getAttr("unchecked"))
          getOp->setAttr("unchecked", rewriter.getUnitAttr());
      }
      if (result) {
        if (!shiftOnce) {
          auto shiftAttr = rewriter.getAttr<PolynomialAttr>(ArrayRef<uint64_t>{0, 1, 0, 0});
          shiftOnce = rewriter.create<Zll::ConstOp>(op.getLoc(), shiftAttr);
        }

        result = rewriter.create<Zll::MulOp>(op.getLoc(), shiftOnce, result);
        result = rewriter.create<Zll::AddOp>(op.getLoc(), val, result);
      } else {
        result = val;
      }
    }
    assert(result && "no elements in extension field?");
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct StoreLayoutPattern : public OpRewritePattern<StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(StoreOp op, PatternRewriter& rewriter) const override {
    auto bindLayout = op.getRef().getDefiningOp<BindLayoutOp>();
    if (!bindLayout)
      return rewriter.notifyMatchFailure(op, "layout not from BindLayoutOp");

    auto layoutAttr = llvm::dyn_cast<RefAttr>(bindLayout.getLayout());
    if (!layoutAttr)
      return rewriter.notifyMatchFailure(op, "Layout not RefAttr");
    size_t offset = layoutAttr.getIndex();

    auto bufType = llvm::dyn_cast<BufferType>(bindLayout.getBuffer().getType());
    if (!bufType)
      return rewriter.notifyMatchFailure(op, "binding not to a buffer");

    if (bufType.getKind() == BufferKind::Global) {
      rewriter.replaceOpWithNewOp<Zll::SetGlobalOp>(
          op, bindLayout.getBuffer(), offset, op.getVal());
    } else {
      rewriter.replaceOpWithNewOp<Zll::SetOp>(op, bindLayout.getBuffer(), offset, op.getVal());
    }

    return success();
  }
};

} // namespace

void LookupOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<LookupLayoutPattern>(context);
}

void SubscriptOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<SubscriptLayoutPattern>(context);
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<LoadLayoutPattern>(context);
}

void StoreOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<StoreLayoutPattern>(context);
}

LogicalResult LookupOp::inferReturnTypes(MLIRContext* ctx,
                                         std::optional<Location>,
                                         Adaptor adaptor,
                                         llvm::SmallVectorImpl<Type>& out) {
  auto inType = adaptor.getBase().getType();
  return TypeSwitch<Type, LogicalResult>(inType)
      .Case<StructType, LayoutType, UnionType>([&](auto ty) {
        auto name = adaptor.getMemberAttr();
        for (const FieldInfo& member : ty.getFields()) {
          if (member.name == name) {
            out.push_back(member.type);
            break;
          }
        }
        if (out.empty()) {
          return failure();
        } else {
          return success();
        }
      })
      .Default([&](auto) { return failure(); });
}

OpFoldResult LookupOp::fold(FoldAdaptor adaptor) {
  Value foldedValue;

  TypeSwitch<mlir::Type>(getBase().getType()).Case<StructType, LayoutType>([&](auto ty) {
    auto fields = ty.getFields();

    if (auto newOp = getBase().getDefiningOp<PackOp>()) {
      if (fields.size() == newOp->getNumOperands()) {
        for (size_t i = 0; i != newOp->getNumOperands(); ++i) {
          if (fields[i].name == getMember()) {
            foldedValue = newOp->getOperand(i);
          }
        }
      }
    }
  });

  if (foldedValue)
    return foldedValue;

  if (Attribute base = derefConst(*this, adaptor.getBase())) {
    if (auto structAttr = dyn_cast_if_present<StructAttr>(adaptor.getBase())) {
      return structAttr.getFields().get(getMember());
    }

    if (auto bufferAttr = dyn_cast_if_present<BoundLayoutAttr>(adaptor.getBase())) {
      if (auto structAttr =
              dyn_cast_if_present<StructAttr>(derefConst(*this, bufferAttr.getLayout()))) {
        auto member = structAttr.getFields().get(getMember());
        return BoundLayoutAttr::get(bufferAttr.getBuffer(), member);
      }
    }
  }
  return {};
}

void LookupOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  // If we're looking up a path, put it all together at once.  This is
  // especially useful for layouts since we only have to call
  // layoutLookup once.
  SmallVector<CodegenIdent<IdentKind::Field>> path;
  LookupOp op = *this;
  Value lastBase;
  while (op) {
    lastBase = op.getBase();
    path.push_back(op.getMemberAttr());
    op = op.getBase().getDefiningOp<LookupOp>();
  }

  if (isLayoutType(getBase().getType())) {

    cg.emitInvokeMacro(cg.getStringAttr("layoutLookup"),
                       {lastBase, [&]() {
                          llvm::interleave(
                              llvm::reverse(path),
                              *cg.getOutputStream(),
                              [&](auto pathElem) { cg << pathElem; },
                              ".");
                        }});
  } else {
    cg << lastBase << ".";
    llvm::interleave(
        llvm::reverse(path), *cg.getOutputStream(), [&](auto pathElem) { cg << pathElem; }, ".");
  }
}

void PackOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  auto names = llvm::to_vector_of<CodegenIdent<IdentKind::Field>>(
      llvm::map_range(getType().getFields(), [&](auto field) { return field.name; }));
  auto values = llvm::to_vector_of<CodegenValue>(
      llvm::map_range(getMembers(), [&](auto member) { return CodegenValue(member).owned(); }));
  cg.emitStructConstruct(getType(), names, values);
}

void SwitchOp::emitStatement(zirgen::codegen::CodegenEmitter& cg) {
  SmallVector<CodegenValue> selectors;
  SmallVector<Block*> arms;
  for (size_t i = 0; i != getArms().size(); ++i) {
    selectors.push_back(getSelector()[i]);
    arms.push_back(&getArms()[i].front());
  }
  cg.emitSwitchStatement({getOut()}, selectors, arms);
}

LogicalResult SwitchOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor) {
  size_t numPresent = 0;
  size_t presentIndex = 0;
  for (size_t i = 0; i != adaptor.getSelector().size(); ++i) {
    auto selector = adaptor.getSelector()[i]->getVal();
    if (selector[0] != 0) {
      ++numPresent;
      presentIndex = i;

      if (selector[0] != 1) {
        return emitError() << "Invalid value " << selector << " in selector";
      }
    }
  }
  if (numPresent != 1) {
    return emitError() << "Mux must have 1 active arm but has " << numPresent;
  }

  auto interpOuts = interp.runBlock(getArms()[presentIndex].front());
  if (failed(interpOuts))
    return failure();
  for (auto [out, interpOut] : llvm::zip(outs, *interpOuts)) {
    out->setAttr(interpOut);
  }
  return success();
}

LogicalResult SwitchOp::verify() {
  if (getRegions().size() != getSelector().size())
    return emitOpError() << "switch op selector count " << getSelector().size()
                         << " must match arm count " << getArms().size() << "\n";

  if (getRegions().empty()) {
    return emitOpError() << "switch must contain at least one mux arm";
  }

  Type outType = getOut().getType();
  for (auto arm : getRegions()) {
    auto termOp = arm->front().getTerminator();
    if (!termOp || !termOp->hasTrait<OpTrait::ReturnLike>())
      return emitOpError() << "Mux arm missing `return'";

    if (termOp->getOperandTypes()[0] != outType)
      return emitOpError() << "Mux arm return type " << termOp->getOperandTypes() << " must match "
                           << outType;
  }

  return success();
}

namespace {

// If the selector of a mux is a compile time constant, replace it with the
// active arm and delete the unreachable ones.
struct RemoveStaticCondition : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const override {
    size_t numTrue = 0;
    size_t trueIndex;
    for (size_t i = 0; i < op.getSelector().size(); i++) {
      Value sel = op.getSelector()[i];
      Attribute constVal;
      if (matchPattern(sel, m_Constant(&constVal))) {
        auto v = llvm::cast<PolynomialAttr>(constVal);
        bool selTrue = llvm::any_of(v.asArrayRef(), [](auto elem) { return elem != 0; });
        if (selTrue) {
          numTrue++;
          trueIndex = i;
        }
      }
    }

    if (numTrue > 1)
      return op.emitError() << "SwitchOps should have a single true selector; this one has "
                            << numTrue;

    if (numTrue == 1) {
      // We can safely delete runtime side effects like witness generation and
      // constraints, but we need to keep AliasLayoutOps around until layout
      // generation.
      auto walkResult = op.walk([](AliasLayoutOp) { return WalkResult::interrupt(); });
      if (walkResult.wasInterrupted())
        return failure();

      // We found the one true region; inline into parent.
      auto* region = op.getRegions()[trueIndex];
      assert(region->hasOneBlock() && "expected single-region block");
      Block* block = &region->front();
      Operation* terminator = block->getTerminator();
      ValueRange results = terminator->getOperands();
      rewriter.inlineBlockBefore(block, op);
      rewriter.replaceOp(op, results);
      rewriter.eraseOp(terminator);
      return success();
    }

    return failure();
  }
};

} // namespace

void SwitchOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<RemoveStaticCondition>(context);
}

LogicalResult YieldOp::evaluate(Interpreter& interp,
                                llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                EvalAdaptor& adaptor) {
  interp.setResultValues({adaptor.getValue()->getAttr(getContext())});
  return success();
}

OpFoldResult ArrayOp::fold(FoldAdaptor adaptor) {
  return ArrayAttr::get(getContext(), adaptor.getElements());
}

void ArrayOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  cg.emitArrayConstruct(getType(),
                        getType().getElement(),
                        llvm::to_vector_of<CodegenValue>(llvm::map_range(
                            getElements(), [&](auto elem) { return CodegenValue(elem).owned(); })));
}

LogicalResult ArrayOp::inferReturnTypes(MLIRContext* ctx,
                                        std::optional<Location>,
                                        Adaptor adaptor,
                                        llvm::SmallVectorImpl<Type>& out) {
  // The array elements are already the same type thanks to SameTypeOperands
  Type elemType = adaptor.getElements().front().getType();
  out.push_back(ArrayType::get(ctx, elemType, adaptor.getElements().size()));
  return success();
}

void MapOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  auto layout = getLayout() ? std::optional<CodegenValue>(getLayout()) : std::nullopt;
  cg.emitMapConstruct(getArray(), layout, getBody());
}

LogicalResult MapOp::verifyRegions() {
  size_t numArgs = getBody().getNumArguments();
  if (numArgs != 1 && numArgs != 2)
    return emitOpError() << "body should have one or two arguments, but has " << numArgs;

  BlockArgument bodyElem = getBody().getArgument(0);
  if (bodyElem.getType() != getArray().getType().getElement()) {
    return emitOpError() << "wrong type of arguments in body, array is " << getArray().getType()
                         << " but induction var is " << bodyElem.getType();
  }

  if (getLayout()) {
    BlockArgument layoutElem = getBody().getArgument(1);
    if (layoutElem.getType() != cast<LayoutArrayType>(getLayout().getType()).getElement()) {
      return emitOpError() << "wrong type of layout argument in body, array is "
                           << getLayout().getType() << " but induction var is "
                           << layoutElem.getType();
    }
  }

  return success();
}

void ReduceOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  auto layout = getLayout() ? std::optional<CodegenValue>(getLayout()) : std::nullopt;
  cg.emitReduceConstruct(getArray(), getInit(), layout, getBody());
}

LogicalResult ReduceOp::verifyRegions() {
  size_t numArgs = getBody().getNumArguments();
  if (numArgs != 2 && numArgs != 3)
    return emitOpError() << "body should have two or three arguments, but has " << numArgs;

  BlockArgument accum = getBody().getArgument(0);
  if (accum.getType() != getInit().getType()) {
    return emitOpError() << "wrong type of accumulator argument in body, init is "
                         << getInit().getType() << " but accumulator induction var is "
                         << accum.getType();
  }

  BlockArgument elem = getBody().getArgument(1);
  if (elem.getType() != getArray().getType().getElement()) {
    return emitOpError() << "wrong type of element argument in body, array is "
                         << getArray().getType() << " but induction var is " << elem.getType();
  }

  return success();
}

void GlobalConstOp::emitGlobal(codegen::CodegenEmitter& cg) {
  cg.emitConstDef(getSymNameAttr(), CodegenValue(getType(), getConstant()));
}

void GlobalConstOp::emitGlobalDecl(codegen::CodegenEmitter& cg) {
  cg.emitConstDecl(getSymNameAttr(), getType());
}

LogicalResult LoadOp::evaluate(Interpreter& interp,
                               llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                               EvalAdaptor& adaptor) {
  auto layoutAttr = adaptor.getRef()->getAttr<BoundLayoutAttr>();
  auto bufName = layoutAttr.getBuffer();
  auto buf = interp.getNamedBuf(bufName);
  size_t offset = llvm::cast<RefAttr>(layoutAttr.getLayout()).getIndex();
  size_t distance = adaptor.getDistance()->getAttr<IntegerAttr>().getInt();
  size_t size = interp.getNamedBufSize(bufName);
  if (size == 0 && distance != 0) {
    return emitError() << "Cannot take back from global";
  }

  if (distance > interp.getCycle() && !interp.getTotCycles()) {
    // TODO: Change this back to a throw once the DSL works enough that we can
    // avoid reading back too far.
    // throw std::runtime_error("Attempt to read back too far");
    llvm::errs() << "WARNING: attempt to read back too far\n";
    outs[0]->setVal(0);
    return success();
  }
  size_t totOffset = size * interp.getBackCycle(distance) + offset;
  if (totOffset >= buf.size()) {
    return emitError() << "Attempting to get out of bounds index " << totOffset
                       << " from buffer of size " << buf.size();
  }

  size_t refK = getRef().getType().getElement().getFieldK();
  Interpreter::Polynomial val;
  llvm::append_range(
      val, llvm::map_range(buf.slice(totOffset, refK), [&](auto elem) { return elem[0]; }));
  if (isInvalid(val)) {
    if (interp.getTotCycles() || !getOperation()->hasAttr("unchecked")) {

      auto diag = emitError() << "LoadOp: Read before write " << bufName << "[" << offset << "]@"
                              << distance;
      Operation* op = getOperation()->getParentOp();
      while (op) {
        diag.attachNote(op->getLoc()) << "contained here";
        op = op->getParentOp();
      }
      return diag;
    }
    outs[0]->setVal(0);
  } else {
    outs[0]->setVal(val);
  }
  return success();
}

LogicalResult StoreOp::evaluate(Interpreter& interp,
                                llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                EvalAdaptor& adaptor) {
  auto layoutAttr = adaptor.getRef()->getAttr<BoundLayoutAttr>();
  auto bufName = layoutAttr.getBuffer();
  auto buf = interp.getNamedBuf(bufName);
  size_t offset = llvm::cast<RefAttr>(layoutAttr.getLayout()).getIndex();
  size_t size = interp.getNamedBufSize(bufName);
  size_t totOffset = size * interp.getCycle() + offset;
  if (totOffset >= buf.size()) {
    return emitError() << "Attempting to set out of bounds index " << totOffset
                       << " in buffer of size " << buf.size();
  }
  size_t refK = getRef().getType().getElement().getFieldK();
  Interpreter::PolynomialRef newVal = adaptor.getVal()->getVal();
  for (size_t i = 0; i < refK; i++) {
    Interpreter::Polynomial newElem(1);
    if (i >= newVal.size())
      newElem[0] = 0;
    else
      newElem[0] = newVal[i];

    Interpreter::Polynomial& oldElem = buf[totOffset + i];
    if (!isInvalid(oldElem) && oldElem != newElem) {
      return emitError() << "StoreOp: Invalid set of " << bufName << "[" << offset << " + " << i
                         << "], cur=" << oldElem << ", new = " << newVal;
    }
    oldElem = newElem;
  }
  return success();
}

LogicalResult BindLayoutOp::evaluate(Interpreter& interp,
                                     llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                     EvalAdaptor& adaptor) {
  auto getBufferOp = getBuffer().getDefiningOp<GetBufferOp>();
  if (!getBufferOp)
    return emitError() << "Missing buffer";

  Attribute layoutAttr = getLayoutAttr();
  if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(layoutAttr)) {
    // Look up by symbol
    auto glob = SymbolTable::lookupNearestSymbolFrom<GlobalConstOp>(*this, symAttr);
    if (!glob) {
      return emitError() << "Unable to find symbol " << getLayout() << "\n";
    }
    layoutAttr = glob.getConstant();
  }

  outs[0]->setAttr(BoundLayoutAttr::get(getBufferOp.getNameAttr(), layoutAttr));

  return success();
}

void BindLayoutOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  auto symAttr = llvm::cast<FlatSymbolRefAttr>(getLayoutAttr());
  auto globOp = SymbolTable::lookupNearestSymbolFrom<GlobalConstOp>(*this, symAttr);
  assert(globOp);
  cg.emitInvokeMacro(cg.getStringAttr("bind_layout"),
                     {CodegenIdent<IdentKind::Const>(symAttr.getAttr()), getBuffer()});
}

LogicalResult BindLayoutOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  if (auto symAttr = llvm::dyn_cast<FlatSymbolRefAttr>(getLayoutAttr())) {
    auto globalConstOp = symbolTable.lookupNearestSymbolFrom<GlobalConstOp>(*this, symAttr);
    if (!globalConstOp)
      return emitOpError() << "Cannot find global constant " << getLayoutAttr();
    if (globalConstOp.getType() != getType())
      return emitOpError() << "Global symbol " << getLayoutAttr() << " type "
                           << globalConstOp.getType() << " does not match expected " << getType();
  }
  return success();
}

void GetBufferOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(cg.getStringAttr("get_buffer"),
                     /*ContextArgs=*/{"ctx"},
                     {codegen::CodegenIdent<codegen::IdentKind::Var>(getNameAttr())});
}

LogicalResult GetBufferOp::evaluate(Zll::Interpreter& interp,
                                    llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                    EvalAdaptor& adaptor) {
  // TODO: Make the semantics be the same no matter what mode we're evaluating in.
  if (interp.hasNamedBuf(adaptor.getName()))
    outs[0]->setBuf(interp.getNamedBuf(adaptor.getName()));
  else
    outs[0]->setAttr(adaptor.getNameAttr());
  return success();
}

LogicalResult AliasLayoutOp::evaluate(Zll::Interpreter& interp,
                                      llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                      EvalAdaptor& adaptor) {
  Attribute lhs = adaptor.getLhs()->getAttr();
  Attribute rhs = adaptor.getRhs()->getAttr();
  if (!deepCmp(lhs, rhs, *this)) {
    return emitError() << "AliasLayoutOp: layout guarantee not satisfied";
  }
  return success();
}

LogicalResult LoadOp::inferReturnTypes(MLIRContext* ctx,
                                       std::optional<Location>,
                                       Adaptor adaptor,
                                       llvm::SmallVectorImpl<Type>& out) {
  auto refType = llvm::dyn_cast<RefType>(adaptor.getRef().getType());
  if (!refType)
    return failure();

  out.push_back(refType.getElement());
  return success();
}

} // namespace zirgen::ZStruct
