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
#include "mlir/IR/IRMapping.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/circuit/verify/wrap.h"

using zirgen::Val;
using zirgen::verify::CircuitBase;
using zirgen::verify::MixState;

namespace zirgen::verify {

using namespace zirgen::Zhlt;
using namespace zirgen::Zll;
using namespace mlir;

class CircuitInterfaceZirgen : public CircuitInterface {
public:
  CircuitInterfaceZirgen(OwningOpRef<mlir::ModuleOp> modArg, ProtocolInfo protocolInfo)
      : ownedMod(std::move(modArg)), bufferAnalysis(mod()), protocolInfo(protocolInfo) {
    initialize();
  }
  const Zll::TapSet& get_taps() const override { return tapSet; }
  Val compute_poly(llvm::ArrayRef<Val> u,
                   llvm::ArrayRef<Val> out,
                   llvm::ArrayRef<Val> accumMix,
                   Val polyMix) const override;
  size_t out_size() const override { return bufferAnalysis.getBuffer("global").regCount; }
  size_t mix_size() const override { return bufferAnalysis.getBuffer("mix").regCount; }
  ProtocolInfo get_circuit_info() const override { return protocolInfo; }

private:
  void initialize();
  void add_taps();

  // ModuleOp's methods aren't very const safe, so operate on a copy instead of dereferencing
  // ownedMod.
  mlir::ModuleOp mod() const { return ModuleOp(*ownedMod); }

private:
  Zll::TapSet tapSet;

  OwningOpRef<mlir::ModuleOp> ownedMod;
  ZStruct::BufferAnalysis bufferAnalysis;
  Zhlt::ValidityTapsFuncOp validityTaps;
  ProtocolInfo protocolInfo;
};

void CircuitInterfaceZirgen::initialize() {
  add_taps();

  validityTaps = mod().lookupSymbol<Zhlt::ValidityTapsFuncOp>(ValidityTapsFuncOp::getSymPrefix());
  assert(validityTaps);
}

void CircuitInterfaceZirgen::add_taps() {
  auto tapsGlob = mod().lookupSymbol<ZStruct::GlobalConstOp>(getTapsConstName());
  assert(tapsGlob && "Taps global not found");

  TapsAnalysis tapsAnalysis(
      mod().getContext(),
      llvm::to_vector(llvm::cast<mlir::ArrayAttr>(tapsGlob.getConstant()).getAsRange<TapAttr>()));
  tapSet = tapsAnalysis.getTapSet();
};

namespace {

ZStruct::ArrayOp createArray(OpBuilder& builder, ArrayRef<Val> edslVals) {
  assert(!edslVals.empty());
  Type elemType = edslVals[0].getValue().getType();
  auto arrayType = builder.getType<ZStruct::ArrayType>(elemType, edslVals.size());
  auto mlirValues =
      llvm::map_to_vector(edslVals, [](auto val) -> mlir::Value { return val.getValue(); });
  return builder.create<ZStruct::ArrayOp>(builder.getUnknownLoc(), arrayType, mlirValues);
}

using NamedTap = std::tuple</*bufferName=*/StringRef, /*offset=*/size_t, /*back=*/size_t>;

struct TapifyLoadOp : public OpRewritePattern<ZStruct::LoadOp> {
  TapifyLoadOp(MLIRContext* ctx, Interpreter& interp, DenseMap<NamedTap, Value>& tapIndex)
      : OpRewritePattern(ctx), interp(interp), tapIndex(tapIndex) {}

  LogicalResult matchAndRewrite(ZStruct::LoadOp op, PatternRewriter& rewriter) const final {
    ZStruct::BoundLayoutAttr ref =
        interp.evaluateConstantOfType<ZStruct::BoundLayoutAttr>(op.getRef());
    if (!ref) {
      op->emitError("couldn't evaluate reference");
      return failure();
    }

    auto distAttr = interp.evaluateConstantOfType<IntegerAttr>(op.getDistance());
    if (!distAttr) {
      op->emitError("couldn't evaluate distance");
      return failure();
    }
    size_t distance = ZStruct::getIndexVal(distAttr);

    auto namedTap = NamedTap{
        ref.getBuffer(), llvm::cast<ZStruct::RefAttr>(ref.getLayout()).getIndex(), distance};
    if (tapIndex.contains(namedTap)) {
      rewriter.replaceOp(op, tapIndex.lookup(namedTap));
      return success();
    } else {
      return failure();
    }
  }

private:
  Interpreter& interp;
  DenseMap<NamedTap, Value>& tapIndex;
};

} // namespace

Val CircuitInterfaceZirgen::compute_poly(llvm::ArrayRef<Val> u,
                                         llvm::ArrayRef<Val> out,
                                         llvm::ArrayRef<Val> accumMix,
                                         Val polyMix) const {
  MLIRContext* ctx = mod().getContext();

  // This is called in an EDSL context, so we need to bind the buffers correctly.
  auto* edslModule = zirgen::Module::getCurModule();
  auto& builder = edslModule->getBuilder();

  ValidityTapsFuncOp tapsWork;
  {
    IRRewriter::InsertionGuard guard(builder);
    builder.setInsertionPoint(validityTaps);
    // Keep our work in the loaded module so we can look up symbols.
    tapsWork = llvm::cast<Zhlt::ValidityTapsFuncOp>(builder.clone(*validityTaps));
  }

  DenseMap<NamedTap, Value> tapIndex;
  // Distances on all BufferKind::Global buffer accesses should be zero.
  for (auto [idx, val] : llvm::enumerate(out))
    tapIndex[std::make_tuple("global", idx, /*distance=*/0)] = val.getValue();
  for (auto [idx, val] : llvm::enumerate(accumMix))
    tapIndex[std::make_tuple("mix", idx, /*distance=*/0)] = val.getValue();

  Interpreter interp(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<TapifyLoadOp>(ctx, interp, tapIndex);

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  LogicalResult rewriteResult = applyPatternsAndFoldGreedily(tapsWork, frozenPatterns);
  assert(rewriteResult.succeeded() &&
         "Unable to apply patterns to generate wrapped zirgen verify function");

  // We should have folded out all ZStruct and ZHLT stuff except for
  // accesses into the taps array, so we can clone the resultant taps
  // evaluator into the target EDSL function.
  IRMapping mapping;
  ZStruct::ArrayOp tapsArray = createArray(builder, u);
  mapping.map(tapsWork.getTaps(), tapsArray);

  // We want to separate out constraint ops (and_eqz, and_cond) into
  // multiplies.  So we need to separate the "Constraint" type into a
  // "mul" and a "tot" component.
  IRMapping mulMapping;
  IRMapping totMapping;

  Value zero = Val({0, 0, 0, 0}).getValue();
  Value one = Val({1, 0, 0, 0}).getValue();

  Value result;
  for (Operation& op : tapsWork.getBody().front()) {
    TypeSwitch<Operation*>(&op)
        .Case<Zll::TrueOp>([&](auto op) {
          totMapping.map(op, zero);
          mulMapping.map(op, one);
        })
        .Case<Zll::AndEqzOp>([&](auto op) {
          auto mulOp = builder.create<Zll::MulOp>(
              op.getLoc(), mulMapping.lookup(op.getIn()), mapping.lookup(op.getVal()));

          totMapping.map(
              op.getOut(),
              builder.create<Zll::MulOp>(op.getLoc(), mulOp, totMapping.lookup(op.getIn())));

          mulMapping.map(op.getOut(),
                         builder.create<Zll::MulOp>(
                             op.getLoc(), mulMapping.lookup(op.getIn()), polyMix.getValue()));
        })
        .Case<Zll::AndCondOp>([&](auto op) {
          auto mulOp1 = builder.create<Zll::MulOp>(
              op.getLoc(), mulMapping.lookup(op.getIn()), totMapping.lookup(op.getInner()));
          auto mulOp2 =
              builder.create<Zll::MulOp>(op.getLoc(), mulOp1, mapping.lookup(op.getCond()));

          totMapping.map(
              op.getOut(),
              builder.create<Zll::MulOp>(op.getLoc(), mulOp2, totMapping.lookup(op.getIn())));

          mulMapping.map(op.getOut(),
                         builder.create<Zll::MulOp>(op.getLoc(),
                                                    mulMapping.lookup(op.getIn()),
                                                    mulMapping.lookup(op.getInner())));
        })
        .Case<arith::ConstantOp, Zll::PolyOp, ZStruct::SubscriptOp, ZStruct::ArrayOp>(
            [&](auto op) { builder.clone(*op, mapping); })
        .Case<Zhlt::ReturnOp>(
            [&](auto returnOp) { result = totMapping.lookup(returnOp.getValue()); })
        .Default([&](auto) {
          llvm::errs() << "Unexpected operation encountered when wrapping zirgen validity taps: "
                       << op << "\n";
          assert(false);
        });
  }
  tapsWork.erase();
  assert(result && "Missing terminator");
  return Val(result);
}

std::unique_ptr<CircuitInterface> getInterfaceZirgen(mlir::ModuleOp mod,
                                                     ProtocolInfo protocolInfo) {
  return std::make_unique<CircuitInterfaceZirgen>(mod, protocolInfo);
}

std::unique_ptr<CircuitInterface>
getInterfaceZirgen(mlir::MLIRContext* ctx, mlir::StringRef filename, ProtocolInfo protocolInfo) {
  ParserConfig config(ctx);
  OwningOpRef<ModuleOp> mod = parseSourceFile<ModuleOp>(filename, config);
  if (!mod)
    return nullptr;
  return std::make_unique<CircuitInterfaceZirgen>(std::move(mod), protocolInfo);
}

} // namespace zirgen::verify
