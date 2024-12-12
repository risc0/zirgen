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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/Conversion/ZStructToZll/PassDetail.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

using namespace mlir;
using namespace zirgen::Zll;
using namespace zirgen::ZStruct;

namespace zirgen::ZStructToZll {

namespace {

// test:
//  bazelisk test //zirgen/Dialect/Zll/IR/test:lowercomps.mlir.test
// run:
//  bazelisk run //zirgen/compiler/tools:zirgen-opt -- --lower-composites
//  `pwd`/zirgen/Dialect/Zll/IR/test/lowercomps.mlir
// additional useful options:
//  --mlir-disable-threading
//  -debug-only=dialect-conversion

BufferType calcBufferType(Type type);

BufferType calcBufferType(ValType type) {
  // Wrap this value type as a single element of a buffer type.
  auto ctx = type.getContext();
  return BufferType::get(ctx, type, 1, BufferKind::Mutable);
}

BufferType calcBufferType(ArrayType type) {
  // Compute the buffer type for each element of this array, then create
  // a buffer large enough to hold one slice of that size for each element.
  auto ctx = type.getContext();
  auto elementBuf = calcBufferType(type.getElement());
  size_t size = type.getSize() * elementBuf.getSize();
  return BufferType::get(ctx, elementBuf.getElement(), size, BufferKind::Mutable);
}

BufferType calcBufferType(StructType type) {
  // Convert each field's type to a buffer, then concatenate. All fields must
  // share the same fundamental element type.
  auto ctx = type.getContext();
  auto element = ValType::get(ctx, kFieldPrimeDefault, 1);
  bool definedElement = false;
  size_t sum = 0;
  for (auto& field : type.getFields()) {
    auto bufType = calcBufferType(field.type);
    sum += bufType.getSize();
    auto fieldElement = bufType.getElement();
    if (definedElement) {
      assert(fieldElement == element);
    } else {
      definedElement = true;
      element = fieldElement;
    }
  }
  return BufferType::get(ctx, element, sum, BufferKind::Mutable);
}

BufferType calcBufferType(UnionType type) {
  auto ctx = type.getContext();
  auto element = ValType::get(ctx, kFieldPrimeDefault, 1);
  bool definedElement = false;
  unsigned max = 0;
  for (auto& field : type.getFields()) {
    auto bufType = calcBufferType(field.type);
    max = std::max(max, bufType.getSize());
    auto fieldElement = bufType.getElement();
    if (definedElement) {
      assert(fieldElement == element);
    } else {
      definedElement = true;
      element = fieldElement;
    }
  }
  return BufferType::get(ctx, element, max, BufferKind::Mutable);
}

BufferType calcBufferType(RefType type) {
  return calcBufferType(type.getElement());
}

BufferType calcBufferType(Type type) {
  return TypeSwitch<mlir::Type, BufferType>(type)
      .Case<ValType>([](ValType t) { return calcBufferType(t); })
      .Case<ArrayType>([](ArrayType t) { return calcBufferType(t); })
      .Case<StructType>([](StructType t) { return calcBufferType(t); })
      .Case<UnionType>([](UnionType t) { return calcBufferType(t); })
      .Case<RefType>([](RefType t) { return calcBufferType(t); });
}

size_t calcBufferSize(Type type) {
  return calcBufferType(type).getSize();
}

class BufferizeTypeConverter : public TypeConverter {
public:
  BufferizeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](ArrayType type) -> Type { return calcBufferType(type); });
    addConversion([](StructType type) -> Type { return calcBufferType(type); });
    addConversion([](UnionType type) -> Type { return calcBufferType(type); });
    addConversion([](RefType type) -> Type { return calcBufferType(type); });
  }
};

struct BufferizeTarget : public ConversionTarget {
  BufferizeTarget(MLIRContext& ctx, TypeConverter& tc) : ConversionTarget(ctx) {
    // Zirgen ops and builtin ops are legal by default.
    addLegalDialect<ZllDialect>();
    // Zirgen ops which specifically work with composite types are not legal.
    addIllegalOp<LoadOp, StoreOp>();
    addIllegalOp<LookupOp, SubscriptOp>();
    // Functions are legal if they have legal (non-composite) types.
    addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) -> bool {
      for (Type t : func.getArgumentTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Type t : func.getResultTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Block& block : func.getBody()) {
        for (Type t : block.getArgumentTypes()) {
          if (!tc.isLegal(t))
            return false;
        }
      }
      return true;
    });
    // Other ops are legal if their types are legal, otherwise they must be converted.
    markUnknownOpDynamicallyLegal([&](Operation* op) -> bool {
      for (Type t : op->getResultTypes()) {
        if (!tc.isLegal(t)) {
          return false;
        }
      }
      for (Type t : op->getOperandTypes()) {
        if (!tc.isLegal(t)) {
          return false;
        }
      }
      return true;
    });
  }
};

struct ConvertLoad : public ConversionPattern {
  ConvertLoad(MLIRContext& context, TypeConverter& converter)
      : ConversionPattern(converter, LoadOp::getOperationName(), 1, &context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    // ins Ref:$ref
    // outs Val:$out
    // Load a value from a ref, which has been converted to a buffer slice.
    auto load = dyn_cast<LoadOp>(op);
    assert(isa<ValType>(load.getResult().getType()));
    auto loc = op->getLoc();
    auto buf = operands[0];
    size_t offset = 0;
    size_t back = 0;
    if (load.getDistance()) {
      back = load.getDistance().getDefiningOp<ConstOp>().getCoefficients()[0];
    }

    auto ret = rewriter.create<GetOp>(loc, buf, offset, back, IntegerAttr());
    rewriter.replaceOp(op, ret->getResults());
    return success();
  }
};

struct ConvertStore : public OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const final {
    // ins Ref:$ref, Val:$val
    // Store a value into a ref, which has been converted to a buffer slice.
    auto loc = op->getLoc();
    auto buf = adaptor.getRef();
    auto val = adaptor.getVal();
    size_t offset = 0;
    auto ret = rewriter.create<SetOp>(loc, buf, offset, val);
    rewriter.replaceOp(op, ret->getResults());
    return success();
  }
};

struct ConvertLookup : public ConversionPattern {
  ConvertLookup(MLIRContext& context, TypeConverter& converter)
      : ConversionPattern(converter, LookupOp::getOperationName(), 1, &context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    // Operands: ContainerType:$base, StrAttr:$member
    // If base is a struct, compute the offset to the named member by iterating over the preceding
    // members and summing their buffer-element-sizes; if base is a union, the offset is zero.
    // Return a slice of the buffer corresponding to the requested member.
    auto lookup = dyn_cast<LookupOp>(op);
    auto memberName = lookup.getMember();
    size_t index = 0;
    if (isa<StructType>(lookup.getBase().getType())) {
      auto t = cast<StructType>(lookup.getBase().getType());
      for (auto& field : t.getFields()) {
        if (field.name == memberName) {
          break;
        }
        index += calcBufferSize(field.type);
      }
    }
    auto loc = op->getLoc();
    assert(1 == operands.size());
    auto base = operands[0];
    size_t size = calcBufferSize(lookup.getOut().getType());
    rewriter.replaceOp(op, rewriter.create<SliceOp>(loc, base, index, size));
    return success();
  }
};

struct ConvertSubscript : public OpConversionPattern<SubscriptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SubscriptOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const final {
    // Compute the base offset and size of the requested array element.
    // Return a slice of the buffer corresponding to that range.
    auto loc = op->getLoc();
    auto base = adaptor.getBase();
    size_t size = calcBufferSize(op.getOut().getType());
    IntegerAttr indexAttr = op.getIndexAsAttr();
    if (!indexAttr) {
      return rewriter.notifyMatchFailure(op, "failed to fold subscript index");
    }
    size_t index = indexAttr.getInt() * size;
    rewriter.replaceOp(op, rewriter.create<SliceOp>(loc, base, index, size));
    return success();
  }
};

struct ConvertReturn : public ConversionPattern {
  ConvertReturn(MLIRContext& context, TypeConverter& converter)
      : ConversionPattern(converter, func::ReturnOp::getOperationName(), 1, &context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    // ins: Variadic<AnyType>:$operands
    auto loc = op->getLoc();
    auto ret = rewriter.create<func::ReturnOp>(loc, operands);
    rewriter.replaceOp(op, ret->getResults());
    return success();
  }
};

struct ConvertCall : public ConversionPattern {
  ConvertCall(MLIRContext& context, TypeConverter& converter)
      : ConversionPattern(converter, func::CallOp::getOperationName(), 1, &context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    // ins: FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands
    // outs: Variadic<AnyType>
    const TypeConverter* converter = getTypeConverter();

    auto loc = op->getLoc();
    auto attr = op->getAttrs();
    auto succ = op->getSuccessors();

    // Convert result types.
    auto inResults = op->getResultTypes();
    llvm::SmallVector<Type, 4> outResults;
    if (failed(converter->convertTypes(inResults, outResults))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }

    // Argument types have already been converted.
    // Generate a new call using the new operands.
    auto nameRef = op->getName().getStringRef();
    OperationState state(loc, nameRef, operands, outResults, attr, succ);
    Operation* ret = rewriter.create(state);
    rewriter.replaceOp(op, ret->getResults());
    return success();
  }
};

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp func,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const TypeConverter* converter = getTypeConverter();

    // Convert argument types.
    size_t numArgs = func.getNumArguments();
    TypeConverter::SignatureConversion signature(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      auto t = func.getArgument(i).getType();
      if (failed(converter->convertSignatureArg(i, t, signature))) {
        return rewriter.notifyMatchFailure(func, "failed to convert argument type");
      }
    }
    auto outArgs = signature.getConvertedTypes();

    // Convert result types.
    auto inResults = func.getResultTypes();
    SmallVector<Type> outResults;
    if (failed(converter->convertTypes(inResults, outResults))) {
      return rewriter.notifyMatchFailure(func, "failed to convert result types");
    }

    // Create complete converted function type from arg & result types.
    // Update the function op to use the new type signature.
    MLIRContext* ctx = func.getContext();
    auto outFuncType = mlir::FunctionType::get(ctx, outArgs, outResults);
    rewriter.startOpModification(func);
    func.setType(outFuncType);
    auto body = &func.getBody();
    if (failed(rewriter.convertRegionTypes(body, *converter, &signature))) {
      return failure();
    }
    rewriter.finalizeOpModification(func);
    return success();
  }
};

struct LowerCompositesPass : public LowerCompositesBase<LowerCompositesPass> {
  void runOnOperation() override {
    // reduce struct, union, and array to buffer
    // convert load to get, store to set
    // convert lookup and subscript to slice
    auto& ctx = getContext();
    BufferizeTypeConverter converter;
    BufferizeTarget target(ctx, converter);
    RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertLoad>(ctx, converter);
    patterns.insert<ConvertStore>(converter, &ctx);
    patterns.insert<ConvertLookup>(ctx, converter);
    patterns.insert<ConvertSubscript>(converter, &ctx);
    patterns.insert<ConvertReturn>(ctx, converter);
    patterns.insert<ConvertCall>(ctx, converter);
    patterns.insert<ConvertFunc>(converter, &ctx);
    if (applyPartialConversion(getOperation(), target, std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // end namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLowerCompositesPass() {
  return std::make_unique<LowerCompositesPass>();
}

} // namespace zirgen::ZStructToZll
