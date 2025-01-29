// Copyright 2025 RISC Zero, Inc.
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
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"

using namespace mlir;

namespace zirgen::ByteCode {

LogicalResult ExitOp::verify() {
  auto execOp = getOperation()->getParentOfType<ExecutorOp>();
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();

  if (execOp && funcOp)
    return emitError() << "Ambiguous parent; is it an ExecutorOp or FuncOp?";

  if (!execOp && !funcOp)
    return emitError() << "Unable to find FuncOp or ExecutorOp";

  if (execOp) {
    if (!llvm::equal(execOp.getFunctionType().getResults(), getVals().getTypes()))
      return emitError() << "ExecutorOp return types " << execOp.getFunctionType().getResults()
                         << " do not match ExitOp return types " << getVals().getTypes();
  } else {
    assert(funcOp);
    if (!llvm::equal(funcOp.getFunctionType().getResults(), getVals().getTypes()))
      return emitError() << "FuncOp function type " << funcOp.getFunctionType()
                         << " does not match ExitOp return types " << getVals().getTypes();
  }
  return success();
}

EncodedElement EncodedOp::getElement(size_t idx) {
  ssize_t encoded = getEncoded()[idx];
  if (encoded >= 0) {
    return size_t(encoded);
  } else {
    // Placeholder for a value
    size_t idx = (-encoded) - 1;
    if (idx >= getValues().size())
      return OpResult(getResults()[idx - getValues().size()]);
    else
      return Value(getValues()[idx]);
  }
}

mlir::ParseResult EncodedOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  SmallVector<std::variant<size_t, Type, Value>> encoded;

  if (parser
          .parseCommaSeparatedList([&]() {
            size_t intVal;
            auto intResult = parser.parseOptionalInteger(intVal);
            if (intResult.has_value()) {
              if (failed(*intResult))
                return failure();
              encoded.push_back(intVal);
              return success();
            }

            if (succeeded(parser.parseOptionalArrow())) {
              Type ty;
              if (failed(parser.parseType(ty)))
                return failure();
              encoded.push_back(ty);
              result.addTypes(ty);
              return success();
            }

            OpAsmParser::UnresolvedOperand newOperand;
            Type ty;
            if (parser.parseOperand(newOperand) || parser.parseColon() || parser.parseType(ty) ||
                parser.resolveOperand(newOperand, ty, result.operands))
              return failure();
            return success();
          })
          .failed())
    return failure();

  SmallVector<ssize_t> encodedInts;
  size_t operandsSeen = 0;
  size_t resultsSeen = 0;
  for (auto [idx, val] : llvm::enumerate(encoded)) {
    if (auto* intVal = std::get_if<size_t>(&val)) {
      encodedInts.push_back(*intVal);
    } else if (auto* operand = std::get_if<Value>(&val)) {
      encodedInts.push_back((-1) - operandsSeen);
      operandsSeen++;
    } else if (auto* ty = std::get_if<Type>(&val)) {
      encodedInts.push_back((-1) - result.operands.size() - resultsSeen);
      resultsSeen++;
    } else {
      llvm_unreachable("unhandled variant");
    }
  }

  NamedAttrList attrDict;
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  OpBuilder builder(result.getContext());
  result.addAttribute("encoded", builder.getDenseI64ArrayAttr(encodedInts));

  return success();
}

void EncodedOp::print(mlir::OpAsmPrinter& p) {
  p << " ";
  llvm::interleaveComma(llvm::seq(size()), p, [&](size_t idx) {
    auto elem = getElement(idx);
    if (auto* intVal = std::get_if<size_t>(&elem)) {
      p << *intVal;
    } else if (auto* operand = std::get_if<Value>(&elem)) {
      p << *operand << " : " << operand->getType();
    } else if (auto* result = std::get_if<OpResult>(&elem)) {
      p << "->";
      p << result->getType();
    } else {
      llvm_unreachable("unhandled variant");
    }
  });
  p.printOptionalAttrDict((*this)->getAttrs(), {"encoded"});
}

void EncodedOp::build(OpBuilder& builder,
                      OperationState& state,
                      ArrayRef<BuildEncodedElement> elems) {
  size_t numOperands =
      llvm::count_if(elems, [&](auto& elem) { return std::holds_alternative<Value>(elem); });
  size_t operandIndex = 0;
  size_t resultIndex = 0;
  SmallVector<ssize_t> encoded;
  for (auto& elem : elems) {
    if (auto* intArg = std::get_if<size_t>(&elem)) {
      encoded.push_back(*intArg);
    } else if (auto* operand = std::get_if<Value>(&elem)) {
      encoded.push_back((-1) - operandIndex);
      state.addOperands(*operand);
      ++operandIndex;
    } else if (auto* ty = std::get_if<Type>(&elem)) {
      encoded.push_back((-1) - numOperands - resultIndex);
      state.addTypes(*ty);
      ++resultIndex;
    }
  }
  assert(operandIndex == numOperands);

  auto& prop = state.getOrAddProperties<Properties>();
  prop.encoded = builder.getDenseI64ArrayAttr(encoded);
}

} // namespace zirgen::ByteCode
