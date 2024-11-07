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

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "llvm/ADT/APSInt.h"
#include <vector>

#include "decode.h"

namespace zirgen::BigInt::Bytecode {

namespace {

class Decoder {
public:
  Decoder(const Program& prog, mlir::MLIRContext* ctx);
  void operandsAB(size_t i, mlir::Value& lhs, mlir::Value& rhs);
  void operandA(size_t i, mlir::Value& val);
  void operandB(size_t i, mlir::Value& val);
  void emit(size_t i, mlir::Value);
  BigIntType type(const Op& op);

private:
  const Program& prog;
  std::vector<BigIntType> types;
  std::vector<mlir::Value> polys;
};

Decoder::Decoder(const Program& prog, mlir::MLIRContext* ctx)
    : prog(prog), types(prog.types.size()), polys(prog.ops.size()) {
  for (size_t i = 0; i < prog.types.size(); ++i) {
    const Type& t = prog.types[i];
    types[i] = BigIntType::get(ctx, t.coeffs, t.maxPos, t.maxNeg, t.minBits);
  }
}

void Decoder::operandsAB(size_t i, mlir::Value& lhs, mlir::Value& rhs) {
  const Op& op = prog.ops[i];
  if (op.operandA >= i || op.operandB >= i) {
    throw std::runtime_error("reference to undefined value");
  }
  lhs = polys[op.operandA];
  rhs = polys[op.operandB];
}

void Decoder::operandA(size_t i, mlir::Value& val) {
  const Op& op = prog.ops[i];
  if (op.operandA >= i) {
    throw std::runtime_error("reference to undefined value");
  }
  val = polys[op.operandA];
}

void Decoder::operandB(size_t i, mlir::Value& val) {
  const Op& op = prog.ops[i];
  if (op.operandB >= i) {
    throw std::runtime_error("reference to undefined value");
  }
  val = polys[op.operandB];
}

void Decoder::emit(size_t i, mlir::Value poly) {
  polys[i] = poly;
}

BigIntType Decoder::type(const Op& op) {
  return types[op.type];
}

} // namespace

mlir::func::FuncOp decode(mlir::ModuleOp module, const Program& prog) {
  auto ctx = module.getContext();
  auto loc = mlir::UnknownLoc::get(ctx);
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto funcType = mlir::FunctionType::get(ctx, {}, {});
  auto out = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
  builder.setInsertionPointToStart(out.addEntryBlock());
  Decoder state(prog, ctx);

  for (size_t i = 0; i < prog.ops.size(); ++i) {
    const Op& op = prog.ops[i];
    switch (op.code) {
    case Op::Eqz: {
      mlir::Value val;
      state.operandA(i, val);
      builder.create<EqualZeroOp>(loc, val);
    } break;
    case Op::Def: {
      const Input& wire = prog.inputs[op.operandA];
      uint64_t label = wire.label;
      uint32_t bits = wire.bitWidth;
      uint32_t min = wire.minBits;
      bool pub = wire.isPublic;
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<DefOp>(loc, t, bits, label, pub, min));
    } break;
    case Op::Con: {
      size_t base = op.operandA;
      size_t count = op.operandB;
      std::vector<uint64_t> words;
      for (size_t i = 0; i < count; ++i) {
        words.push_back(prog.constants[base + i]);
      }
      mlir::APInt value(count * 64, mlir::ArrayRef<uint64_t>(words));
      mlir::Type t = state.type(op);
      auto attr = mlir::IntegerAttr::get(ctx, llvm::APSInt(value));
      state.emit(i, builder.create<ConstOp>(loc, t, attr));
    } break;
    case Op::Load: {
      uint32_t arena = op.operandA >> 16;
      uint32_t offset = op.operandA & 0xffff;
      mlir::Type t = state.type(op);
      uint32_t bitWidth = llvm::dyn_cast<BigIntType>(t).getCoeffs() * 8;
      state.emit(i, builder.create<LoadOp>(loc, bitWidth, arena, offset));
    } break;
    case Op::Store: {
      uint32_t arena = op.operandA >> 16;
      uint32_t offset = op.operandA & 0xffff;
      mlir::Value in;
      state.operandB(i, in);
      builder.create<StoreOp>(loc, in, arena, offset);
    } break;
    case Op::Add: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<AddOp>(loc, t, lhs, rhs));
    } break;
    case Op::Sub: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<SubOp>(loc, t, lhs, rhs));
    } break;
    case Op::Mul: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<MulOp>(loc, t, lhs, rhs));
    } break;
    case Op::Rem: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<NondetRemOp>(loc, t, lhs, rhs));
    } break;
    case Op::Quo: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<NondetQuotOp>(loc, t, lhs, rhs));
    } break;
    case Op::Inv: {
      mlir::Value lhs, rhs;
      state.operandsAB(i, lhs, rhs);
      mlir::Type t = state.type(op);
      state.emit(i, builder.create<NondetInvOp>(loc, t, lhs, rhs));
    } break;
    }
  }
  // Add terminator op, for the sake of propriety.
  builder.create<mlir::func::ReturnOp>(loc);
  return out;
}

} // namespace zirgen::BigInt::Bytecode
