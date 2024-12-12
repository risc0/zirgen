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

#include "llvm/ADT/TypeSwitch.h"
#include <map>
#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

#include "encode.h"

namespace zirgen::BigInt::Bytecode {

namespace {

class Builder {
  std::unique_ptr<Program> output;
  std::map<mlir::Operation*, size_t> oplocs;

public:
  Builder() : output(std::make_unique<Program>()) {}
  std::unique_ptr<Program> finish() { return std::move(output); }
  void enroll(mlir::Operation&);
  size_t lookup(mlir::Operation&);
  size_t lookup(mlir::Type);
  void emit(Op&);
  void emitBin(Op::Code, size_t type, mlir::Value lhs, mlir::Value rhs);
  size_t def(const Input&);
  size_t def(const mlir::APInt&);
};

void Builder::enroll(mlir::Operation& op) {
  oplocs[&op] = output->ops.size();
}

size_t Builder::lookup(mlir::Operation& op) {
  auto iter = oplocs.find(&op);
  if (iter == oplocs.end()) {
    llvm::errs() << op << "\n";
    throw std::runtime_error("Reference before definition failure");
  }
  return iter->second;
}

size_t Builder::lookup(mlir::Type opType) {
  BigIntType bit = mlir::cast<BigIntType>(opType);
  Type t;
  t.coeffs = bit.getCoeffs();
  t.maxPos = bit.getMaxPos();
  t.maxNeg = bit.getMaxNeg();
  t.minBits = bit.getMinBits();
  // this is inefficient, but for some reason, std::map isn't working:
  // just linear scan the type table
  for (size_t i = 0; i < output->types.size(); ++i) {
    const Type& e = output->types[i];
    if (t.coeffs != e.coeffs) {
      continue;
    }
    if (t.maxPos != e.maxPos) {
      continue;
    }
    if (t.maxNeg != e.maxNeg) {
      continue;
    }
    if (t.minBits != e.minBits) {
      continue;
    }
    return i;
  }
  size_t typeIndex = output->types.size();
  output->types.push_back(t);
  return typeIndex;
}

void Builder::emit(Op& op) {
  output->ops.push_back(op);
}

void Builder::emitBin(Op::Code code, size_t type, mlir::Value lhs, mlir::Value rhs) {
  Op op;
  op.code = code;
  op.type = type;
  op.operandA = lookup(*lhs.getDefiningOp());
  op.operandB = lookup(*rhs.getDefiningOp());
  emit(op);
}

size_t Builder::def(const Input& input) {
  size_t out = output->inputs.size();
  output->inputs.push_back(input);
  return out;
}

size_t Builder::def(const mlir::APInt& value) {
  size_t out = output->constants.size();
  size_t words = value.getNumWords();
  for (size_t i = 0; i < words; ++i) {
    output->constants.push_back(value.getRawData()[i]);
  }
  return out;
}

} // namespace

std::unique_ptr<Program> encode(mlir::func::FuncOp func) {
  Builder builder;
  for (mlir::Operation& origOp : func.getBody().front().without_terminator()) {
    builder.enroll(origOp);
    llvm::TypeSwitch<mlir::Operation*>(&origOp)
        .Case<DefOp>([&](auto op) {
          Op newOp{};
          newOp.code = Op::Def;
          newOp.type = builder.lookup(origOp.getResultTypes()[0]);
          Input input;
          input.label = op.getLabel();
          input.bitWidth = op.getBitWidth();
          if (op.getBitWidth() >= 0xFFFFFFFFU) {
            throw std::runtime_error("unexpectedly large bitWidth");
          }
          input.minBits = op.getMinBits();
          if (op.getMinBits() >= 0xFFFFU) {
            throw std::runtime_error("unexpectedly large minBits");
          }
          input.isPublic = op.getIsPublic();
          newOp.operandA = builder.def(input);
          builder.emit(newOp);
        })
        .Case<ConstOp>([&](auto op) {
          Op newOp{};
          newOp.code = Op::Con;
          newOp.type = builder.lookup(origOp.getResultTypes()[0]);
          mlir::APInt value = op.getValue();
          newOp.operandA = builder.def(value);
          newOp.operandB = value.getNumWords();
          builder.emit(newOp);
        })
        .Case<LoadOp>([&](auto op) {
          Op newOp{};
          newOp.code = Op::Load;
          newOp.type = builder.lookup(origOp.getResultTypes()[0]);
          newOp.operandA = op.getArena() << 16 | op.getOffset();
          builder.emit(newOp);
        })
        .Case<StoreOp>([&](auto op) {
          Op newOp{};
          newOp.code = Op::Store;
          newOp.type = builder.lookup(op.getIn().getType());
          newOp.operandA = op.getArena() << 16 | op.getOffset();
          newOp.operandB = builder.lookup(*op.getIn().getDefiningOp());
          builder.emit(newOp);
        })
        .Case<AddOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Add, type, op.getLhs(), op.getRhs());
        })
        .Case<SubOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Sub, type, op.getLhs(), op.getRhs());
        })
        .Case<MulOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Mul, type, op.getLhs(), op.getRhs());
        })
        .Case<NondetRemOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Rem, type, op.getLhs(), op.getRhs());
        })
        .Case<NondetQuotOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Quo, type, op.getLhs(), op.getRhs());
        })
        .Case<NondetInvOp>([&](auto op) {
          size_t type = builder.lookup(origOp.getResultTypes()[0]);
          builder.emitBin(Op::Inv, type, op.getLhs(), op.getRhs());
        })
        .Case<EqualZeroOp>([&](auto op) {
          Op newOp{};
          newOp.code = Op::Eqz;
          newOp.operandA = builder.lookup(*op.getIn().getDefiningOp());
          builder.emit(newOp);
        })
        .Case<InvOp>([&](auto op) {
          llvm::errs() << *op << "\n";
          throw std::runtime_error("Cannot write ModularInvOp; use lower-modular-inv pass");
        })
        .Case<ReduceOp>([&](auto op) {
          llvm::errs() << *op << "\n";
          throw std::runtime_error("Cannot write ReduceOp; use lower-reduce pass");
        })
        .Default([&](mlir::Operation* op) {
          llvm::errs() << *op << "\n";
          throw std::runtime_error("Cannot write unknown op");
        });
  }

  return builder.finish();
}

} // namespace zirgen::BigInt::Bytecode
