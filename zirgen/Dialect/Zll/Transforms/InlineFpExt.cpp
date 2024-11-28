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
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct Rewriter {
  OpBuilder& builder;
  bool invalid;
  size_t prime;
  size_t kExt;
  Type fpType;
  Type fpExtType;
  DenseMap<Value, SmallVector<Value, 4>> expansions;
  IRMapping mapper;
  Value zero;
  Value beta;
  Value nbeta;

  // Currently this is fixed to the Baby Bear extension field
  Rewriter(OpBuilder& builder)
      : builder(builder)
      , invalid(false)
      , prime(kFieldPrimeDefault)
      , kExt(kBabyBearExtSize)
      , fpType(ValType::getBaseType(builder.getContext()))
      , fpExtType(ValType::getExtensionType(builder.getContext())) {}

  SmallVector<Value, 4> getExpanded(Value orig, bool doZero) {
    // If orig is an ext element, return it's parts, otherwise extend
    auto it = expansions.find(orig);
    if (it != expansions.end()) {
      assert(cast<ValType>(orig.getType()).getFieldK() == kExt);
      return it->second;
    }
    assert(cast<ValType>(orig.getType()).getFieldK() == 1);
    assert(mapper.contains(orig));
    SmallVector<Value, 4> out(kExt, mapper.lookupOrNull(orig));
    if (doZero) {
      for (size_t i = 1; i < kExt; i++) {
        out[i] = zero;
      }
    }
    return out;
  }

  template <typename Op> void doUnaryEltwise(Op op) {
    auto inType = cast<ValType>(op.getIn().getType());
    if (inType.getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    auto in = expansions[op.getIn()];
    SmallVector<Value, 4> out;
    for (size_t i = 0; i < kExt; i++) {
      out.push_back(builder.create<Op>(op.getLoc(), in[i]));
    }
    expansions[op.getOut()] = out;
  }

  template <typename Op> void doBinaryEltwise(Op op, bool doZero) {
    auto lhsType = cast<ValType>(op.getLhs().getType());
    auto rhsType = cast<ValType>(op.getRhs().getType());
    if (lhsType.getFieldK() == 1 && rhsType.getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    auto lhs = getExpanded(op.getLhs(), doZero);
    auto rhs = getExpanded(op.getRhs(), doZero);
    SmallVector<Value, 4> out;
    for (size_t i = 0; i < kExt; i++) {
      out.push_back(builder.create<Op>(op.getLoc(), lhs[i], rhs[i]));
    }
    expansions[op.getOut()] = out;
  }

  void doIopRead(Iop::ReadOp op) {
    Type type = op.getOuts()[0].getType();
    if (!isa<ValType>(type) || cast<ValType>(type).getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    size_t n = op.getOuts().size();
    std::vector<Value> vals;
    for (size_t i = 0; i < kExt * n; i++) {
      auto newReadOp = builder.create<Iop::ReadOp>(
          op.getLoc(), mlir::ArrayRef<Type>({fpType}), op.getIop(), false);
      vals.push_back(newReadOp.getOuts()[0]);
    }
    for (size_t i = 0; i < n; i++) {
      llvm::SmallVector<Value, 4> poly(kExt);
      for (size_t j = 0; j < kExt; j++) {
        if (op.getFlip()) {
          poly[j] = vals[i * kExt + j];
        } else {
          poly[j] = vals[j * n + i];
        }
      }
      expansions[op.getOuts()[i]] = poly;
    }
  }

  void doHash(HashOp op) {
    if (cast<ValType>(op.getIn()[0].getType()).getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    size_t n = op.getIn().size();
    std::vector<Value> vals(kExt * n);
    for (size_t i = 0; i < n; i++) {
      auto poly = expansions[op.getIn()[i]];
      for (size_t j = 0; j < kExt; j++) {
        if (op.getFlip()) {
          vals[i * kExt + j] = poly[j];
        } else {
          vals[j * n + i] = poly[j];
        }
      }
    }
    Type digestType = DigestType::get(builder.getContext(), DigestKind::Default);
    auto newOp = builder.create<HashOp>(op.getLoc(), digestType, false, vals);
    mapper.map(op.getOut(), newOp.getOut());
  }

  void doSelect(SelectOp op) {
    auto elemType = op.getElems()[0].getType();
    if (!isa<ValType>(elemType) || cast<ValType>(elemType).getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    size_t n = op.getElems().size();
    SmallVector<Value, 4> out;
    Value idx = mapper.lookupOrNull(op.getIdx());
    for (size_t i = 0; i < kExt; i++) {
      std::vector<Value> elems;
      for (size_t j = 0; j < n; j++) {
        elems.push_back(expansions[op.getElems()[j]][i]);
      }
      out.push_back(builder.create<SelectOp>(op.getLoc(), fpType, idx, elems));
    }
    expansions[op.getOut()] = out;
  }

  void doConst(ConstOp op) {
    if (cast<ValType>(op.getOut().getType()).getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    SmallVector<Value, 4> out;
    for (size_t i = 0; i < kExt; i++) {
      uint64_t val = op.getCoefficients()[i];
      out.push_back(builder.create<ConstOp>(op.getLoc(), fpType, val));
    }
    expansions[op.getOut()] = out;
  }

  void doInv(InvOp op) {
    size_t k = cast<ValType>(op.getIn().getType()).getFieldK();
    if (k == 1) {
      builder.clone(*op, mapper);
      return;
    }
    if (k != 4) {
      op->emitError("Invalid Inv, only degree 4 supported");
      invalid = true;
      return;
    }

    auto a = expansions[op.getIn()];
    auto add = [&](Value a, Value b) -> Value { return builder.create<AddOp>(op.getLoc(), a, b); };
    auto sub = [&](Value a, Value b) -> Value { return builder.create<SubOp>(op.getLoc(), a, b); };
    auto mul = [&](Value a, Value b) -> Value { return builder.create<MulOp>(op.getLoc(), a, b); };
    auto neg = [&](Value a) -> Value { return builder.create<NegOp>(op.getLoc(), a); };
    auto inv = [&](Value a) -> Value { return builder.create<InvOp>(op.getLoc(), a); };

    Value b0 = add(mul(a[0], a[0]), mul(beta, sub(mul(a[1], add(a[3], a[3])), mul(a[2], a[2]))));
    Value b2 = add(sub(mul(a[0], add(a[2], a[2])), mul(a[1], a[1])), mul(beta, mul(a[3], a[3])));
    Value c = add(mul(b0, b0), mul(beta, mul(b2, b2)));
    Value ic = inv(c);
    b0 = mul(b0, ic);
    b2 = mul(b2, ic);
    expansions[op.getOut()] = {
        add(mul(a[0], b0), mul(beta, mul(a[2], b2))),
        neg(add(mul(a[1], b0), mul(beta, mul(a[3], b2)))),
        sub(mul(a[2], b0), mul(a[0], b2)),
        sub(mul(a[1], b2), mul(a[3], b0)),
    };
  }

  void doMul(MulOp op) {
    size_t kLhs = cast<ValType>(op.getLhs().getType()).getFieldK();
    size_t kRhs = cast<ValType>(op.getRhs().getType()).getFieldK();
    if (kLhs == 1 || kRhs == 1) {
      doBinaryEltwise(op, false);
      return;
    }
    auto lhsExp = expansions[op.getLhs()];
    auto rhsExp = expansions[op.getRhs()];
    SmallVector<Value, 4> out(kExt, zero);
    for (size_t i = 0; i < kLhs; i++) {
      for (size_t j = 0; j < kRhs; j++) {
        size_t pos = i + j;
        Value mul = builder.create<MulOp>(op.getLoc(), lhsExp[i], rhsExp[j]);
        if (pos >= kLhs) {
          pos -= kLhs;
          mul = builder.create<MulOp>(op.getLoc(), nbeta, mul);
        }
        out[pos] = builder.create<AddOp>(op.getLoc(), out[pos], mul);
      }
    }
    expansions[op.getOut()] = out;
  }

  void doEqualZero(EqualZeroOp op) {
    auto inType = cast<ValType>(op.getIn().getType());
    if (inType.getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    auto in = expansions[op.getIn()];
    for (size_t i = 0; i < kExt; i++) {
      builder.create<EqualZeroOp>(op.getLoc(), in[i]);
    }
  }

  void doIopRngVal(Iop::RngValOp op) {
    if (cast<ValType>(op.getOut().getType()).getFieldK() == 1) {
      builder.clone(*op, mapper);
      return;
    }
    SmallVector<Value, 4> out;
    for (size_t i = 0; i < kExt; i++) {
      out.push_back(builder.create<Iop::RngValOp>(op.getLoc(), fpType, op.getIop()));
    }
    expansions[op.getOut()] = out;
  }

  void runOnBlock(Location loc, Block& block) {
    Type ty = ValType::getBaseType(builder.getContext());
    zero = builder.create<ConstOp>(loc, ty, 0);
    beta = builder.create<ConstOp>(loc, ty, 11);
    nbeta = builder.create<NegOp>(loc, beta);
    for (Operation& origOp : block.without_terminator()) {
      TypeSwitch<Operation*>(&origOp)
          .Case<NegOp>([&](auto op) { doUnaryEltwise(op); })
          .Case<AddOp, SubOp>([&](auto op) { doBinaryEltwise(op, true); })
          .Case<BitAndOp, ModOp>([&](auto op) { doBinaryEltwise(op, false); })
          .Case<Iop::ReadOp>([&](auto op) { doIopRead(op); })
          .Case<Iop::RngValOp>([&](auto op) { doIopRngVal(op); })
          .Case<HashOp>([&](auto op) { doHash(op); })
          .Case<SelectOp>([&](auto op) { doSelect(op); })
          .Case<ConstOp>([&](auto op) { doConst(op); })
          .Case<EqualZeroOp>([&](auto op) { doEqualZero(op); })
          .Case<InvOp>([&](auto op) { doInv(op); })
          .Case<MulOp>([&](auto op) { doMul(op); })
          .Case<NormalizeOp>([&](auto op) { doUnaryEltwise(op); })
          .Case<ExternOp>([&](auto op) {
            if (op.getName() != "log") {
              op->emitWarning() << "Warning: discarding non-log extern";
            }
          })
          .Case<HashFoldOp,
                HashAssertEqOp,
                Iop::CommitOp,
                Iop::RngBitsOp,
                SetGlobalOp,
                SetGlobalDigestOp>([&](auto op) { builder.clone(*op, mapper); })
          .Default([&](Operation* op) {
            op->emitError("Invalid operation in InlineFpExtPass");
            invalid = true;
          });
    }
  }
};

struct InlineFpExtPass : public InlineFpExtBase<InlineFpExtPass> {
  void runOnOperation() override {
    // Get the function to run on + make a builder
    auto func = getOperation();
    auto loc = func.getLoc();
    Block* funcBlock = &func.front();
    Block* newBlock = new Block;
    func.getBody().push_back(newBlock);
    auto builder = OpBuilder::atBlockBegin(newBlock);

    // Make the rewriter + run on the entry block
    Rewriter rewriter(builder);
    rewriter.runOnBlock(loc, *funcBlock);

    // Fail if it's bad
    if (rewriter.invalid) {
      signalPassFailure();
      return;
    }

    // Add terminator
    builder.create<func::ReturnOp>(loc);

    // Replace block with new block
    funcBlock->clear();
    funcBlock->getOperations().splice(
        funcBlock->begin(), newBlock->getOperations(), newBlock->begin(), newBlock->end());
    newBlock->erase();
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInlineFpExtPass() {
  return std::make_unique<InlineFpExtPass>();
}

} // namespace zirgen::Zll
