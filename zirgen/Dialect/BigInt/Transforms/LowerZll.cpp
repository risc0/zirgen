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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "zirgen/Dialect/BigInt/Transforms/PassDetail.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/compiler/zkp/hash.h"

using namespace mlir;

namespace zirgen::BigInt {

namespace {

void lower(func::FuncOp inFunc) {
  // Do module + function boilerplate
  auto ctx = inFunc.getContext();
  auto builder = OpBuilder(inFunc);
  auto loc = inFunc.getLoc();
  auto extType = Zll::ValType::getExtensionType(ctx);
  auto baseType = Zll::ValType::getBaseType(ctx);
  auto funcType = FunctionType::get(
      ctx,
      {Zll::BufferType::get(
           ctx, baseType, 32 /* recursion global output buffer size */, Zll::BufferKind::Global),
       Iop::IOPType::get(ctx)},
      {});
  auto func = builder.create<func::FuncOp>(loc, inFunc.getName(), funcType);
  size_t iters = getIterationCount(inFunc);
  assert(iters > 0);
  setIterationCount(func, iters);
  auto block = func.addEntryBlock();
  auto buf = block->getArgument(0);
  auto iop = block->getArgument(1);
  builder.setInsertionPointToEnd(block);

  // Compute size of various witnesses
  size_t countConst = 0;
  size_t countPrivate = 0;
  size_t countPublic = 0;
  for (Operation& origOp : inFunc.getBody().front().without_terminator()) {
    llvm::TypeSwitch<Operation*>(&origOp)
        .Case<DefOp>([&](auto op) {
          if (op.getIsPublic()) {
            countPublic += op.getOut().getType().getNormalWitnessSize();
          } else {
            countPrivate += op.getOut().getType().getNormalWitnessSize();
          }
        })
        .Case<ConstOp>([&](auto op) { countConst += op.getOut().getType().getNormalWitnessSize(); })
        .Case<NondetRemOp, NondetQuotOp, NondetInvOp>(
            [&](auto op) { countPrivate += op.getOut().getType().getNormalWitnessSize(); })
        .Case<EqualZeroOp>(
            [&](auto op) { countPrivate += op.getIn().getType().getCarryWitnessSize(); });
  }

  llvm::errs() << "countPublic: " << countPublic << "\n";

  // Read control root and store it in globals.  We just pass it on
  // here; it's verified by the resolve predicate.
  {
    Type digestType = Zll::DigestType::get(builder.getContext(), Zll::DigestKind::Default);
    std::vector<Type> types(1, digestType);
    auto readOp = builder.create<Iop::ReadOp>(loc, types, iop, /*flip=*/false);
    builder.create<Zll::SetGlobalDigestOp>(loc, buf, 0, readOp.getOuts()[0]);
  }

  auto zero = builder.create<Zll::ConstOp>(loc, baseType, 0);
  auto one = builder.create<Zll::ConstOp>(loc, baseType, 1);
  auto four = builder.create<Zll::ConstOp>(loc, baseType, 4);
  auto byte = builder.create<Zll::ConstOp>(loc, baseType, 1 << kBitsPerCoeff);

  llvm::errs() << "verifying " << iters << " claims\n";
  std::vector<mlir::Value> digestVals;
  for (size_t iter = 0; iter != iters; ++iter) {
    // Read the value of z
    auto z = builder.create<Iop::ReadOp>(loc, TypeRange{extType}, iop, false).getOuts()[0];
    auto byteMinusZ = builder.create<Zll::SubOp>(loc, byte, z);

    // Generate polynomials of the form 1 + z + z^2 + ... z^n
    // CSE will make this very cheap when reused.
    auto manyOnes = [&](size_t count) {
      Value tot = zero;
      Value curMul = one;
      for (size_t i = 0; i < count; i++) {
        tot = builder.create<Zll::AddOp>(loc, tot, curMul);
        curMul = builder.create<Zll::MulOp>(loc, curMul, z);
      }
      return tot;
    };

    // Make the various witnesses + set starting location
    auto cbConst = builder.create<Zll::HashCheckedBytesOp>(loc, z, countConst);
    auto cbPublic = builder.create<Zll::HashCheckedBytesPublicOp>(loc, z, countPublic);
    auto cbPrivate = builder.create<Zll::HashCheckedBytesOp>(loc, z, countPrivate);
    size_t curConst = 0;
    size_t curPublic = 0;
    size_t curPrivate = 0;

    // Define a way to extract a polynomial from a witness
    auto extractPoly = [&](mlir::ValueRange evals, size_t& cur, BigIntType type) {
      size_t size = type.getNormalWitnessSize();
      Value tot = zero;
      for (size_t i = 0; i < size; i++) {
        Value polyPart = evals[cur++];
        Value powZ = builder.create<Zll::PowOp>(loc, z, i * kCoeffsPerPoly);
        Value prod = builder.create<Zll::MulOp>(loc, polyPart, powZ);
        tot = builder.create<Zll::AddOp>(loc, tot, prod);
      }
      return tot;
    };

    // Make a map from input values to output values
    llvm::DenseMap<Value, Value> valMap;

    // Keep track of physical values of constant witness
    std::vector<BytePoly> constantWitness;

    // Let's go!
    for (Operation& origOp : inFunc.getBody().front().without_terminator()) {
      llvm::TypeSwitch<Operation*>(&origOp)
          .Case<DefOp>([&](auto op) {
            if (op.getIsPublic()) {
              valMap[op.getOut()] =
                  extractPoly(cbPublic.getEvaluations(), curPublic, op.getOut().getType());
            } else {
              valMap[op.getOut()] =
                  extractPoly(cbPublic.getEvaluations(), curPrivate, op.getOut().getType());
            }
            // builder.create<Zll::ExternOp>(loc, TypeRange{}, ValueRange{valMap[op.getOut()]},
            // "log", "DefOp: %p");
          })
          .Case<ConstOp>([&](auto op) {
            uint32_t coeffs = op.getOut().getType().getCoeffs();
            auto poly = fromAPInt(op.getValue(), coeffs);
            constantWitness.push_back(poly);
            valMap[op.getOut()] =
                extractPoly(cbConst.getEvaluations(), curConst, op.getOut().getType());
          })
          .Case<AddOp>([&](auto op) {
            valMap[op.getOut()] =
                builder.create<Zll::AddOp>(loc, valMap[op.getLhs()], valMap[op.getRhs()]);
          })
          .Case<SubOp>([&](auto op) {
            valMap[op.getOut()] =
                builder.create<Zll::SubOp>(loc, valMap[op.getLhs()], valMap[op.getRhs()]);
          })
          .Case<MulOp>([&](auto op) {
            valMap[op.getOut()] =
                builder.create<Zll::MulOp>(loc, valMap[op.getLhs()], valMap[op.getRhs()]);
          })
          .Case<NondetRemOp, NondetQuotOp, NondetInvOp>([&](auto op) {
            valMap[op.getOut()] =
                extractPoly(cbPrivate.getEvaluations(), curPrivate, op.getOut().getType());
          })
          .Case<EqualZeroOp>([&](auto op) {
            std::vector<Value> parts(op.getIn().getType().getCarryBytes());
            for (size_t i = 0; i < parts.size(); i++) {
              parts[i] = extractPoly(cbPrivate.getEvaluations(), curPrivate, op.getIn().getType());
            }
            // If we have 4 bytes, bytes 3 + 4 should be related
            if (parts.size() == 4) {
              Value mul4 = builder.create<Zll::MulOp>(loc, parts[2], four);
              Value diff = builder.create<Zll::SubOp>(loc, mul4, parts[3]);
              builder.create<Zll::EqualZeroOp>(loc, diff);
            }
            // Build the 'full width' version of the carry polynomial
            Value tot = zero;
            for (size_t i = 0; i < std::min(parts.size(), size_t(3)); i++) {
              Value powByte = builder.create<Zll::PowOp>(loc, byte, i);
              Value prod = builder.create<Zll::MulOp>(loc, parts[i], powByte);
              tot = builder.create<Zll::AddOp>(loc, tot, prod);
            }
            // Subtract the offset
            Value offset = builder.create<Zll::ConstOp>(loc, op.getIn().getType().getCarryOffset());
            Value ones = manyOnes(op.getIn().getType().getCoeffs());
            Value allOffsets = builder.create<Zll::MulOp>(loc, ones, offset);
            tot = builder.create<Zll::SubOp>(loc, tot, allOffsets);
            // Multiply by (256 - z)
            Value carryImpact = builder.create<Zll::MulOp>(loc, tot, byteMinusZ);
            // Check is delta is zero
            Value diff = builder.create<Zll::SubOp>(loc, carryImpact, valMap[op.getIn()]);
            builder.create<Zll::EqualZeroOp>(loc, diff);
          });
    }

    // Hash the constant witness
    Digest constDigest = computeDigest(constantWitness);
    // Convert to witness elements
    auto witvec = poseidon2HashSuite()->encode(constDigest, 8);
    std::vector<Value> valvec;
    for (size_t i = 0; i < 8; i++) {
      valvec.push_back(builder.create<Zll::ConstOp>(loc, witvec[i]));
    }
    auto constDigestVal = builder.create<Zll::IntoDigestOp>(
        loc, Zll::DigestType::get(ctx, Zll::DigestKind::Poseidon2), valvec);
    // Match it against the computed value
    builder.create<Zll::HashAssertEqOp>(loc, constDigestVal, cbConst.getDigest());

    // Combine the public + private
    auto folded = builder.create<Zll::HashFoldOp>(loc, cbPublic.getDigest(), cbPrivate.getDigest());
    // builder.create<Zll::ExternOp>(loc, TypeRange{}, ValueRange{cbConst.getDigest()}, "log",
    // "Const: %h"); builder.create<Zll::ExternOp>(loc, TypeRange{},
    // ValueRange{cbPublic.getDigest()}, "log", "Public: %h"); builder.create<Zll::ExternOp>(loc,
    // TypeRange{}, ValueRange{cbPrivate.getDigest()}, "log", "Private: %h");
    // builder.create<Zll::ExternOp>(loc, TypeRange{}, ValueRange{folded}, "log", "Folded: %h");

    // Commit to IOP
    builder.create<Iop::CommitOp>(loc, iop, folded);
    // Use IOP to select a random Z
    Value newZ = builder.create<Iop::RngValOp>(loc, extType, iop);
    // builder.create<Zll::ExternOp>(loc, TypeRange{}, ValueRange{newZ}, "log", "newZ: %p");
    // Compare to the z I initially read
    Value diffZ = builder.create<Zll::SubOp>(loc, z, newZ);
    builder.create<Zll::EqualZeroOp>(loc, diffZ);

    digestVals.push_back(cbPublic.getPubDigest());
  }

  // Write the public commitment list out so that the digest matches
  // that returned by risc0_binfmt::tagged_list
  std::vector<mlir::Value> zeroDigestVals(16, zero);
  mlir::Value listDigest = builder.create<Zll::IntoDigestOp>(
      loc, Zll::DigestType::get(ctx, Zll::DigestKind::Sha256), zeroDigestVals);

  while (!digestVals.empty()) {
    listDigest = builder.create<Zll::TaggedStructOp>(
        loc, "risc0.BigIntClaims", ValueRange{digestVals.back(), listDigest}, ValueRange{});
    digestVals.pop_back();
  }

  builder.create<Zll::SetGlobalDigestOp>(loc, buf, 1, listDigest);

  // End the function
  builder.create<func::ReturnOp>(loc);
}

struct LowerZllPass : public LowerZllBase<LowerZllPass> {
  void runOnOperation() override {
    SmallVector<func::FuncOp> toErase;
    getOperation().walk([&](func::FuncOp func) {
      lower(func);
      toErase.push_back(func);
    });
    for (auto op : toErase) {
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerZllPass() {
  return std::make_unique<LowerZllPass>();
}

} // namespace zirgen::BigInt
