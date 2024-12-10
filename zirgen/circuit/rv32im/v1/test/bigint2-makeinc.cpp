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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/NativeFormatting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "zirgen/Dialect/BigInt/Bytecode/encode.h"
#include "zirgen/Dialect/BigInt/Bytecode/file.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/circuit/bigint/elliptic_curve.h"

using namespace zirgen;

llvm::cl::opt<std::string>
    outFile(llvm::cl::Positional, llvm::cl::desc("<output file>"), llvm::cl::Required);

const APInt secp256k1_prime = APInt::getAllOnes(256) - APInt::getOneBitSet(256, 32) -
                              APInt::getOneBitSet(256, 9) - APInt::getOneBitSet(256, 8) -
                              APInt::getOneBitSet(256, 7) - APInt::getOneBitSet(256, 6) -
                              APInt::getOneBitSet(256, 4);
const APInt secp256k1_a(8, 0);
const APInt secp256k1_b(8, 7);
/*
// Base point
const APInt
    secp256k1_G_x(256, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16);
const APInt
    secp256k1_G_y(256, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16);
const APInt
    secp256k1_order(256, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
*/

int kArenaConst = 28;         // Reg T3 = x28
const uint32_t kArenaTmp = 2; // Reg SP = x2

struct PolyAtom {
  uint32_t arena;
  uint32_t offset;
  uint32_t size;
  bool doWrite;
  PolyAtom() : arena(0), offset(0), size(0) {}
  PolyAtom(uint32_t arena, uint32_t offset, uint32_t size, bool doWrite)
      : arena(arena), offset(offset), size(size), doWrite(doWrite) {}
  bool operator<(const PolyAtom& rhs) const {
    if (arena != rhs.arena) {
      return arena < rhs.arena;
    }
    if (offset != rhs.offset) {
      return offset < rhs.offset;
    }
    return size < rhs.size;
  }
};

struct PolyProd {
  std::map<PolyAtom, unsigned> atoms;
  PolyProd() {}
  PolyProd(const PolyAtom& atom) { atoms[atom] = 1; }
  bool operator<(const PolyProd& rhs) const { return atoms < rhs.atoms; }
  size_t degree() const {
    size_t tot = 0;
    for (const auto& atom : atoms) {
      tot += atom.second;
    }
    return tot;
  }
  PolyProd operator*(const PolyProd& rhs) const {
    PolyProd out = *this;
    for (const auto& atom : rhs.atoms) {
      out.atoms[atom.first] += atom.second;
    }
    return out;
  }
};

struct Polynomial {
  std::map<PolyProd, int> terms;
  Polynomial() {}
  Polynomial(PolyProd rhs) { terms[rhs] = 1; }
  size_t degree() const {
    size_t max = 0;
    for (const auto& term : terms) {
      max = std::max(max, term.first.degree());
    }
    return max;
  }
  void normalize() {
    for (auto it = terms.begin(); it != terms.end();) {
      if (it->second == 0) {
        it = terms.erase(it);
      } else {
        ++it;
      }
    }
  }
  Polynomial operator+(const Polynomial& rhs) const {
    Polynomial out = *this;
    for (const auto& term : rhs.terms) {
      out.terms[term.first] += term.second;
    }
    out.normalize();
    return out;
  }
  Polynomial operator-(const Polynomial& rhs) const {
    Polynomial out = *this;
    for (const auto& term : rhs.terms) {
      out.terms[term.first] -= term.second;
    }
    out.normalize();
    return out;
  }
  Polynomial operator*(const Polynomial& rhs) const {
    Polynomial out;
    for (const auto& lhsTerm : terms) {
      for (const auto& rhsTerm : rhs.terms) {
        auto prod = lhsTerm.first * rhsTerm.first;
        out.terms[prod] += lhsTerm.second * rhsTerm.second;
      }
    }
    out.normalize();
    return out;
  }
};

// TODO: Maybe don't copy;
namespace PolyOp {
constexpr size_t kNop = 0;
constexpr size_t kOpShift = 1;
constexpr size_t kOpSetTerm = 2;
constexpr size_t kOpAddTot = 3;
constexpr size_t kOpCarry1 = 4;
constexpr size_t kOpCarry2 = 5;
constexpr size_t kOpEqz = 6;
} // namespace PolyOp

namespace MemOp {
constexpr size_t kRead = 0;
constexpr size_t kWrite = 1;
constexpr size_t kNop = 2;
} // namespace MemOp

struct Flattener {
  std::vector<uint32_t> out;
  mlir::DenseSet<std::pair<uint32_t, uint32_t>> written;
  Flattener() {}
  void finalize() {
    // Add final nop to return
    out.push_back(MemOp::kNop << 28 | PolyOp::kNop);
  }
  void flatten(const PolyAtom& atom, bool doFinal, int coeff) {
    uint32_t memOp = MemOp::kRead;
    if (atom.doWrite) {
      auto key = std::make_pair(atom.arena, atom.offset);
      if (!written.count(key)) {
        memOp = MemOp::kWrite;
        written.insert(key);
      }
    }
    uint32_t finalPolyOp = (doFinal ? PolyOp::kOpAddTot : PolyOp::kOpSetTerm);
    for (size_t i = 0; i < atom.size; i++) {
      uint32_t polyOp = (i + 1 == atom.size ? finalPolyOp : PolyOp::kOpShift);
      out.push_back(memOp << 28 | polyOp << 24 | (coeff + 4) << 21 | atom.arena << 16 |
                    (atom.offset + atom.size - 1 - i));
    }
  }
  void flatten(const PolyProd& prod, int coeff) {
    std::vector<PolyAtom> atomsFlat;
    for (const auto& kvpAtom : prod.atoms) {
      for (size_t i = 0; i < kvpAtom.second; i++) {
        atomsFlat.push_back(kvpAtom.first);
      }
    }
    if (coeff > 3 || coeff < -3) {
      llvm::errs() << "TOOO: implement handing of high coefficients\n";
      throw std::runtime_error("Unimplemented");
    }
    if (atomsFlat.size() == 1) {
      flatten(atomsFlat[0], true, coeff);
    } else if (atomsFlat.size() == 2) {
      flatten(atomsFlat[0], false, coeff);
      flatten(atomsFlat[1], true, coeff);
    } else {
      llvm::errs() << "Invalid coefficent degree\n";
      throw std::runtime_error("Invalid degree");
    }
  }
  void flatten(const Polynomial& poly, BigInt::BigIntType bit) {
    for (const auto& kvp : poly.terms) {
      flatten(kvp.first, kvp.second);
    }
    size_t carryCount = (bit.getCoeffs() + 15) / 16;
    for (size_t i = 0; i < carryCount; i++) {
      uint32_t common = MemOp::kNop << 28 | (carryCount - 1 - i);
      out.push_back(PolyOp::kOpCarry1 << 24 | common);
      out.push_back(PolyOp::kOpCarry2 << 24 | common);
      if (i == carryCount - 1) {
        out.push_back(PolyOp::kOpEqz << 24 | common);
      } else {
        out.push_back(PolyOp::kOpShift << 24 | common);
      }
    }
  }
};

struct PolySplitState {
  PolySplitState() : neededForEq(0), neededForNondet(0) {}
  PolySplitState(int neededForEq, int neededForNondet)
      : neededForEq(neededForEq), neededForNondet(neededForNondet) {}

  int neededForEq;
  int neededForNondet;
  PolyAtom atom;

  void apply(const PolySplitState& rhs) {
    neededForEq += rhs.neededForEq;
    neededForNondet += rhs.neededForNondet;
  }
};

Polynomial eval(DenseMap<Value, PolySplitState>& state, Operation* origOp) {
  return TypeSwitch<Operation*, Polynomial>(origOp)
      .Case<BigInt::AddOp>([&](auto op) {
        Polynomial evalLhs = eval(state, op.getLhs().getDefiningOp());
        Polynomial evalRhs = eval(state, op.getRhs().getDefiningOp());
        return evalLhs + evalRhs;
      })
      .Case<BigInt::SubOp>([&](auto op) {
        Polynomial evalLhs = eval(state, op.getLhs().getDefiningOp());
        Polynomial evalRhs = eval(state, op.getRhs().getDefiningOp());
        return evalLhs - evalRhs;
      })
      .Case<BigInt::MulOp>([&](auto op) {
        Polynomial evalLhs = eval(state, op.getLhs().getDefiningOp());
        Polynomial evalRhs = eval(state, op.getRhs().getDefiningOp());
        return evalLhs * evalRhs;
      })
      .Case<BigInt::ConstOp,
            BigInt::LoadOp,
            BigInt::NondetRemOp,
            BigInt::NondetQuotOp,
            BigInt::NondetInvOp>([&](auto op) { return Polynomial(PolyProd(state[op].atom)); })
      .Default([&](auto op) {
        llvm::errs() << "Invalid poly op: " << *op << "\n";
        throw std::runtime_error("Invalid op in split");
        return Polynomial();
      });
}

void polySplit(mlir::func::FuncOp func) {
  DenseMap<Value, PolySplitState> state;

  Block* block = &func.getBody().front();
  std::vector<uint32_t> constData;
  int offsetTmp = 0;
  int offsetConst = 0;
  // Do analysis to split into polynomial evaluations + nondet computations
  for (auto& origOp : llvm::reverse(block->without_terminator())) {
    TypeSwitch<Operation*>(&origOp)
        .Case<BigInt::EqualZeroOp>([&](auto op) { state[op.getIn()].apply(PolySplitState(1, 0)); })
        .Case<BigInt::StoreOp>([&](auto op) {
          auto bit = dyn_cast<BigInt::BigIntType>(op.getIn().getType());
          size_t size = (bit.getCoeffs() + 15) / 16;
          if (bit.getMaxPos() > 255 || bit.getMaxNeg() > 0) {
            llvm::errs() << "Store must be of a normalized BigInt";
            throw std::runtime_error("Invalid store");
          }
          state[op.getIn()].apply(PolySplitState(0, 1));
          state[op.getIn()].atom = PolyAtom(op.getArena(), op.getOffset(), size, true);
        })
        .Case<BigInt::NondetRemOp, BigInt::NondetQuotOp, BigInt::NondetInvOp>([&](auto op) {
          size_t size = (dyn_cast<BigInt::BigIntType>(op.getType()).getCoeffs() + 15) / 16;
          if (state[op].neededForEq && state[op].atom.arena == 0) {
            state[op].atom = PolyAtom(kArenaTmp, offsetTmp, size, true);
            offsetTmp += size;
          }
          if (state[op].neededForEq || state[op].neededForNondet) {
            state[op].neededForNondet = 1;
            state[op.getLhs()].apply(PolySplitState(0, 1));
            state[op.getRhs()].apply(PolySplitState(0, 1));
          }
        })
        .Case<BigInt::AddOp, BigInt::SubOp, BigInt::MulOp>([&](auto op) {
          if (state[op].neededForEq || state[op].neededForNondet) {
            state[op.getLhs()].apply(state[op]);
            state[op.getRhs()].apply(state[op]);
          }
        })
        .Case<BigInt::ConstOp>([&](auto op) {
          size_t size = (dyn_cast<BigInt::BigIntType>(op.getType()).getCoeffs() + 15) / 16;
          APInt value = op.getValue();
          value = value.zext(128 * size);
          for (size_t i = 0; i < size * 4; i++) {
            constData.push_back(value.extractBitsAsZExtValue(32, i * 32));
          }
          state[op].atom = PolyAtom(kArenaConst, offsetConst, size, false);
          offsetConst += size;
        })
        .Case<BigInt::LoadOp>([&](auto op) {
          size_t size = (dyn_cast<BigInt::BigIntType>(op.getType()).getCoeffs() + 15) / 16;
          state[op].atom = PolyAtom(op.getArena(), op.getOffset(), size, false);
        })
        .Case<mlir::func::ReturnOp>([&](auto op) {})
        .Default([&](auto op) {
          llvm::errs() << "Invalid op: " << *op << "\n";
          throw std::runtime_error("Invalid op in split");
        });
  }
  // Gather all the polynomials to evaluate
  Flattener flattener;
  for (auto& origOp : *block) {
    if (auto op = dyn_cast<BigInt::EqualZeroOp>(&origOp)) {
      // Compute the polynomial
      Polynomial p = eval(state, op.getIn().getDefiningOp());
      auto bit = dyn_cast<BigInt::BigIntType>(op.getIn().getType());
      flattener.flatten(p, bit);
    }
  }
  flattener.finalize();
  // Now, destroy all the unneeded ops
  for (auto& op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    if (dyn_cast<BigInt::EqualZeroOp>(op)) {
      op.erase();
    }
    if (op.getNumResults() == 1 && !state[op.getResult(0)].neededForNondet) {
      op.erase();
    }
  }
  // Add in 'stores' for any tmp values
  OpBuilder builder(block->getTerminator());
  for (const auto& kvp : state) {
    if (kvp.second.atom.arena == kArenaTmp) {
      builder.create<BigInt::StoreOp>(func.getLoc(), kvp.first, kArenaTmp, kvp.second.atom.offset);
    }
  }
  // Make into a flat buffer in the form of:
  // 0: size of witgen code
  // 1: size of verification code
  // 2: size of constants
  // 3: size of tmps
  // [witgen code]
  // [verification code]
  // [constants]
  auto prog = BigInt::Bytecode::encode(func);
  size_t progSizeBytes = BigInt::Bytecode::tell(*prog);
  assert(progSizeBytes % 4 == 0);
  size_t progSizeWords = progSizeBytes / 4;
  std::vector<uint32_t> flat(progSizeWords + 4); // 4 'header' values
  BigInt::Bytecode::write(*prog, &flat[4], progSizeBytes);
  flat[0] = progSizeWords;
  flat[1] = flattener.out.size();
  flat[2] = constData.size();
  flat[3] = offsetTmp * 4;

  // Append the flat verifier
  flat.insert(flat.end(), flattener.out.begin(), flattener.out.end());
  flat.insert(flat.end(), constData.begin(), constData.end());

  // Now print everything as blob ready to include into c code
  std::error_code ignore;
  llvm::raw_fd_ostream out(outFile, ignore);
  for (size_t i = 0; i < flat.size(); i++) {
    if (i != 0 && i % 8 == 0) {
      out << "\n";
    }
    write_hex(out, flat[i], llvm::HexPrintStyle::PrefixLower, 10);
    out << ", ";
  }
  out << "\n";
}

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "bigint2-makeinc");

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<BigInt::BigIntDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto loc = mlir::UnknownLoc::get(&ctx);
  auto module = mlir::ModuleOp::create(loc);
  auto builder = mlir::OpBuilder(&ctx);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto funcType = FunctionType::get(&ctx, {}, {});
  auto func = builder.create<func::FuncOp>(loc, "foo", funcType);
  builder.setInsertionPointToStart(func.addEntryBlock());

  mlir::Type int256Type = builder.getIntegerType(256);
  auto primeAttr = builder.getIntegerAttr(int256Type, secp256k1_prime);
  auto prime = builder.create<BigInt::ConstOp>(loc, primeAttr);
  mlir::Type int8Type = builder.getIntegerType(8);
  auto aAttr = builder.getIntegerAttr(int8Type, secp256k1_a);
  auto a = builder.create<BigInt::ConstOp>(loc, aAttr);
  auto bAttr = builder.getIntegerAttr(int8Type, secp256k1_b);
  auto b = builder.create<BigInt::ConstOp>(loc, bAttr);

  auto curve = std::make_shared<BigInt::EC::WeierstrassCurve>(prime, a, b);

  auto a_x = builder.create<BigInt::LoadOp>(loc, 256, 11, 0);
  auto a_y = builder.create<BigInt::LoadOp>(loc, 256, 11, 2);
  auto pt = BigInt::EC::AffinePt(a_x, a_y, curve);
  auto doubled = BigInt::EC::doub(builder, loc, pt);
  builder.create<BigInt::StoreOp>(loc, doubled.x(), 12, 0);
  builder.create<BigInt::StoreOp>(loc, doubled.y(), 12, 2);
  builder.create<func::ReturnOp>(loc);

  // Remove reduce
  PassManager pm(&ctx);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(BigInt::createLowerReducePass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }

  polySplit(func);
}
