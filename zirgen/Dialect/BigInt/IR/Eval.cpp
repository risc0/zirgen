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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "risc0/core/util.h"
#include "risc0/fp/fpext.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace mlir;
using namespace risc0;

#define DEBUG_TYPE "bigint"

namespace zirgen::BigInt {

namespace {

APInt toAPInt(const BytePoly& val) {
  size_t maxBits = val.size() * kBitsPerCoeff + 32;
  APInt out(maxBits, 0);
  APInt mul(maxBits, 1);
  for (size_t i = 0; i < val.size(); i++) {
    out += mul.smul_sat(APInt(maxBits, val[i], true));
    mul = mul * (1 << kBitsPerCoeff);
  }
  return out;
}

} // namespace

BytePoly fromAPInt(APInt value, size_t coeffs) {
  BytePoly out(coeffs);
  size_t maxWidth = std::max(value.getBitWidth(), uint32_t(coeffs * kBitsPerCoeff));
  value = value.zext(maxWidth);
  for (size_t i = 0; i < coeffs; i++) {
    out[i] = value.extractBits(kBitsPerCoeff, i * kBitsPerCoeff).getLimitedValue();
  }
  LLVM_DEBUG({ dbgs() << "fromAPInt: " << toAPInt(out) << "\n"; });
  return out;
}

namespace {
BytePoly add(const BytePoly& lhs, const BytePoly& rhs) {
  BytePoly out(std::max(lhs.size(), rhs.size()));
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] += lhs[i];
  }
  for (size_t i = 0; i < rhs.size(); i++) {
    out[i] += rhs[i];
  }
  LLVM_DEBUG({ dbgs() << "add: " << toAPInt(out) << "\n"; });
  return out;
}

BytePoly sub(const BytePoly& lhs, const BytePoly& rhs) {
  BytePoly out(std::max(lhs.size(), rhs.size()));
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] += lhs[i];
  }
  for (size_t i = 0; i < rhs.size(); i++) {
    out[i] -= rhs[i];
  }
  LLVM_DEBUG({ dbgs() << "sub: " << toAPInt(out) << "\n"; });
  return out;
}

BytePoly mul(const BytePoly& lhs, const BytePoly& rhs) {
  BytePoly out(lhs.size() + rhs.size() - 1);
  for (size_t i = 0; i < lhs.size(); i++) {
    for (size_t j = 0; j < rhs.size(); j++) {
      out[i + j] += lhs[i] * rhs[j];
    }
  }
  LLVM_DEBUG({ dbgs() << "mul: " << toAPInt(out) << "\n"; });
  return out;
}

BytePoly nondetQuot(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  auto lhsInt = toAPInt(lhs);
  auto rhsInt = toAPInt(rhs);
  size_t maxSize = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
  auto quot = lhsInt.zext(maxSize).udiv(rhsInt.zext(maxSize));
  LLVM_DEBUG({ dbgs() << "quot: " << quot << "\n"; });
  return fromAPInt(quot, coeffs);
}

BytePoly nondetRem(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  auto lhsInt = toAPInt(lhs);
  auto rhsInt = toAPInt(rhs);
  size_t maxSize = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
  auto rem = lhsInt.zext(maxSize).urem(rhsInt.zext(maxSize));
  LLVM_DEBUG({ dbgs() << "rem: " << rem << "\n"; });
  return fromAPInt(rem, coeffs);
}

BytePoly nondetInv(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  // Uses the formula n^(p-2) * n = 1  (mod p) to invert `lhs` (mod `rhs`)
  // (via the square and multiply technique)
  auto lhsInt = toAPInt(lhs);
  auto rhsInt = toAPInt(rhs);
  size_t maxSize = rhsInt.getBitWidth();
  APInt inv(2 * maxSize,
            1); // Initialize inverse to zero, twice the width of `prime` to allow multiplication
  APInt sqr(lhsInt);              // Will be repeatedly squared
  APInt position(2 * maxSize, 1); // Bit at `idx` will be 1, other bits will be 0
  sqr = sqr.zext(2 * maxSize);
  rhsInt = rhsInt.zext(2 * maxSize);
  APInt exp = rhsInt - 2;
  for (size_t idx = 0; idx < maxSize; idx++) {
    if (exp.intersects(position)) {
      // multiply in the current power of n (i.e., n^(2^idx))
      inv = (inv * sqr).urem(rhsInt);
    }
    position <<= 1;                 // increment the bit position to test in `exp`
    sqr = (sqr * sqr).urem(rhsInt); // square `sqr` to increment to `n^(2^(idx+1))`
  }
  inv = inv.trunc(maxSize); // We don't need the extra space used as multiply buffer
  LLVM_DEBUG({ dbgs() << "inv (mod " << rhsInt << "): " << inv << "\n"; });
  return fromAPInt(inv, coeffs);
}

void printEval(const std::string& message, BytePoly poly) {
  risc0::FpExt tot(0);
  risc0::FpExt mul(1);
  risc0::FpExt z(0, 1, 0, 0);
  for (size_t i = 0; i < poly.size(); i++) {
    tot = tot + mul * (poly[i] >= 0 ? Fp(poly[i]) : Fp((1 << 27) * 15 + 1 + poly[i]));
    mul *= z;
    // dbgs() << " " << poly[i];
  }
  // dbgs() << "\n";
  auto& e = tot.elems;
  dbgs() << message << "(" << e[0].asUInt32() << ", " << e[1].asUInt32() << ", " << e[2].asUInt32()
         << ", " << e[3].asUInt32() << ")\n";
}

} // namespace

Digest computeDigest(std::vector<BytePoly> witness, size_t groupCount) {
  std::vector<uint32_t> words;
  std::array<uint32_t, kCoeffsPerPoly> cur = {0};
  size_t group = 0;
  for (size_t i = 0; i < witness.size(); i++) {
    for (size_t j = 0; j < witness[i].size(); j += kCoeffsPerPoly) {
      for (size_t k = 0; k < kCoeffsPerPoly; k++) {
        cur[k] *= 256;
        if (j + k < witness[i].size()) {
          cur[k] += witness[i][j + k];
        }
      }
      group++;
      if (group == groupCount) {
        for (size_t k = 0; k < kCoeffsPerPoly; k++) {
          words.push_back(cur[k]);
          cur[k] = 0;
        }
        group = 0;
      }
    }
  }
  if (group != 0) {
    for (size_t k = 0; k < kCoeffsPerPoly; k++) {
      words.push_back(cur[k]);
    }
  }
  return poseidon2Hash(words.data(), words.size());
}

struct Def {
  virtual APInt load(uint32_t arena, uint32_t offset, uint32_t count) = 0;
  virtual void store(uint32_t arena, uint32_t offset, uint32_t count, APInt val) = 0;
};

EvalOutput eval(func::FuncOp inFunc, BigIntIO& io, bool computeZ) {
  EvalOutput ret;

  llvm::DenseMap<Value, BytePoly> polys;

  for (Operation& origOp : inFunc.getBody().front().without_terminator()) {
    llvm::TypeSwitch<Operation*>(&origOp)
        .Case<DefOp>([&](auto op) {
          APInt val = io.load(0, op.getLabel(), 0);
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          auto poly = fromAPInt(val, coeffs);
          polys[op.getOut()] = poly;
          if (op.getIsPublic()) {
            ret.publicWitness.push_back(poly);
          } else {
            ret.privateWitness.push_back(poly);
          }
          LLVM_DEBUG({ printEval("Def op: ", poly); });
        })
        .Case<ConstOp>([&](auto op) {
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          auto poly = fromAPInt(op.getValue(), coeffs);
          polys[op.getOut()] = poly;
          ret.constantWitness.push_back(poly);
        })
        .Case<LoadOp>([&](auto op) {
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          uint32_t count = (coeffs + 15) / 16;
          APInt val = io.load(op.getArena(), op.getOffset(), count);
          auto poly = fromAPInt(val, coeffs);
          polys[op.getOut()] = poly;
        })
        .Case<StoreOp>([&](auto op) {
          uint32_t coeffs = op.getIn().getType().getCoeffs();
          uint32_t count = (coeffs + 15) / 16;
          auto poly = polys[op.getIn()];
          auto val = toAPInt(poly);
          io.store(op.getArena(), op.getOffset(), count, val.trunc(coeffs * 8));
        })
        .Case<AddOp>(
            [&](auto op) { polys[op.getOut()] = add(polys[op.getLhs()], polys[op.getRhs()]); })
        .Case<SubOp>(
            [&](auto op) { polys[op.getOut()] = sub(polys[op.getLhs()], polys[op.getRhs()]); })
        .Case<MulOp>(
            [&](auto op) { polys[op.getOut()] = mul(polys[op.getLhs()], polys[op.getRhs()]); })
        .Case<NondetRemOp>([&](auto op) {
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          auto poly = nondetRem(polys[op.getLhs()], polys[op.getRhs()], coeffs);
          polys[op.getOut()] = poly;
          ret.privateWitness.push_back(poly);
        })
        .Case<NondetQuotOp>([&](auto op) {
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          auto poly = nondetQuot(polys[op.getLhs()], polys[op.getRhs()], coeffs);
          polys[op.getOut()] = poly;
          ret.privateWitness.push_back(poly);
        })
        .Case<NondetInvOp>([&](auto op) {
          uint32_t coeffs = op.getOut().getType().getCoeffs();
          auto poly = nondetInv(polys[op.getLhs()], polys[op.getRhs()], coeffs);
          polys[op.getOut()] = poly;
          ret.privateWitness.push_back(poly);
        })
        .Case<EqualZeroOp>([&](auto op) {
          auto poly = polys[op.getIn()];
          if (toAPInt(poly) != 0) {
            errs() << "EQZ is nonzero: " << toAPInt(poly) << "\n";
            throw std::runtime_error("NONZERO");
          }
          uint32_t coeffs = op.getIn().getType().getCoeffs();
          int32_t carryOffset = op.getIn().getType().getCarryOffset();
          size_t carryBytes = op.getIn().getType().getCarryBytes();
          std::vector<BytePoly> carryPolys;
          for (size_t i = 0; i < carryBytes; i++) {
            carryPolys.emplace_back(coeffs);
          };
          int32_t carry = 0;
          for (size_t i = 0; i < coeffs; i++) {
            carry = (poly[i] + carry) / 256;
            uint32_t carryU = carry + carryOffset;
            carryPolys[0][i] = carryU & 0xff;
            if (carryBytes > 1) {
              carryPolys[1][i] = ((carryU >> 8) & 0xff);
            }
            if (carryBytes > 2) {
              carryPolys[2][i] = ((carryU >> 16) & 0xff);
              carryPolys[3][i] = ((carryU >> 16) & 0xff) * 4;
            }
          }
          // Verify carry computation
          BytePoly bigCarry(coeffs);
          for (size_t i = 0; i < coeffs; i++) {
            bigCarry[i] = carryPolys[0][i];
            if (carryBytes > 1) {
              bigCarry[i] += 256 * carryPolys[1][i];
            }
            if (carryBytes > 2) {
              bigCarry[i] += 65536 * carryPolys[2][i];
            }
            bigCarry[i] -= carryOffset;
          }
          for (size_t i = 0; i < coeffs; i++) {
            int32_t shouldBeZero = poly[i];
            shouldBeZero -= 256 * bigCarry[i];
            if (i != 0) {
              shouldBeZero += bigCarry[i - 1];
            }
            if (shouldBeZero != 0) {
              errs() << "Invalid carry computation\n";
              throw std::runtime_error("CARRY");
            }
          }
          // Store the results
          for (size_t i = 0; i < carryPolys.size(); i++) {
            ret.privateWitness.push_back(carryPolys[i]);
          }
        })
        .Default([&](Operation* op) {
          errs() << *op << "\n";
          throw std::runtime_error("Unknown op in eval");
        });
  }

  if (computeZ) {
    Digest publicDigest = computeDigest(ret.publicWitness, 1);
    LLVM_DEBUG({ dbgs() << "publicDigest: " << publicDigest << "\n"; });
    Digest privateDigest = computeDigest(ret.privateWitness, 3);
    LLVM_DEBUG({ dbgs() << "privateDigest: " << privateDigest << "\n"; });
    Digest folded = poseidon2HashPair(publicDigest, privateDigest);
    LLVM_DEBUG({ dbgs() << "folded: " << folded << "\n"; });

    // Now, compute the value of Z
    Poseidon2Rng rng;
    rng.mix(folded);
    for (size_t i = 0; i < 4; i++) {
      ret.z[i] = rng.generateFp();
    }
  }

  return ret;
}

namespace {

struct DefBigIntIO : public BigIntIO {
  ArrayRef<APInt> witnessValues;
  APInt load(uint32_t arena, uint32_t offset, uint32_t count) override {
    assert(arena == 0);
    assert(count == 0);
    assert(offset < witnessValues.size());
    return witnessValues[offset];
  }
  void store(uint32_t arena, uint32_t offset, uint32_t count, APInt val) override {
    throw std::runtime_error("Unimplemented");
  }
};

} // namespace

EvalOutput eval(func::FuncOp inFunc, ArrayRef<APInt> witnessValues) {
  DefBigIntIO io;
  io.witnessValues = witnessValues;
  return eval(inFunc, io, true);
}

namespace {

void printPolys(llvm::raw_ostream& os, llvm::ArrayRef<BytePoly> polys) {
  if (!polys.empty())
    os << "  ";
  interleave(
      polys,
      os,
      [&](const BytePoly& poly) {
        for (auto byte : poly) {
          if (byte > 255 || byte < 0) {
            os << "(" << byte << "?)";
          } else {
            os << hexdigit(byte >> 4, /*lowercase=*/true)
               << hexdigit(byte & 0xF, /*lowercase=*/true);
          }
        }
      },
      ",\n  ");
}

} // namespace

void EvalOutput::print(llvm::raw_ostream& os) const {
  os << "z = [";
  llvm::interleaveComma(z, os);
  os << "]\n";
  os << "constantWitness = [\n";
  printPolys(os, constantWitness);
  os << "]\n";
  os << "publicWitness = [\n";
  printPolys(os, publicWitness);
  os << "]\n";
  os << "privateWitness = [\n";
  printPolys(os, privateWitness);
  os << "]\n";
}

} // namespace zirgen::BigInt
