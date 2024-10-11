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


// Do not include any LLVM or MLIR headers!
// This is meant to be a standalone reimplementation of the canonical eval
// function which is implemented in zirgen/Dialect/BigInt/IR/Eval.cpp, but
// this one does NOT depend on LLVM or MLIR: only the C standard library.
#include <stdexcept>
#include "zirgen/compiler/zkp/poseidon2.h"
#include "zirgen/Dialect/BigInt/Bytecode/eval.h"

namespace zirgen::BigInt::Bytecode {

// We must unfortunately redefine these constants whose original definitions
// live in zirgen/Dialect/BigInt/IR/BigInt.h
constexpr size_t kBitsPerCoeff = 8;
constexpr size_t kCoeffsPerPoly = 16;

BytePoly fromBQInt(BQInt value, size_t coeffs) {
  BytePoly out(coeffs);
  size_t maxWidth = std::max(value.getBitWidth(), uint32_t(coeffs * kBitsPerCoeff));
  value = value.zext(maxWidth);
  for (size_t i = 0; i < coeffs; i++) {
    out[i] = value.extractBits(kBitsPerCoeff, i * kBitsPerCoeff).getLimitedValue();
  }
  return out;
}

namespace {

BQInt toBQInt(const BytePoly& val) {
  size_t maxBits = val.size() * kBitsPerCoeff + 32;
  BQInt out(maxBits, 0);
  BQInt mul(maxBits, 1);
  for (size_t i = 0; i < val.size(); i++) {
    out += mul.smul_sat(BQInt(maxBits, val[i], true));
    mul = mul * (1 << kBitsPerCoeff);
  }
  return out;
}

BytePoly add(const BytePoly& lhs, const BytePoly& rhs) {
  BytePoly out(std::max(lhs.size(), rhs.size()));
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] += lhs[i];
  }
  for (size_t i = 0; i < rhs.size(); i++) {
    out[i] += rhs[i];
  }
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
  return out;
}

BytePoly mul(const BytePoly& lhs, const BytePoly& rhs) {
  BytePoly out(lhs.size() + rhs.size());
  for (size_t i = 0; i < lhs.size(); i++) {
    for (size_t j = 0; j < rhs.size(); j++) {
      out[i + j] += lhs[i] * rhs[j];
    }
  }
  return out;
}

BytePoly nondetQuot(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  auto lhsInt = toBQInt(lhs);
  auto rhsInt = toBQInt(rhs);
  size_t maxSize = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
  auto quot = lhsInt.zext(maxSize).udiv(rhsInt.zext(maxSize));
  return fromBQInt(quot, coeffs);
}

BytePoly nondetRem(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  auto lhsInt = toBQInt(lhs);
  auto rhsInt = toBQInt(rhs);
  size_t maxSize = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
  auto rem = lhsInt.zext(maxSize).urem(rhsInt.zext(maxSize));
  return fromBQInt(rem, coeffs);
}

BytePoly nondetInvMod(const BytePoly& lhs, const BytePoly& rhs, size_t coeffs) {
  // Uses the formula n^(p-2) * n = 1  (mod p) to invert `lhs` (mod `rhs`)
  // (via the square and multiply technique)
  auto lhsInt = toBQInt(lhs);
  auto rhsInt = toBQInt(rhs);
  size_t maxSize = rhsInt.getBitWidth();
  BQInt inv(2 * maxSize,
            1); // Initialize inverse to zero, twice the width of `prime` to allow multiplication
  BQInt sqr(lhsInt);              // Will be repeatedly squared
  BQInt position(2 * maxSize, 1); // Bit at `idx` will be 1, other bits will be 0
  sqr = sqr.zext(2 * maxSize);
  rhsInt = rhsInt.zext(2 * maxSize);
  BQInt exp = rhsInt - 2;
  for (size_t idx = 0; idx < maxSize; idx++) {
    if (exp.intersects(position)) {
      // multiply in the current power of n (i.e., n^(2^idx))
      inv = (inv * sqr).urem(rhsInt);
    }
    position <<= 1;                 // increment the bit position to test in `exp`
    sqr = (sqr * sqr).urem(rhsInt); // square `sqr` to increment to `n^(2^(idx+1))`
  }
  inv = inv.trunc(maxSize); // We don't need the extra space used as multiply buffer
  return fromBQInt(inv, coeffs);
}

size_t getCarryOffset(const Type& type) {
  size_t coeffMagnitude = std::max(type.maxPos, type.maxNeg);
  return (coeffMagnitude + 3*kBitsPerCoeff) / kBitsPerCoeff;
}

size_t getCarryBytes(const Type& type) {
  size_t maxCarry = getCarryOffset(type) * 2;
  if (maxCarry < 256) { return 1; }
  maxCarry /= 256;
  if (maxCarry < 256) { return 2; }
  return 4;
}

} // namespace


EvalOutput eval(const Program &inFunc, std::vector<BQInt> &witnessValues) {
  EvalOutput ret;

  // Store the output of each op, as we compute it
  std::vector<BytePoly> polys;
  polys.resize(inFunc.ops.size());

  for (size_t opIndex = 0; opIndex < inFunc.ops.size(); ++opIndex) {
    const Op &op = inFunc.ops[opIndex];
    switch (op.code) {
      case Op::Def: {
        const Type &type = inFunc.types[op.type];
        const Input &wire = inFunc.inputs[op.operandA];
        BQInt val = witnessValues[wire.label];
        uint32_t coeffs = type.coeffs;
        auto poly = fromBQInt(val, coeffs);
        polys[opIndex] = poly;
        // TODO
        // if (wire.isPublic) {
        //   ret.publicWitness.push_back(poly);
        // } else {
        //   ret.privateWitness.push_back(poly);
        // }
      } break;
      case Op::Con: {
        const Type &type = inFunc.types[op.type];
        uint32_t coeffs = type.coeffs;
        size_t offset = op.operandA;
        size_t words = op.operandB;
        BQInt value(64*words, words, inFunc.constants[offset]);
        auto poly = fromBQInt(value, coeffs);
        polys[opIndex] = poly;
        ret.constantWitness.push_back(poly);
      } break;
      case Op::Add: {
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        polys[opIndex] = add(lhs, rhs);
      } break;
      case Op::Sub: {
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        polys[opIndex] = sub(lhs, rhs);
      } break;
      case Op::Mul: {
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        polys[opIndex] = mul(lhs, rhs);
      } break;
      case Op::Rem: {
        const Type &type = inFunc.types[op.type];
        uint32_t coeffs = type.coeffs;
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        auto poly = nondetRem(lhs, rhs, coeffs);
        polys[opIndex] = poly;
        ret.privateWitness.push_back(poly);
      } break;
      case Op::Quo: {
        const Type &type = inFunc.types[op.type];
        uint32_t coeffs = type.coeffs;
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        auto poly = nondetQuot(lhs, rhs, coeffs);
        polys[opIndex] = poly;
        ret.privateWitness.push_back(poly);
      } break;
      case Op::Inv: {
        const Type &type = inFunc.types[op.type];
        uint32_t coeffs = type.coeffs;
        auto lhs = polys[op.operandA];
        auto rhs = polys[op.operandB];
        auto poly = nondetInvMod(lhs, rhs, coeffs);
        polys[opIndex] = poly;
        ret.privateWitness.push_back(poly);
      } break;
      case Op::Eqz: {
        auto poly = polys[op.operandA];
        if (toBQInt(poly) != 0) {
          throw std::runtime_error("NONZERO");
        }
        const Type &type = inFunc.types[op.type];
        uint32_t coeffs = type.coeffs;
        int32_t carryOffset = getCarryOffset(type);
        size_t carryBytes = getCarryBytes(type);
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
            throw std::runtime_error("INVALID CARRY");
          }
        }
        // Store the results
        for (size_t i = 0; i < carryPolys.size(); i++) {
          ret.privateWitness.push_back(carryPolys[i]);
        }
      } break;
      default: {
        throw std::runtime_error("Unknown op in eval");
      }
    }
  }

  Digest publicDigest = computeDigest(ret.publicWitness, 1);
  Digest privateDigest = computeDigest(ret.privateWitness, 3);
  Digest folded = poseidon2HashPair(publicDigest, privateDigest);

  // Now, compute the value of Z
  Poseidon2Rng rng;
  rng.mix(folded);
  for (size_t i = 0; i < 4; i++) {
    ret.z[i] = rng.generateFp();
  }

  return ret;
}

} // namespace zirgen::BigInt::Bytecode
