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

#pragma once

#include "llvm/ADT/ArrayRef.h"

namespace zirgen::Zll {

// Default field prime to baby bear
static constexpr uint64_t kFieldPrimeDefault = ((1 << 27) * 15 + 1);

struct Field {
  uint64_t prime = kFieldPrimeDefault;
  Field() = default;
  explicit Field(uint64_t p) : prime{p} {}
  uint64_t Add(uint64_t a, uint64_t b) const;
  uint64_t Sub(uint64_t a, uint64_t b) const;
  uint64_t Mul(uint64_t a, uint64_t b) const;
  uint64_t Inv(uint64_t a) const;
};

static constexpr uint64_t kFieldInvalid = 0xffffffff;

/// Represents an extension field of a particular degree over a particular prime
/// field. The elements are polynomials of degree less than the extension degree
/// with coefficients modulo the prime. Results of operations are modulo an
/// "irreducible polynomial" which looked up in a table.
///
/// This particular implementation is used for constant folding.
struct ExtensionField {
  using FieldResult = llvm::SmallVector<uint64_t, 4>;
  using FieldArg = llvm::ArrayRef<uint64_t>;

  Field subfield = Field();
  uint64_t degree = 1;

  ExtensionField() = default;
  ExtensionField(uint64_t p, uint64_t k) : subfield(p), degree(k) {}

  FieldResult Add(FieldArg a, FieldArg b) const;
  FieldResult Sub(FieldArg a, FieldArg b) const;
  FieldResult Mul(uint64_t a, FieldArg b) const;
  FieldResult Mul(FieldArg a, FieldArg b) const;
  FieldResult Inv(FieldArg a) const;
  FieldResult Neg(FieldArg a) const;
  FieldResult BitAnd(FieldArg a, FieldArg b) const;
  FieldResult Mod(FieldArg a, FieldArg b) const;

  FieldResult Zero() const { return FieldResult(degree); }

  FieldResult One() const {
    FieldResult a(degree);
    a[0] = 1;
    return a;
  }

private:
  /// Represents an irreducible polynomial in the extension field which is used
  /// for multiplication. Usually these polynomials look like a + bx + ... + zx^n = 0,
  /// but without loss of generality we take the restriction z = 1 and represent
  /// the polynomial as the RHS of x^degree = -a - bx + ... . Since the original
  /// polynomial is irreducible, scaling it by z^-1 produces another irreducible
  /// polynomial where the highest degree term has a coefficient of 1.
  FieldResult getIrreduciblePolynomial() const;

  // Multiplies a and b without making use of the irreducible polynomial. This
  // may result in a polynomial that is outside of the field.
  FieldResult NaiveMul(FieldArg a, FieldArg b) const;

  /// computes the quotient q of a and b, such that a = q * b + r and the degree
  /// of r is less than the degree of b. (c.f. polynomial long division). This
  /// may result in a polynomial that is outside of the field.
  FieldResult Div(FieldArg a, FieldArg b) const;
};

bool isInvalid(ExtensionField::FieldArg a);
bool isZero(ExtensionField::FieldArg a);

} // namespace zirgen::Zll
