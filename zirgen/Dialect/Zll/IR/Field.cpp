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

#include "zirgen/Dialect/Zll/IR/Field.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

namespace zirgen::Zll {

uint64_t Field::Add(uint64_t a, uint64_t b) const {
  uint64_t o = a + b;
  if (o < a || o >= prime) {
    o -= prime;
  }
  return o;
}

uint64_t Field::Sub(uint64_t a, uint64_t b) const {
  return Add(a, prime - b);
}

uint64_t Field::Mul(uint64_t a, uint64_t b) const {
  using u128 = unsigned __int128;
  return uint64_t((u128(a) * u128(b)) % u128(prime));
}

// Compute the field inverse be the extended Euclidean algorithm
// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
uint64_t Field::Inv(uint64_t a) const {
  uint64_t t0 = 0;
  uint64_t t1 = 1;
  uint64_t r0 = prime;
  uint64_t r1 = a;

  while (r1 != 0) {
    uint64_t quot = r0 / r1;
    uint64_t t2 = Sub(t0, Mul(quot, t1));
    t0 = t1;
    t1 = t2;
    uint64_t r2 = r0 - quot * r1;
    r0 = r1;
    r1 = r2;
  }
  return t0;
}

static uint64_t getDegree(ExtensionField::FieldArg a) {
  for (size_t i = a.size() - 1; i > 0; i--) {
    if (a[i] != 0) {
      return i;
    }
  }
  return 0;
}

ExtensionField::FieldResult ExtensionField::Add(ExtensionField::FieldArg a,
                                                ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(a.begin(), a.end());
  c.resize(degree);
  for (size_t i = 0; i < b.size(); i++) {
    c[i] = subfield.Add(c[i], b[i]);
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::Sub(ExtensionField::FieldArg a,
                                                ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(a.begin(), a.end());
  c.resize(degree);
  for (size_t i = 0; i < std::min(size_t(degree), b.size()); i++) {
    c[i] = subfield.Sub(c[i], b[i]);
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::Mul(uint64_t a, ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(degree);
  for (size_t i = 0; i < degree; i++) {
    c[i] = subfield.Mul(a, b[i]);
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::Mul(ExtensionField::FieldArg a,
                                                ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c = NaiveMul(a, b);

  // Reduce the degree using the irreducible polynomial
  ExtensionField::FieldResult irreducible = getIrreduciblePolynomial();
  for (size_t i = 2 * degree - 2; i >= degree; i--) {
    for (size_t j = 0; j < degree; j++) {
      c[i - degree] = subfield.Add(c[i - degree], subfield.Mul(c[i], irreducible[j]));
    }
    c[i] = 0;
  }
  c.truncate(degree);
  return c;
}

ExtensionField::FieldResult ExtensionField::Inv(ExtensionField::FieldArg a) const {
  if (getDegree(a) == 0) {
    return {subfield.Inv(a[0])};
  }

  // Work in a slightly larger extension field that can represent this field's
  // irreducible polynomial.
  ExtensionField f(subfield.prime, degree + 1);
  ExtensionField::FieldResult t0 = f.Zero();
  ExtensionField::FieldResult t1 = f.One();
  ExtensionField::FieldResult r0 = getIrreduciblePolynomial();
  r0.push_back(subfield.prime - 1);
  ExtensionField::FieldResult r1(degree + 1);
  for (size_t i = 0; i < degree; i++) {
    r1[i] = a[i];
  }

  while (getDegree(r1) > 0) {
    ExtensionField::FieldResult quot = Div(r0, r1);
    ExtensionField::FieldResult t2 = f.Sub(t0, f.NaiveMul(quot, t1));
    t0 = t1;
    t1 = t2;
    ExtensionField::FieldResult r2 = f.Sub(r0, f.NaiveMul(quot, r1));
    r0 = r1;
    r1 = r2;
  }
  return Mul(subfield.Inv(r1[0]), t1);
}

ExtensionField::FieldResult ExtensionField::Neg(ExtensionField::FieldArg a) const {
  ExtensionField::FieldResult b(degree);
  for (size_t i = 0; i < degree; i++) {
    b[i] = subfield.Sub(0, a[i]);
  }
  return b;
}

bool isInvalid(ExtensionField::FieldArg a) {
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] == kFieldInvalid) {
      return true;
    }
  }
  return false;
}

bool isZero(ExtensionField::FieldArg a) {
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != 0) {
      return false;
    }
  }
  return true;
}

ExtensionField::FieldResult ExtensionField::BitAnd(ExtensionField::FieldArg a,
                                                   ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(degree);
  for (size_t i = 0; i < degree; i++) {
    c[i] = a[i] & b[i];
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::Mod(ExtensionField::FieldArg a,
                                                ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(degree);
  for (size_t i = 0; i < degree; i++) {
    if (b[i] == 0) {
      throw std::runtime_error("Invalid mod by zero");
    }
    c[i] = a[i] % b[i];
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::getIrreduciblePolynomial() const {
  assert(subfield.prime == kFieldPrimeDefault && "unsupported field prime");
  switch (degree) {
  case 1:
    return {1};
  case 2:
    return {subfield.prime - 11, 0};
  case 4:
    return {subfield.prime - 11, 0, 0, 0};
  default:
    llvm_unreachable("unsupported field extension degree");
    return {};
  }
}

ExtensionField::FieldResult ExtensionField::NaiveMul(ExtensionField::FieldArg a,
                                                     ExtensionField::FieldArg b) const {
  ExtensionField::FieldResult c(2 * degree - 1);
  for (size_t i = 0; i < a.size(); i++) {
    for (size_t j = 0; j < b.size(); j++) {
      c[i + j] = subfield.Add(c[i + j], subfield.Mul(a[i], b[j]));
    }
  }
  return c;
}

ExtensionField::FieldResult ExtensionField::Div(ExtensionField::FieldArg a,
                                                ExtensionField::FieldArg b) const {
  const uint64_t aDegree = getDegree(a);
  const uint64_t bDegree = getDegree(b);

  // Work in an extension field that can fit this extension field's irreducible
  // polynomial. This is necessary because `a` will be the irreducible polynomial
  // in the first step of Euclid's agorithm for finding multiplicative inverses.
  ExtensionField extension(subfield.prime, degree + 1);
  ExtensionField::FieldResult b2(degree + 1), r(degree + 1), q(degree + 1);
  for (size_t i = 0; i <= aDegree; i++) {
    r[i] = a[i];
  }
  for (size_t i = 0; i <= bDegree; i++) {
    b2[i] = b[i];
  }

  uint64_t rDegree = aDegree;
  while (rDegree >= bDegree) {
    assert(q[rDegree - bDegree] == 0);
    q[rDegree - bDegree] = subfield.Mul(r[rDegree], subfield.Inv(b2[bDegree]));
    r = extension.Sub(a, extension.NaiveMul(q, b2));
    rDegree = getDegree(r);
  }
  return q;
}

} // namespace zirgen::Zll
