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

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt::EC {

class AffinePt;

class WeierstrassCurve {
  // An elliptic curve in short Weierstrass form
  // Formula:
  //  y^2 = x^3 + a*x + b  (mod p)
public:
  WeierstrassCurve(Value prime, Value a_coeff, Value b_coeff)
      : _prime(prime), _a_coeff(a_coeff), _b_coeff(b_coeff){};
  const Value& a() const { return _a_coeff; };
  const Value& b() const { return _b_coeff; };
  const Value& prime() const { return _prime; };
  void validate_contains(OpBuilder builder, Location loc, const AffinePt& pt) const;

private:
  Value _prime;
  Value _a_coeff;
  Value _b_coeff;
};

class AffinePt {
  // A point on a Weierstrass curve expressed in affine coordinates
public:
  AffinePt(Value x_coord, Value y_coord, std::shared_ptr<WeierstrassCurve> curve)
      : _x(x_coord), _y(y_coord), _curve(curve){};
  const Value& x() const { return _x; };
  const Value& y() const { return _y; };
  const std::shared_ptr<WeierstrassCurve>& curve() const { return _curve; };
  void validate_equal(OpBuilder builder, Location loc, const AffinePt& other) const;
  void validate_on_curve(OpBuilder builder, Location loc) const;
  bool on_same_curve_as(const AffinePt& other) const;

private:
  Value _x;
  Value _y;
  // The elliptic curve this point lies on
  std::shared_ptr<WeierstrassCurve> _curve;
};

AffinePt add(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs);
AffinePt doub(OpBuilder builder, Location loc, const AffinePt& pt);
AffinePt mul(OpBuilder builder, Location loc, Value scalar, const AffinePt& pt);
AffinePt neg(OpBuilder builder, Location loc, const AffinePt& pt);
AffinePt sub(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs);

// Full Programs
void genECAdd(mlir::OpBuilder& builder, mlir::Location loc, size_t bitwidth);
void genECDouble(mlir::OpBuilder& builder, mlir::Location loc, size_t bitwidth);

// Test functions
void makeECAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECDoubleTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECMultiplyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECNegateTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECSubtractTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECValidatePointsEqualTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECValidatePointOnCurveTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

// The "Freely" test functions run the op without checking the output
// These are mostly useful for testing expected failures e.g. P + -P should always fail
void makeECAddFreelyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECDoubleFreelyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECMultiplyFreelyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECNegateFreelyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECSubtractFreelyTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

// Perf test functions
void makeRepeatedECAddTest(mlir::OpBuilder builder, mlir::Location _loc, size_t bits, size_t reps);
void makeRepeatedECDoubleTest(mlir::OpBuilder builder,
                              mlir::Location _loc,
                              size_t bits,
                              size_t reps);

} // namespace zirgen::BigInt::EC
