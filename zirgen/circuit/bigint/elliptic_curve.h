// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

class AffinePt;

class WeierstrassCurve {
  // An elliptic curve in short Weierstrass form
  // Formula:
  //  y^2 = x^3 + a*x + b  (mod p)
public:
  WeierstrassCurve(APInt prime, APInt a_coeff, APInt b_coeff) : _prime(prime), _a_coeff(a_coeff), _b_coeff(b_coeff) {};
  const APInt& a() const { return _a_coeff; };
  const APInt& b() const { return _b_coeff; };
  const APInt& prime() const { return _prime; };
  Value a_as_bigint(OpBuilder builder, Location loc) const {
    mlir::Type type = builder.getIntegerType(_a_coeff.getBitWidth());  // TODO: I haven't thought through signedness
    auto attr = builder.getIntegerAttr(type, _a_coeff);
    return builder.create<BigInt::ConstOp>(loc, attr);
  };
  Value b_as_bigint(OpBuilder builder, Location loc) const {
    mlir::Type type = builder.getIntegerType(_b_coeff.getBitWidth());  // TODO: I haven't thought through signedness
    auto attr = builder.getIntegerAttr(type, _b_coeff);
    return builder.create<BigInt::ConstOp>(loc, attr);
  };;
  Value prime_as_bigint(OpBuilder builder, Location loc) const {
    mlir::Type type = builder.getIntegerType(_prime.getBitWidth());  // TODO: I haven't thought through signedness
    auto attr = builder.getIntegerAttr(type, _prime);
    return builder.create<BigInt::ConstOp>(loc, attr);
  };;
  void validate_contains(OpBuilder builder, Location loc, const AffinePt& pt) const;

private:
  APInt _prime;
  APInt _a_coeff;
  APInt _b_coeff;
};

class AffinePt {
  // A point expressed in affine coordinates
public:
  AffinePt(Value x_coord, Value y_coord, std::shared_ptr<WeierstrassCurve> curve) : _x(x_coord), _y(y_coord), _curve(curve) {};
  const Value& x() const { return _x; };
  const Value& y() const { return _y; };
  const std::shared_ptr<WeierstrassCurve>& curve() const { return _curve; };
  void validate_equal(OpBuilder builder, Location loc, const AffinePt& other) const;
  void validate_on_curve(OpBuilder builder, Location loc) const;
  bool on_same_curve_as(const AffinePt& other) const;

private:
  // TODO: Can we constrain the type better than just "Value"?
  // (i.e. to BigInts)
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

void ECDSA_verify(OpBuilder builder, Location loc, const AffinePt& base_pt, const AffinePt& pub_key, Value hashed_msg, Value r, Value s);

void makeECDSAVerify(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b,
    APInt order
);

// Test functions
// TODO: Choose a good suite of test functions
void makeECAffineAddTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
void makeECAffineDoubleTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
void makeECAffineMultiplyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
void makeECAffineNegateTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
void makeECAffineSubtractTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
// void makeECAffineValidatePointOrderTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeECAffineValidatePointsEqualTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);
void makeECAffineValidatePointOnCurveTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

// The "Freely" test functions run the op without checking the output
// These are mostly useful for testing expected failures e.g. P + -P should always fail
// TODO: Might not need this for Doub or Neg
void makeECAddFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

void makeECDoubleFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

void makeECMultiplyFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

void makeECNegateFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

void makeECSubtractFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
);

// Perf test functions
void makeRepeatedECAffineAddTest(mlir::OpBuilder builder,
                                 mlir::Location _loc,
                                 size_t bits,
                                 size_t reps,
                                 APInt prime,
                                 APInt curve_a,
                                 APInt curve_b);
void makeRepeatedECAffineDoubleTest(mlir::OpBuilder builder,
                                    mlir::Location _loc,
                                    size_t bits,
                                    size_t reps,
                                    APInt prime,
                                    APInt curve_a,
                                    APInt curve_b);

} // namespace zirgen::BigInt