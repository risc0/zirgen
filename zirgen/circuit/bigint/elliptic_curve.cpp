// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/bigint/elliptic_curve.h"

namespace zirgen::BigInt {

void WeierstrassCurve::validate_contains(OpBuilder builder, Location loc, const AffinePt& pt) const {
  auto prime = prime_as_bigint(builder, loc);
  Value y_sqr = builder.create<BigInt::MulOp>(loc, pt.y(), pt.y());

  Value x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), pt.x());
  x_cube = builder.create<BigInt::ReduceOp>(loc, x_cube, prime);
  x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), x_cube);

  Value ax = builder.create<BigInt::MulOp>(loc, a_as_bigint(builder, loc), pt.x());

  Value weierstrass_rhs = builder.create<BigInt::AddOp>(loc, x_cube, ax);
  weierstrass_rhs = builder.create<BigInt::AddOp>(loc, weierstrass_rhs, b_as_bigint(builder, loc));

  Value diff = builder.create<BigInt::SubOp>(loc, weierstrass_rhs, y_sqr);
  diff = builder.create<BigInt::AddOp>(loc, diff, builder.create<BigInt::MulOp>(loc, prime, prime));  // Ensure `diff` nonnegative
  diff = builder.create<BigInt::ReduceOp>(loc, diff, prime);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void AffinePt::validate_equal(OpBuilder builder, Location loc, const AffinePt& other) const {
  Value x_diff = builder.create<BigInt::SubOp>(loc, x(), other.x());
  builder.create<BigInt::EqualZeroOp>(loc, x_diff);
  Value y_diff = builder.create<BigInt::SubOp>(loc, y(), other.y());
  builder.create<BigInt::EqualZeroOp>(loc, y_diff);

  // Curve information is only for constructing appropriate EC operations and doesn't appear on-circuit except as operation parameters
  // (and so doesn't need to be verified on circuit here), but you're doing something wrong if you expect two points on different curves to be equal.
  assert(on_same_curve_as(other));
}

void AffinePt::validate_on_curve(OpBuilder builder, Location loc) const {
  _curve->validate_contains(builder, loc, *this);
}

bool AffinePt::on_same_curve_as(const AffinePt& other) const {
  // Curves are only treated as equal if they are equal as pointers
  // This is reasonable because don't really want multiple copies of curves floating around
  // (and if we do have multiple copies of the same curve, maybe that represents a semantic difference and they shouldn't be treated as the same thing anyway)
  return _curve == other._curve;
}

AffinePt add(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs) {
  // This assumes `pt` is actually on the curve
  // This assumption isn't checked here, so other code must ensure it's met
  // Also, this can fail for: A + A (doesn't use doubling algorithm) and A + (-A) (can't write 0)
  // Trying to calculate either of those cases will result in an EQZ failure
  // Formulas (all mod `prime`):
  //   lambda = (yQ - yP) / (xQ - xP)
  //       nu = yP - lambda * xP
  //       xR = lambda^2 - xP - xQ
  //       yR = -(lambda * xR + nu)

  assert(lhs.on_same_curve_as(rhs));
  auto prime = lhs.curve()->prime_as_bigint(builder, loc);

  // Construct the constant 1
  mlir::Type oneType = builder.getIntegerType(8);
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  Value y_diff = builder.create<BigInt::SubOp>(loc, rhs.y(), lhs.y());
  y_diff = builder.create<BigInt::AddOp>(loc, y_diff, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  Value x_diff = builder.create<BigInt::SubOp>(loc, rhs.x(), lhs.x());
  x_diff = builder.create<BigInt::AddOp>(loc, x_diff, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity


  Value x_diff_inv = builder.create<BigInt::NondetInvModOp>(loc, x_diff, prime);
  // Enforce that xDiffInv is the inverse of x_diff
  Value x_diff_inv_check = builder.create<BigInt::MulOp>(loc, x_diff, x_diff_inv);
  Value x_diff_inv_check_quot = builder.create<BigInt::NondetQuotOp>(loc, x_diff_inv_check, prime);
  x_diff_inv_check_quot = builder.create<BigInt::MulOp>(loc, x_diff_inv_check_quot, prime);
  x_diff_inv_check = builder.create<BigInt::SubOp>(loc, x_diff_inv_check, x_diff_inv_check_quot);
  x_diff_inv_check = builder.create<BigInt::SubOp>(loc, x_diff_inv_check, one);
  builder.create<BigInt::EqualZeroOp>(loc, x_diff_inv_check);

  Value lambda = builder.create<BigInt::MulOp>(loc, y_diff, x_diff_inv);
  lambda = builder.create<BigInt::NondetRemOp>(loc, lambda, prime);
  // Verify `lambda` is `y_diff / x_diff` by verifying that `lambda * x_diff == y_diff + k * prime`
  Value lambda_check = builder.create<BigInt::MulOp>(loc, lambda, x_diff);
  lambda_check = builder.create<BigInt::SubOp>(loc, lambda_check, y_diff);
  lambda_check = builder.create<BigInt::AddOp>(loc, lambda_check, prime);
  lambda_check = builder.create<BigInt::AddOp>(loc, lambda_check, prime);
  Value k_lambda = builder.create<BigInt::NondetQuotOp>(loc, lambda_check, prime);
  lambda_check = builder.create<BigInt::SubOp>(loc, lambda_check, builder.create<BigInt::MulOp>(loc, k_lambda, prime));
  builder.create<BigInt::EqualZeroOp>(loc, lambda_check);

  Value nu = builder.create<BigInt::MulOp>(loc, lambda, lhs.x());
  nu = builder.create<BigInt::SubOp>(loc, lhs.y(), nu);
  nu = builder.create<BigInt::AddOp>(loc, nu, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity

  Value lambda_sqr = builder.create<BigInt::MulOp>(loc, lambda, lambda);
  Value xR = builder.create<BigInt::SubOp>(loc, lambda_sqr, lhs.x());
  xR = builder.create<BigInt::SubOp>(loc, xR, rhs.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  Value xR_unreduced = xR;
  Value k_x = builder.create<BigInt::NondetQuotOp>(loc, xR, prime);
  xR = builder.create<BigInt::NondetRemOp>(loc, xR, prime);

  Value yR = builder.create<BigInt::MulOp>(loc, lambda, xR);
  yR = builder.create<BigInt::AddOp>(loc, yR, nu);
  yR = builder.create<BigInt::SubOp>(loc, prime, yR);  // i.e., negate (mod prime) 
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);
  Value prime_sqr = builder.create<BigInt::MulOp>(loc, prime, prime);
  yR = builder.create<BigInt::AddOp>(loc, yR, prime_sqr);  // The prime^2 term is for the original lambda * xR
  Value k_y = builder.create<BigInt::NondetQuotOp>(loc, yR, prime);
  yR = builder.create<BigInt::NondetRemOp>(loc, yR, prime);

  // Verify xR
  Value x_check = builder.create<BigInt::SubOp>(loc, lambda_sqr, lhs.x());
  x_check = builder.create<BigInt::SubOp>(loc, x_check, rhs.x());
  x_check = builder.create<BigInt::AddOp>(loc, x_check, prime);
  x_check = builder.create<BigInt::AddOp>(loc, x_check, prime);
  Value kx_prime = builder.create<BigInt::MulOp>(loc, k_x, prime);
  x_check = builder.create<BigInt::SubOp>(loc, x_check, kx_prime);
  x_check = builder.create<BigInt::SubOp>(loc, x_check, xR);
  builder.create<BigInt::EqualZeroOp>(loc, x_check);

  // Verify yR
  Value y_check = builder.create<BigInt::MulOp>(loc, k_y, prime);
  y_check = builder.create<BigInt::AddOp>(loc, y_check, yR);
  Value y_check_other = builder.create<BigInt::SubOp>(loc, lhs.x(), xR);
  y_check_other = builder.create<BigInt::MulOp>(loc, lambda, y_check_other);
  y_check_other = builder.create<BigInt::SubOp>(loc, y_check_other, lhs.y());
  y_check_other = builder.create<BigInt::AddOp>(loc, y_check_other, prime);
  y_check_other = builder.create<BigInt::AddOp>(loc, y_check_other, prime);
  y_check_other = builder.create<BigInt::AddOp>(loc, y_check_other, prime_sqr);
  y_check = builder.create<BigInt::SubOp>(loc, y_check, y_check_other);
  builder.create<BigInt::EqualZeroOp>(loc, y_check);

  return AffinePt(xR, yR, lhs.curve());
}

AffinePt mul(OpBuilder builder, Location loc, Value scalar, const AffinePt& pt) {
  // This assumes `pt` is actually on the curve
  // This assumption isn't checked here, so other code must ensure it's met
  // This algorithm doesn't work if `scalar` is a multiple of `pt`'s order
  // This doesn't need a special check, as it always computes a P + -P, causing an EQZ failure

  // Construct constants
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);
  // mlir::Type twoType = builder.getIntegerType(2);  // a `2` is bitwidth 2
  mlir::Type twoType = builder.getIntegerType(3);  // a `2` is bitwidth 2
  auto twoAttr = builder.getIntegerAttr(twoType, 2);  // value 2
  auto two = builder.create<BigInt::ConstOp>(loc, twoAttr);

  // We can't represent the identity in affine coordinates.
  // Therefore, instead of computing scale * P, compute P + rounded_down_to_even(scale) * P - if_scale_even_else_0(P)
  // This can fail (notably at scale = 1 or -1) but is cryptographically unlikely and is only a completeness (not soundness) limitation
  auto result = pt;

  // The repeatedly doubled point
  auto doubled_pt = pt;

  Value subtract_pt;
  Value dont_subtract_pt;

  for (size_t it = 0; it < llvm::cast<BigIntType>(scalar.getType()).getMaxBits(); it++) {
    // Compute the remainder of scale mod 2
    // We need exactly 0 or 1, not something congruent to them mod 2
    // Therefore, directly use the nondets, and check not just that the q * d + r = n but also that r * (r - 1) == 0
    auto rem = builder.create<BigInt::NondetRemOp>(loc, scalar, two);
    auto quot = builder.create<BigInt::NondetQuotOp>(loc, scalar, two);
    auto quot_prod = builder.create<BigInt::MulOp>(loc, two, quot);
    auto resum = builder.create<BigInt::AddOp>(loc, quot_prod, rem);
    auto check = builder.create<BigInt::SubOp>(loc, resum, scalar);
    builder.create<BigInt::EqualZeroOp>(loc, check);
    // Also need 1 - (scale % 2)
    auto one_minus_rem = builder.create<BigInt::SubOp>(loc, one, rem);
    // `check_bit` is nonzero iff `rem` is neither 0 nor 1 -- which is not allowed
    auto check_bit = builder.create<BigInt::MulOp>(loc, rem, one_minus_rem);
    builder.create<BigInt::EqualZeroOp>(loc, check_bit);

    // A special case for the first iteration so we don't have to start from 0:
    // What we will do is start at 1 * pt, and on the first iteration, instead of adding pt if
    // scalar is odd, we store that we (eventually) need to subtract off pt if scalar is even
    // Then after the main multiply algorithm is done, we do that subtraction (if needed)
    if (it == 0) {
      subtract_pt = one_minus_rem;
      dont_subtract_pt = rem;
    } else {
      // When the bit is one, add the current doubling point; otherwise retain the current point
      // Compute "If P then =A, else =B" via the formula
      //   result = P * A + (1 - P) * B
      // TODO: I'm concerned about the uncomputability of the untaken branch when scale = order - 2^(bitwidth_order - 1)
      auto sum = add(builder, loc, result, doubled_pt);
      auto xIfAdd = builder.create<BigInt::MulOp>(loc, sum.x(), rem);
      auto yIfAdd = builder.create<BigInt::MulOp>(loc, sum.y(), rem);
      auto xIfNotAdd = builder.create<BigInt::MulOp>(loc, result.x(), one_minus_rem);
      auto yIfNotAdd = builder.create<BigInt::MulOp>(loc, result.y(), one_minus_rem);
      auto xMerged = builder.create<BigInt::AddOp>(loc, xIfAdd, xIfNotAdd);
      auto yMerged = builder.create<BigInt::AddOp>(loc, yIfAdd, yIfNotAdd);
      // TODO: The reduces keep the coeff size small enough, but aren't otherwise needed for correctness; could maybe eek out a bit of perf with lower-level nondets
      auto newX = builder.create<BigInt::ReduceOp>(loc, xMerged, result.curve()->prime_as_bigint(builder, loc));
      auto newY = builder.create<BigInt::ReduceOp>(loc, yMerged, result.curve()->prime_as_bigint(builder, loc));

      result = AffinePt(newX, newY, result.curve());
    }

    // Double the doubling point
    doubled_pt = doub(builder, loc, doubled_pt);
    // The lowest order bit has been used, so halve the scale factor and iterate
    scalar = quot;
  }
  // Now subtract off the original point if needed
  auto subtracted = sub(builder, loc, result, pt);
  auto xIfSub = builder.create<BigInt::MulOp>(loc, subtracted.x(), subtract_pt);
  auto yIfSub = builder.create<BigInt::MulOp>(loc, subtracted.y(), subtract_pt);
  auto xIfNotSub = builder.create<BigInt::MulOp>(loc, result.x(), dont_subtract_pt);
  auto yIfNotSub = builder.create<BigInt::MulOp>(loc, result.y(), dont_subtract_pt);
  Value xFinal = builder.create<BigInt::AddOp>(loc, xIfSub, xIfNotSub);
  xFinal = builder.create<BigInt::ReduceOp>(loc, xFinal, result.curve()->prime_as_bigint(builder, loc));
  Value yFinal = builder.create<BigInt::AddOp>(loc, yIfSub, yIfNotSub);
  yFinal = builder.create<BigInt::ReduceOp>(loc, yFinal, result.curve()->prime_as_bigint(builder, loc));

  return AffinePt(xFinal, yFinal, result.curve());
}

AffinePt neg(OpBuilder builder, Location loc, const AffinePt& pt){
  Value yR = builder.create<BigInt::SubOp>(loc, pt.curve()->prime_as_bigint(builder, loc), pt.y());
  return AffinePt(pt.x(), yR, pt.curve());
}

AffinePt doub(OpBuilder builder, Location loc, const AffinePt& pt){
  // This assumes `pt` is actually on the curve and that `pt` is not order 2
  // These assumptions aren't checked here, so other code must ensure they're met
  // If you need to check for the latter case, verify that `y_in` is not 0 (mod `prime`)

  // Formulas (all mod `prime`):
  //   lambda = (3 * x_in^2 + a_coeff) / (2 * y_in)
  //       nu = y_in - lambda * x_in
  //    x_out = lambda^2 - 2 * x_in
  //    y_out = -(lambda * x_out + nu)
  //
  // What we check is that there exist integers k_* such that:
  //   k_lambda * prime + 2 * y_in * lambda = 2 * prime^2 + 3 * x_in^2 + a
  //                    k_x * prime + x_out = 2 * prime + lambda^2 - 2 * x_in
  //                    k_y * prime + y_out = prime^2 + prime - lambda * x_out - y_in + lambda * x_in

  auto prime = pt.curve()->prime_as_bigint(builder, loc);

  Value x_sqr = builder.create<BigInt::MulOp>(loc, pt.x(), pt.x());
  Value lambda_num = builder.create<BigInt::AddOp>(loc, x_sqr, x_sqr);
  lambda_num = builder.create<BigInt::AddOp>(loc, lambda_num, x_sqr);
  lambda_num = builder.create<BigInt::AddOp>(loc, lambda_num, pt.curve()->a_as_bigint(builder, loc));
  Value prime_sqr = builder.create<BigInt::MulOp>(loc, prime, prime);  // Adding a prime^2 to enforce positivity
  Value lambda_check_rhs = builder.create<BigInt::AddOp>(loc, prime_sqr, prime_sqr);
  lambda_check_rhs = builder.create<BigInt::AddOp>(loc, lambda_check_rhs, lambda_num);

  Value two_y = builder.create<BigInt::AddOp>(loc, pt.y(), pt.y());

  Value two_y_inv = builder.create<BigInt::NondetInvModOp>(loc, two_y, prime);

  // Normalize to not overflow coefficient size
  // TODO: Is there a better way? This version adds ~25k cycles to secp256k1 EC Mul
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);
  Value lambda_num_normal = builder.create<BigInt::NondetQuotOp>(loc, lambda_num, one);
  builder.create<BigInt::EqualZeroOp>(loc, builder.create<BigInt::SubOp>(loc, lambda_num_normal, lambda_num));

  Value lambda = builder.create<BigInt::MulOp>(loc, lambda_num_normal, two_y_inv);
  lambda = builder.create<BigInt::NondetRemOp>(loc, lambda, prime);

  Value two_y_lambda = builder.create<BigInt::MulOp>(loc, pt.y(), lambda);
  two_y_lambda = builder.create<BigInt::AddOp>(loc, two_y_lambda, two_y_lambda);
  Value lambda_check_diff = builder.create<BigInt::SubOp>(loc, lambda_check_rhs, two_y_lambda);
  Value k_lambda = builder.create<BigInt::NondetQuotOp>(loc, lambda_check_diff, prime);

  // Now enforce `k_lambda * prime + 2 * y_in * lambda = 2 * prime^2 + 3 * x_in^2 + a`
  // (ensuring nondets `k_lambda` and `lambda` are valid)
  Value lambda_check = builder.create<BigInt::MulOp>(loc, k_lambda, prime);
  lambda_check = builder.create<BigInt::AddOp>(loc, lambda_check, two_y_lambda);
  lambda_check = builder.create<BigInt::SubOp>(loc, lambda_check, lambda_check_rhs);
  builder.create<BigInt::EqualZeroOp>(loc, lambda_check);

  // Compute x_out and enforce `k_x * prime + x_out = 2 * prime + lambda^2 - 2 * x_in`
  Value x_numerator = builder.create<BigInt::MulOp>(loc, lambda, lambda);
  x_numerator = builder.create<BigInt::AddOp>(loc, x_numerator, prime);
  x_numerator = builder.create<BigInt::AddOp>(loc, x_numerator, prime);
  x_numerator = builder.create<BigInt::SubOp>(loc, x_numerator, pt.x());
  x_numerator = builder.create<BigInt::SubOp>(loc, x_numerator, pt.x());
  // TODO: Perhaps there's a prebuilt op for this?
  Value k_x = builder.create<BigInt::NondetQuotOp>(loc, x_numerator, prime);
  Value x_out = builder.create<BigInt::NondetRemOp>(loc, x_numerator, prime);
  Value x_check = builder.create<BigInt::MulOp>(loc, k_x, prime);
  x_check = builder.create<BigInt::AddOp>(loc, x_check, x_out);
  x_check = builder.create<BigInt::SubOp>(loc, x_check, x_numerator);
  builder.create<BigInt::EqualZeroOp>(loc, x_check);

  // Compute y_out and enforce `k_y * prime + y_out = prime^2 + prime - lambda * x_out - y_in + lambda * x_in`
  Value y_numerator = builder.create<BigInt::MulOp>(loc, lambda, x_out);
  y_numerator = builder.create<BigInt::SubOp>(loc, prime_sqr, y_numerator);
  y_numerator = builder.create<BigInt::AddOp>(loc, y_numerator, prime);
  y_numerator = builder.create<BigInt::SubOp>(loc, y_numerator, pt.y());
  Value lambda_x_in = builder.create<BigInt::MulOp>(loc, lambda, pt.x());
  y_numerator = builder.create<BigInt::AddOp>(loc, y_numerator, lambda_x_in);
  // TODO: Perhaps there's a prebuilt op for this?
  Value k_y = builder.create<BigInt::NondetQuotOp>(loc, y_numerator, prime);
  Value y_out = builder.create<BigInt::NondetRemOp>(loc, y_numerator, prime);
  Value y_check = builder.create<BigInt::MulOp>(loc, k_y, prime);
  y_check = builder.create<BigInt::AddOp>(loc, y_check, y_out);
  y_check = builder.create<BigInt::SubOp>(loc, y_check, y_numerator);
  builder.create<BigInt::EqualZeroOp>(loc, y_check);

  return AffinePt(x_out, y_out, pt.curve());
}

AffinePt sub(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs) {
  // This assumes `pt` is actually on the curve
  // This assumption isn't checked here, so other code must ensure it's met
  // Also, this can fail for: A - A (can't write 0) and A - (-A) (doesn't use doubling algorithm)
  // Trying to calculate either of those cases will result in an EQZ failure
  auto neg_rhs = neg(builder, loc, rhs);
  return add(builder, loc, lhs, neg_rhs);
}

void ECDSA_verify(OpBuilder builder, Location loc, const AffinePt& base_pt, const AffinePt& pub_key, Value hashed_msg, Value r, Value s, Value order) {
  pub_key.on_same_curve_as(base_pt);

  // Note: we don't need to validate `base_pt` on the curve as it's a publicly pre-committed parameter whose presence on the curve we can verify ahead of time
  // (much like we can verify the order of the curve ahead of time)
  pub_key.validate_on_curve(builder, loc);

  // TODO: Need anything to check the order of various points?

  // Mathematically, we need to ensure also that `pub_key` is not the identity.
  // But it's not possible to express the identity in affine coords, so this comes for free just by validating the point is on the curve

  // Construct constants
  // TODO: Check that this doesn't contruct multiple `one`s when called in various places
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  // Compute s_inv
  Value s_inv = builder.create<BigInt::NondetInvModOp>(loc, s, order);  // TODO: Do we need to handle `order` specially?
  Value s_inv_check = builder.create<BigInt::MulOp>(loc, s, s_inv);
  s_inv_check = builder.create<BigInt::ReduceOp>(loc, s_inv_check, order);  // TODO: Do we need to handle `order` specially?
  s_inv_check = builder.create<BigInt::SubOp>(loc, s_inv_check, one);
  builder.create<BigInt::EqualZeroOp>(loc, s_inv_check);

  // Compute u multipliers
  Value u1 = builder.create<BigInt::MulOp>(loc, hashed_msg, s_inv);
  u1 = builder.create<BigInt::ReduceOp>(loc, u1, order);  // TODO: Do we need to handle `order` specially?
  Value u2 = builder.create<BigInt::MulOp>(loc, r, s_inv);
  u2 = builder.create<BigInt::ReduceOp>(loc, u2, order);  // TODO: Do we need to handle `order` specially?

  // Calculate test point
  AffinePt u1G = mul(builder, loc, u1, base_pt);
  AffinePt u2Q = mul(builder, loc, u2, pub_key);
  AffinePt test_pt = add(builder, loc, u1G, u2Q);
  // n.b. no need to test for == identity, as `add` fails when adding a point to its negative

  // Validate signature
  Value sig_test = builder.create<BigInt::SubOp>(loc, r, test_pt.x());
  builder.create<BigInt::EqualZeroOp>(loc, sig_test);

  // TODO: Delete me (useless code that's fixing a temporary unused code error)
  Value TODO_y_test = builder.create<BigInt::SubOp>(loc, test_pt.y(), test_pt.y());
  builder.create<BigInt::EqualZeroOp>(loc, TODO_y_test);
  // End TODO
}

void makeECDSAVerify(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,  // TODO: These `bits` parameters could maybe be inferred from the prime (and definitely from the prime + the order)
    APInt prime,
    APInt curve_a,
    APInt curve_b,
    APInt order
) {
  // TODO: What can we move from an input to a parameter?
  // Since curve order differs from prime by at most 2 * sqrt(prime), we only need 1 more bit than `prime`
  auto order_bits = bits + 1;
  auto base_pt_X = builder.create<BigInt::DefOp>(loc, bits, 0, true);  // TODO: Or get from a parameter to this call?
  auto base_pt_Y = builder.create<BigInt::DefOp>(loc, bits, 1, true);  // TODO: Or get from a parameter to this call?
  auto pub_key_X = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto pub_key_Y = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto msg_hash = builder.create<BigInt::DefOp>(loc, order_bits, 4, true);
  auto r = builder.create<BigInt::DefOp>(loc, order_bits, 5, true);
  auto s = builder.create<BigInt::DefOp>(loc, order_bits, 6, true);

  // Add order as a constant
  mlir::Type order_type = builder.getIntegerType(order.getBitWidth());  // TODO: I haven't thought through signedness
  auto order_attr = builder.getIntegerAttr(order_type, order);
  auto order_const = builder.create<BigInt::ConstOp>(loc, order_attr);

  // TODO: Think through if we need to validate any of this (e.g. the orders, points being on curves)
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt base_pt(base_pt_X, base_pt_Y, curve);
  AffinePt pub_key(pub_key_X, pub_key_Y, curve);

  ECDSA_verify(builder, loc, base_pt, pub_key, msg_hash, r, s, order_const);
}

// Test functions

void makeECAffineAddTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 5, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  AffinePt expected(xR, yR, curve);
  auto result = add(builder, loc, lhs, rhs);
  result.validate_equal(builder, loc, expected);
}

void makeECAffineDoubleTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  AffinePt expected(xR, yR, curve);
  auto result = doub(builder, loc, inp);
  result.validate_equal(builder, loc, expected);
}

void makeECAffineMultiplyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  // This test is only valid for curves whose order is of bitwidth no more than the prime's bitwidth
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto scale = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 4, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  AffinePt expected(xR, yR, curve);
  auto result = mul(builder, loc, scale, inp);
  result.validate_equal(builder, loc, expected);
}

void makeECAffineNegateTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto order_bits = bits;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  AffinePt expected(xR, yR, curve);
  auto result = neg(builder, loc, inp);
  result.validate_equal(builder, loc, expected);
}

void makeECAffineSubtractTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 5, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  AffinePt expected(xR, yR, curve);
  auto result = sub(builder, loc, lhs, rhs);
  result.validate_equal(builder, loc, expected);
}

void makeECAffineValidatePointsEqualTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  lhs.validate_equal(builder, loc, rhs);
}

// TODO: Drop Affine from all these names
void makeECAffineValidatePointOnCurveTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt pt(xP, yP, curve);
  pt.validate_on_curve(builder, loc);
}

// TODO
// void makeECAffineValidatePointOrderTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

// The "Freely" test functions run the op without checking the output
void makeECAddFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  auto result = add(builder, loc, lhs, rhs);
  // We check result == result so it doesn't get DCE'd
  result.validate_equal(builder, loc, result);
}

void makeECDoubleFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  auto result = doub(builder, loc, inp);
  // We check result == result so it doesn't get DCE'd
  result.validate_equal(builder, loc, result);
}

void makeECMultiplyFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  // This test is only valid for curves whose order is of bitwidth no more than the prime's bitwidth
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto scale = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  auto result = mul(builder, loc, scale, inp);
  // We check result == result so it doesn't get DCE'd
  result.validate_equal(builder, loc, result);
}

void makeECNegateFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto order_bits = bits;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  auto result = neg(builder, loc, inp);
  // We check result == result so it doesn't get DCE'd
  result.validate_equal(builder, loc, result);
}

void makeECSubtractFreelyTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  auto result = sub(builder, loc, lhs, rhs);
  // We check result == result so it doesn't get DCE'd
  result.validate_equal(builder, loc, result);
}

// Perf Test function
void makeRepeatedECAffineAddTest(mlir::OpBuilder builder,
                                 mlir::Location loc,
                                 size_t bits,
                                 size_t reps,
                                 APInt prime,
                                 APInt curve_a,
                                 APInt curve_b) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 5, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt lhs(xP, yP, curve);
  AffinePt rhs(xQ, yQ, curve);
  AffinePt expected(xR, yR, curve);
  auto result = add(builder, loc, lhs, rhs);
  // iterate from 1 because the first repetition was already done
  for (size_t rp = 1; rp < reps; rp++) {
    result = add(builder, loc, result, rhs);
  }
  result.validate_equal(builder, loc, expected);
}

// Perf Test function
void makeRepeatedECAffineDoubleTest(mlir::OpBuilder builder,
                                    mlir::Location loc,
                                    size_t bits,
                                    size_t reps,
                                    APInt prime,
                                    APInt curve_a,
                                    APInt curve_b) {
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 3, true);

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  AffinePt expected(xR, yR, curve);
  auto result = doub(builder, loc, inp);
  // iterate from 1 because the first repetition was already done
  for (size_t rp = 1; rp < reps; rp++) {
    result = doub(builder, loc, inp);
  }
  result.validate_equal(builder, loc, expected);
}

} // namespace zirgen::BigInt