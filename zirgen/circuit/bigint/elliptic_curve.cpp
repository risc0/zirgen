// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/bigint/elliptic_curve.h"

namespace zirgen::BigInt {

void WeierstrassCurve::validate_contains(OpBuilder builder, Location loc, const AffinePt& pt) const {
  auto prime = prime_as_bigint(builder, loc);
  Value y_sqr = builder.create<BigInt::MulOp>(loc, pt.y(), pt.y());

  Value x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), pt.x());
  x_cube = builder.create<BigInt::ReduceOp>(loc, x_cube, prime);  // TODO: Reduce isn't required for correctness, perf better if skipped?  // TODO: But seems to make an overflow if dropped?
  x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), x_cube);

  Value ax = builder.create<BigInt::MulOp>(loc, a_as_bigint(builder, loc), pt.x());

  Value weierstrass_rhs = builder.create<BigInt::AddOp>(loc, x_cube, ax);
  weierstrass_rhs = builder.create<BigInt::AddOp>(loc, weierstrass_rhs, b_as_bigint(builder, loc));

  Value diff = builder.create<BigInt::SubOp>(loc, weierstrass_rhs, y_sqr);
  diff = builder.create<BigInt::AddOp>(loc, diff, builder.create<BigInt::MulOp>(loc, prime, prime));  // Ensure `diff` nonnegative
  diff = builder.create<BigInt::ReduceOp>(loc, diff, prime);  // TODO: Testing doing here instead of on its inputs
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void AffinePt::validate_equal(OpBuilder builder, Location loc, const AffinePt& other) const {
  // Curve information is only for constructing appropriate EC operations and doesn't appear on-circuit except as operation parameters
  // (and so doesn't need to be verified on circuit here), but you're doing something wrong if you expect two points on different curves to be equal.
  assert(on_same_curve_as(other));

  Value x_diff = builder.create<BigInt::SubOp>(loc, x(), other.x());
  builder.create<BigInt::EqualZeroOp>(loc, x_diff);
  Value y_diff = builder.create<BigInt::SubOp>(loc, y(), other.y());
  builder.create<BigInt::EqualZeroOp>(loc, y_diff);
}

void AffinePt::validate_on_curve(OpBuilder builder, Location loc) const {
  _curve->validate_contains(builder, loc, *this);
}

bool AffinePt::on_same_curve_as(const AffinePt& other) const {
  // Curves are only treated as equal if they are equal as pointers
  // Probably fine b/c we don't really want multiple copies of curves floating around
  // (and if we do have multiple copies of the same curve, maybe that represents a semantic difference and they shouldn't be treated as the same thing anyway)
  return _curve == other._curve;
}

AffinePt add(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs) {
  // Note: `add` can fail in two ways: A + A (doesn't use doubling algorithm) or A + (-A) (can't write 0)  [TODO: Document?]
  // Formulas (all mod `prime`):
  //   lambda = (yQ - yP) / (xQ - xP)
  //       nu = yP - lambda * xP
  //       xR = lambda^2 - xP - xQ
  //       yR = -(lambda * xR + nu)

  assert(lhs.on_same_curve_as(rhs));
  auto prime = lhs.curve()->prime_as_bigint(builder, loc);

  // TODO: How much to reduce? smaller bitwidth is nice, but so are fewer operations...
  // Construct the constant 1
  mlir::Type oneType = builder.getIntegerType(8);  // a `1` is bitwidth 1  // TODO
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



  // TODO: Doubling the prime seems to not work well

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
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  xR = builder.create<BigInt::SubOp>(loc, xR, rhs.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity
  Value k_x = builder.create<BigInt::NondetQuotOp>(loc, xR, prime);
  xR = builder.create<BigInt::NondetRemOp>(loc, xR, prime);

  Value yR = builder.create<BigInt::MulOp>(loc, lambda, xR);
  yR = builder.create<BigInt::AddOp>(loc, yR, nu);
  yR = builder.create<BigInt::SubOp>(loc, prime, yR);  // i.e., negate (mod prime) 
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);  // Quot/Rem needs nonnegative inputs, so enforce positivity  // TODO: better with using 3*prime for sub?
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);
  Value prime_sqr = builder.create<BigInt::MulOp>(loc, prime, prime);
  yR = builder.create<BigInt::AddOp>(loc, yR, prime_sqr);  // The prime^2 term is for the original lambda * xR
  Value k_y = builder.create<BigInt::NondetQuotOp>(loc, yR, prime);
  yR = builder.create<BigInt::NondetRemOp>(loc, yR, prime);

  // Verify xR
  // TODO: Can skip recomputing the things calculated pre-nondet above
  Value x_check = builder.create<BigInt::MulOp>(loc, k_x, prime);
  x_check = builder.create<BigInt::SubOp>(loc, lambda_sqr, x_check);
  x_check = builder.create<BigInt::SubOp>(loc, x_check, lhs.x());
  x_check = builder.create<BigInt::SubOp>(loc, x_check, rhs.x());
  x_check = builder.create<BigInt::AddOp>(loc, x_check, prime);
  x_check = builder.create<BigInt::AddOp>(loc, x_check, prime);
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

AffinePt mul(OpBuilder builder, Location loc, Value scalar, const AffinePt& pt, const AffinePt& arbitrary) {
  assert(arbitrary.on_same_curve_as(pt));
  // Construct constants
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);
  // mlir::Type twoType = builder.getIntegerType(2);  // a `2` is bitwidth 2
  mlir::Type twoType = builder.getIntegerType(3);  // a `2` is bitwidth 2
  auto twoAttr = builder.getIntegerAttr(twoType, 2);  // value 2
  auto two = builder.create<BigInt::ConstOp>(loc, twoAttr);

  // This algorithm doesn't work if `scalar` is a multiple of `pt`'s order
  // No separate testing is needed, as this will compute the `result` value (inclusive of `arbitrary`) to
  // be equal to `arbitrary` in this case, and then compute `sub` of `arbitrary` with itself, which fails
  // (via attempted division by zero) in `sub`.

  // We can't represent the identity in affine coordinates.
  // Therefore, instead of computing scale * P, compute Arb + scale * P - Arb
  // where Arb is some arbitrary point
  // This can fail if choosing an unlucky point Arb, but no soundness issues and can adjust Arb to get completeness
  auto result = arbitrary;

  // The repeatedly doubled point
  auto doubled_pt = pt;
  // TODO: If we call this multiple times with the same `arbitrary`, then `validate_on_curve` will be called multiple times for it
  // I think this is OK as CSE removes everything except the final eqz, and this doesn't get called _that_ often
  // TODO: Perhaps have CSE remove duplicate eqz's too?
  // TODO: Removing this slightly reduces the cycle count, so the CSE isn't perfect here. Possible location for minor perf gains
  // NOTE: We assume `pt` is already validated as on the curve (either manually already, or by construction) [TODO: Document better?]
  // `arbitrary` has not need to be constructed in any particular way, so we pretty much always need to validate it's on the curve
  // Hence, we do so here
  arbitrary.validate_on_curve(builder, loc);

  // TODO: Temporarily hacking to the an exponent large enough to cover the prime
  // (i.e., since we have a test on order 43, to 6)
  // What we should actually do is read this value off the prime
  // Note that until we do this, the small tests will fail due to the extra collision opportunities with Arbitrary
  llvm::outs() << "    EC mul with " + std::to_string(llvm::cast<BigIntType>(scalar.getType()).getMaxBits()) + " iterations\n";  // TODO: Temporary log
  for (size_t it = 0; it < llvm::cast<BigIntType>(scalar.getType()).getMaxBits(); it++) {  // TODO: Why is this slightly larger than the bitwidth?
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
    // TODO: I think these may not actually be needed ...
    // TODO: These seem necessary for bitwidth reasons and/or coeff size reasons, but probably shouldn't be needed for correctness; perhaps there's a workaround?
    auto newX = builder.create<BigInt::ReduceOp>(loc, xMerged, result.curve()->prime_as_bigint(builder, loc));
    auto newY = builder.create<BigInt::ReduceOp>(loc, yMerged, result.curve()->prime_as_bigint(builder, loc));

    result = AffinePt(newX, newY, result.curve());
    // Double the doubling point
    doubled_pt = doub(builder, loc, doubled_pt);
    // The lowest order bit has been used, so halve the scale factor and iterate
    scalar = quot;
  }

  // Subtract off (xArb, yArb) before returning
  // TODO: Reduce negYArb? Shouldn't be needed if provided a point in 1 < * < prime, and I *think* it's not _dangerous_ even if not
  return sub(builder, loc, result, arbitrary);
}

AffinePt neg(OpBuilder builder, Location loc, const AffinePt& pt){
  Value yR = builder.create<BigInt::SubOp>(loc, pt.curve()->prime_as_bigint(builder, loc), pt.y());
  return AffinePt(pt.x(), yR, pt.curve());
}

AffinePt doub(OpBuilder builder, Location loc, const AffinePt& pt){
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

  // TODO: This assumes `pt` is actually on the curve and that `pt` is not order 2
  // If you need to check for the latter case, verify that `y_in` is not 0 (mod `prime`)

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

  Value lambda = builder.create<BigInt::MulOp>(loc, lambda_num, two_y_inv);
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
  // Note: `sub` can fail in two ways: A - A (can't write 0) or A - (-A) (doesn't use doubling algorithm) [TODO: Document?]
  auto neg_rhs = neg(builder, loc, rhs);
  return add(builder, loc, lhs, neg_rhs);
}

void ECDSA_verify(OpBuilder builder, Location loc, const AffinePt& base_pt, const AffinePt& pub_key, Value hashed_msg, Value r, Value s, const AffinePt& arbitrary, Value order) {
  pub_key.on_same_curve_as(base_pt);
  arbitrary.on_same_curve_as(base_pt);

  // Note: we don't need to validate `base_pt` on the curve as it's a publicly pre-committed parameter whose presence on the curve we can verify ahead of time
  // (much like we can verify the order of the curve ahead of time)
  pub_key.validate_on_curve(builder, loc);
  // TODO: We will verify `arbitrary` on the curve elsewhere, so this is redundant
  // But CSE eliminates most of the dup'd calculations, so maybe this is fine?
  arbitrary.validate_on_curve(builder, loc);

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
  AffinePt u1G = mul(builder, loc, u1, base_pt, arbitrary);
  AffinePt u2Q = mul(builder, loc, u2, pub_key, arbitrary);
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
    /* TODO*/
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
  auto arbitrary_X = builder.create<BigInt::DefOp>(loc, bits, 7, true);
  auto arbitrary_Y = builder.create<BigInt::DefOp>(loc, bits, 8, true);

  // Add order as a constant
  mlir::Type order_type = builder.getIntegerType(order.getBitWidth());  // TODO: I haven't thought through signedness
  auto order_attr = builder.getIntegerAttr(order_type, order);
  auto order_const = builder.create<BigInt::ConstOp>(loc, order_attr);

  // TODO: Think through if we need to validate any of this (e.g. the orders, points being on curves)
  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt base_pt(base_pt_X, base_pt_Y, curve);
  AffinePt pub_key(pub_key_X, pub_key_Y, curve);
  AffinePt arbitrary(arbitrary_X, arbitrary_Y, curve);

  ECDSA_verify(builder, loc, base_pt, pub_key, msg_hash, r, s, arbitrary, order_const);
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
  // auto order_bits = bits + 1;  // TODO:
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto scale = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  auto xArb = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yArb = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 5, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 6, true);


  // TODO: Basic sanity test section (DELETE ME)
  auto xP_diff = builder.create<BigInt::SubOp>(loc, xP, xP);
  auto yP_diff = builder.create<BigInt::SubOp>(loc, yP, yP);
  auto scale_diff = builder.create<BigInt::SubOp>(loc, scale, scale);
  auto xArb_diff = builder.create<BigInt::SubOp>(loc, xArb, xArb);
  auto yArb_diff = builder.create<BigInt::SubOp>(loc, yArb, yArb);
  auto xR_diff = builder.create<BigInt::SubOp>(loc, xR, xR);
  auto yR_diff = builder.create<BigInt::SubOp>(loc, yR, yR);
  builder.create<BigInt::EqualZeroOp>(loc, xP_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yP_diff);
  builder.create<BigInt::EqualZeroOp>(loc, scale_diff);
  builder.create<BigInt::EqualZeroOp>(loc, xArb_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yArb_diff);
  builder.create<BigInt::EqualZeroOp>(loc, xR_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yR_diff);
  // TODO: End of sanity test section

  auto curve = std::make_shared<WeierstrassCurve>(prime, curve_a, curve_b);
  AffinePt inp(xP, yP, curve);
  AffinePt arb(xArb, yArb, curve);
  AffinePt expected(xR, yR, curve);
  auto result = mul(builder, loc, scale, inp, arb);
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

// void makeECAffineValidatePointOrderTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
// void makeECAffineValidatePointOnCurveTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

// Perf Test function
void makeRepeatedECAffineAddTest(mlir::OpBuilder builder,
                                 mlir::Location loc,
                                 size_t bits,
                                 size_t reps,
                                 APInt prime,
                                 APInt curve_a,
                                 APInt curve_b) {
  // auto order_bits = bits + 1;  // TODO
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
  // auto order_bits = bits + 1;  // TODO
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
