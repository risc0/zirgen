// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/bigint/elliptic_curve.h"

namespace zirgen::BigInt {

void WeierstrassCurve::validate_contains(OpBuilder builder, Location loc, const AffinePt& pt) const {
  auto prime = prime_as_bigint(builder, loc);
  Value y_sqr = builder.create<BigInt::MulOp>(loc, pt.y(), pt.y());
  // y_sqr = builder.create<BigInt::ReduceOp>(loc, y_sqr, prime);  // TODO: Better to skip reduce here and do it on the difference?

  Value x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), pt.x());
  x_cube = builder.create<BigInt::ReduceOp>(loc, x_cube, prime);  // TODO: Reduce isn't required for correctness, perf better if skipped?  // TODO: But seems to make an overflow if dropped?
  x_cube = builder.create<BigInt::MulOp>(loc, pt.x(), x_cube);
  // x_cube = builder.create<BigInt::ReduceOp>(loc, x_cube, prime);  // TODO: Reduce isn't required for correctness, perf better if skipped?

  Value ax = builder.create<BigInt::MulOp>(loc, a_as_bigint(builder, loc), pt.x());
  // ax = builder.create<BigInt::ReduceOp>(loc, ax, prime);  // TODO: Reduce isn't required for correctness, perf better if skipped?

  Value weierstrass_rhs = builder.create<BigInt::AddOp>(loc, x_cube, ax);
  weierstrass_rhs = builder.create<BigInt::AddOp>(loc, weierstrass_rhs, b_as_bigint(builder, loc));
  // weierstrass_rhs = builder.create<BigInt::ReduceOp>(loc, weierstrass_rhs, prime);  // TODO: Better to skip reduce here and do it on the difference?

  Value diff = builder.create<BigInt::SubOp>(loc, y_sqr, weierstrass_rhs);
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

void AffinePt::validate_order(OpBuilder builder, Location loc, const AffinePt& arbitrary) const {
  // This validates that the true order is _at most_ the claimed order (it could be smaller)
  // It does so by confirming that `order * pt == identity`
  // (Specifically, via the equivalent calculation that `(order - 1) * pt == -pt`)
  // NOTE: If the claimed order is prime, this validates that the true order is _exactly_ the claimed order
  // This is because the true order must divide any scalar that sends the point to the identity, and
  // the order can't be 1 b/c the identity can't be represented in affine coords.

  // Construct the constant 1
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  Value order_minus_one = builder.create<BigInt::SubOp>(loc, order(), one);
  AffinePt times_order_minus_one = mul(builder, loc, order_minus_one, *this, arbitrary);
  times_order_minus_one.validate_equal(builder, loc, neg(builder, loc, *this));
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
  y_diff = builder.create<BigInt::AddOp>(loc, y_diff, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity  // TODO: Can this be removed?
  // y_diff = builder.create<BigInt::NondetRemOp>(loc, y_diff, prime);
  Value x_diff = builder.create<BigInt::SubOp>(loc, rhs.x(), lhs.x());
  x_diff = builder.create<BigInt::AddOp>(loc, x_diff, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  // x_diff = builder.create<BigInt::NondetRemOp>(loc, x_diff, prime);


  Value x_diff_inv = builder.create<BigInt::NondetInvModOp>(loc, x_diff, prime);
  // // Enforce that xDiffInv is the inverse of x_diff
  // Value x_diff_inv_check = builder.create<BigInt::MulOp>(loc, x_diff, x_diff_inv);
  // x_diff_inv_check = builder.create<BigInt::ReduceOp>(loc, x_diff_inv_check, prime);
  // x_diff_inv_check = builder.create<BigInt::SubOp>(loc, x_diff_inv_check, one);
  // builder.create<BigInt::EqualZeroOp>(loc, x_diff_inv_check);



  // TODO: Doubling the prime seems to not work well

  Value lambda = builder.create<BigInt::MulOp>(loc, y_diff, x_diff_inv);
  Value k_lambda = builder.create<BigInt::NondetQuotOp>(loc, lambda, prime);
  lambda = builder.create<BigInt::NondetRemOp>(loc, lambda, prime);
  // Verify `lambda` is `y_diff / x_diff` by verifying that `lambda * x_diff == y_diff + k * prime`
  Value lambda_gap = builder.create<BigInt::MulOp>(loc, k_lambda, prime);
  lambda_gap = builder.create<BigInt::MulOp>(loc, lambda_gap, x_diff);  // TODO: implied factor by method of calculating k_lambda
  Value lambda_check = builder.create<BigInt::MulOp>(loc, lambda, x_diff);
  lambda_check = builder.create<BigInt::SubOp>(loc, lambda_check, y_diff);
  lambda_check = builder.create<BigInt::AddOp>(loc, lambda_check, lambda_gap);
  builder.create<BigInt::EqualZeroOp>(loc, lambda_check);   // TODO: This is inadequate, x_diff and y_diff aren't trustworthy -- TODO: I should fix above
  // // TODO: Replaced above with fake
  //   lambda_check = builder.create<BigInt::SubOp>(loc, lambda_check, lambda_check);
  //   builder.create<BigInt::EqualZeroOp>(loc, lambda_check);

  Value nu = builder.create<BigInt::MulOp>(loc, lambda, lhs.x());
  nu = builder.create<BigInt::NondetRemOp>(loc, nu, prime);  // TODO: Reduce isn't required for correctness, perf better if skipped?
  nu = builder.create<BigInt::SubOp>(loc, lhs.y(), nu);
  nu = builder.create<BigInt::AddOp>(loc, nu, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  nu = builder.create<BigInt::NondetRemOp>(loc, nu, prime);

  Value xR = builder.create<BigInt::MulOp>(loc, lambda, lambda);
  xR = builder.create<BigInt::NondetRemOp>(loc, xR, prime);  // TODO: Not needed for correctness, so can experiment with removing
  xR = builder.create<BigInt::SubOp>(loc, xR, lhs.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  xR = builder.create<BigInt::SubOp>(loc, xR, rhs.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity  // TODO: Merge the 2 adds? Not without constant propagation
  xR = builder.create<BigInt::NondetRemOp>(loc, xR, prime);

  Value yR = builder.create<BigInt::MulOp>(loc, lambda, xR);
  yR = builder.create<BigInt::NondetRemOp>(loc, yR, prime);
  yR = builder.create<BigInt::AddOp>(loc, yR, nu);
  yR = builder.create<BigInt::SubOp>(loc, prime, yR);  // i.e., negate (mod prime) 
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity  // TODO: better with using 2*prime for sub?
  // return AffinePt(xR, yR, lhs.curve(), lhs.order());  // TODO: Only for testing
  yR = builder.create<BigInt::AddOp>(loc, yR, prime); // TODO: Just more testing...
  yR = builder.create<BigInt::NondetRemOp>(loc, yR, prime);

  // TODO: Verify the x and y
  // Value x_check = builder.create<BigInt::MulOp>(loc, lambda, lambda); // TODO Do more

  // TODO: This order calculation presumes both points are of the same prime order
  return AffinePt(xR, yR, lhs.curve(), lhs.order());
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

  // This algorithm doesn't work if `scalar` is congruent to 0 mod `order`
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
    auto newX = builder.create<BigInt::ReduceOp>(loc, xMerged, result.curve()->prime_as_bigint(builder, loc));
    auto newY = builder.create<BigInt::ReduceOp>(loc, yMerged, result.curve()->prime_as_bigint(builder, loc));

    // TODO: This assume arbitrary has the same prime order as pt
    result = AffinePt(newX, newY, result.curve(), pt.order());
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
  return AffinePt(pt.x(), yR, pt.curve(), pt.order());
}

AffinePt doub(OpBuilder builder, Location loc, const AffinePt& pt){
  // Formulas (all mod `prime`):
  //   lambda = (3 * xP^2 + a_coeff) / (2 * yP)
  //       nu = yP - lambda * xP
  //       xR = lambda^2 - 2 * xP
  //       yR = -(lambda * xR + nu)

  auto prime = pt.curve()->prime_as_bigint(builder, loc);

  // TODO: How much to reduce? smaller bitwidth is nice, but so are fewer operations...
  // Construct constants
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);
  mlir::Type threeType = builder.getIntegerType(3);  // a `3` is bitwidth 3 (incl. sign)
  auto threeAttr = builder.getIntegerAttr(threeType, 3);  // value 3
  auto three = builder.create<BigInt::ConstOp>(loc, threeAttr);

  Value lambda_num = builder.create<BigInt::MulOp>(loc, pt.x(), pt.x());
  lambda_num = builder.create<BigInt::ReduceOp>(loc, lambda_num, prime);
  lambda_num = builder.create<BigInt::MulOp>(loc, three, lambda_num);  // three is less than prime, so can add again before reduce
  lambda_num = builder.create<BigInt::AddOp>(loc, lambda_num, pt.curve()->a_as_bigint(builder, loc));
  lambda_num = builder.create<BigInt::ReduceOp>(loc, lambda_num, prime);
  Value two_y = builder.create<BigInt::AddOp>(loc, pt.y(), pt.y());
  two_y = builder.create<BigInt::ReduceOp>(loc, two_y, prime);

  // Enforce that two_y_inv is the inverse of lambda_denom
  Value two_y_inv = builder.create<BigInt::NondetInvModOp>(loc, two_y, prime);
  Value two_y_inv_check = builder.create<BigInt::MulOp>(loc, two_y, two_y_inv);
  two_y_inv_check = builder.create<BigInt::ReduceOp>(loc, two_y_inv_check, prime);
  two_y_inv_check = builder.create<BigInt::SubOp>(loc, two_y_inv_check, one);
  builder.create<BigInt::EqualZeroOp>(loc, two_y_inv_check);

  Value lambda = builder.create<BigInt::MulOp>(loc, lambda_num, two_y_inv);
  lambda = builder.create<BigInt::ReduceOp>(loc, lambda, prime);

  Value nu = builder.create<BigInt::MulOp>(loc, lambda, pt.x());
  nu = builder.create<BigInt::ReduceOp>(loc, nu, prime);
  nu = builder.create<BigInt::SubOp>(loc, pt.y(), nu);
  nu = builder.create<BigInt::AddOp>(loc, nu, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  nu = builder.create<BigInt::ReduceOp>(loc, nu, prime);

  Value xR = builder.create<BigInt::MulOp>(loc, lambda, lambda);
  xR = builder.create<BigInt::ReduceOp>(loc, xR, prime);
  xR = builder.create<BigInt::SubOp>(loc, xR, pt.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  xR = builder.create<BigInt::SubOp>(loc, xR, pt.x());
  xR = builder.create<BigInt::AddOp>(loc, xR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  xR = builder.create<BigInt::ReduceOp>(loc, xR, prime);

  Value yR = builder.create<BigInt::MulOp>(loc, lambda, xR);
  yR = builder.create<BigInt::ReduceOp>(loc, yR, prime);
  yR = builder.create<BigInt::AddOp>(loc, yR, nu);
  yR = builder.create<BigInt::SubOp>(loc, prime, yR);  // i.e., negate (mod prime)
  yR = builder.create<BigInt::AddOp>(loc, yR, prime);  // TODO: Reduce op doesn't work with negatives, so enforcing positivity
  yR = builder.create<BigInt::ReduceOp>(loc, yR, prime);

  // TODO: This order calculation assumes pt.order() is relatively prime to 2 (i.e. odd)
  return AffinePt(xR, yR, pt.curve(), pt.order());
}

AffinePt sub(OpBuilder builder, Location loc, const AffinePt& lhs, const AffinePt& rhs) {
  // Note: `sub` can fail in two ways: A - A (can't write 0) or A - (-A) (doesn't use doubling algorithm) [TODO: Document?]
  auto neg_rhs = neg(builder, loc, rhs);
  return add(builder, loc, lhs, neg_rhs);
}

void ECDSA_verify(OpBuilder builder, Location loc, const AffinePt& base_pt, const AffinePt& pub_key, Value hashed_msg, Value r, Value s, const AffinePt& arbitrary) {
  pub_key.on_same_curve_as(base_pt);
  arbitrary.on_same_curve_as(base_pt);

  // Note: we don't need to validate `base_pt` on the curve as it's a publicly pre-committed parameter whose presence on the curve we can verify ahead of time
  // (much like we can verify the order of the curve ahead of time)
  pub_key.validate_on_curve(builder, loc);
  // TODO: We will verify `arbitrary` on the curve elsewhere, so this is redundant
  // But CSE eliminates most of the dup'd calculations, so maybe this is fine?
  arbitrary.validate_on_curve(builder, loc);

  assert(pub_key.order() == base_pt.order());  // TODO: I... think this isn't right, I think it's comparing _Values_ _outside_ the circuit (TODO Check)
  // Note: We don't need to validate `base_pt`'s order as it's a publicly pre-committed parameter whose order we can verify ahead of time
  pub_key.validate_order(builder, loc, arbitrary);
  // Mathematically, we need to ensure also that `pub_key` is not the identity.
  // But it's not possible to express the identity in affine coords, so this comes for free just by validating the point is on the curve

  // Construct constants
  // TODO: Check that this doesn't contruct multiple `one`s when called in various places
  mlir::Type oneType = builder.getIntegerType(1);  // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1);  // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  // Compute s_inv
  Value s_inv = builder.create<BigInt::NondetInvModOp>(loc, s, base_pt.order());  // TODO: Do we need to handle `order` specially?
  Value s_inv_check = builder.create<BigInt::MulOp>(loc, s, s_inv);
  s_inv_check = builder.create<BigInt::ReduceOp>(loc, s_inv_check, base_pt.order());  // TODO: Do we need to handle `order` specially?
  s_inv_check = builder.create<BigInt::SubOp>(loc, s_inv_check, one);
  builder.create<BigInt::EqualZeroOp>(loc, s_inv_check);

  // Compute u multipliers
  Value u1 = builder.create<BigInt::MulOp>(loc, hashed_msg, s_inv);
  u1 = builder.create<BigInt::ReduceOp>(loc, u1, base_pt.order());  // TODO: Do we need to handle `order` specially?
  Value u2 = builder.create<BigInt::MulOp>(loc, r, s_inv);
  u2 = builder.create<BigInt::ReduceOp>(loc, u2, base_pt.order());  // TODO: Do we need to handle `order` specially?

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
    APInt curve_b
    /* TODO*/
) {
  // TODO: What can we move from an input to a parameter?
  // Since curve order differs from prime by at most 2 * sqrt(prime), we only need 1 more bit than `prime`
  auto order_bits = bits + 1;
  auto base_pt_X = builder.create<BigInt::DefOp>(loc, bits, 0, true);  // TODO: Or get from a parameter to this call?
  auto base_pt_Y = builder.create<BigInt::DefOp>(loc, bits, 1, true);  // TODO: Or get from a parameter to this call?
  auto base_pt_order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto pub_key_X = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto pub_key_Y = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto msg_hash = builder.create<BigInt::DefOp>(loc, order_bits, 5, true);
  auto r = builder.create<BigInt::DefOp>(loc, order_bits, 6, true);
  auto s = builder.create<BigInt::DefOp>(loc, order_bits, 7, true);
  auto arbitrary_X = builder.create<BigInt::DefOp>(loc, bits, 8, true);
  auto arbitrary_Y = builder.create<BigInt::DefOp>(loc, bits, 9, true);

  // TODO: Think through if we need to validate any of this (e.g. the orders, points being on curves)
  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt base_pt(base_pt_X, base_pt_Y, curve, base_pt_order);
  AffinePt pub_key(pub_key_X, pub_key_Y, curve, base_pt_order);
  AffinePt arbitrary(arbitrary_X, arbitrary_Y, curve, base_pt_order);

  ECDSA_verify(builder, loc, base_pt, pub_key, msg_hash, r, s, arbitrary);
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
  // auto order_bits = bits + 1;  // TODO
  auto order_bits = bits;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 5, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 6, true);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO

  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt lhs(xP, yP, curve, order);
  AffinePt rhs(xQ, yQ, curve, order);
  AffinePt expected(xR, yR, curve, order);
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
  auto order_bits = bits + 1;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt inp(xP, yP, curve, order);
  AffinePt expected(xR, yR, curve, order);
  auto result = doub(builder, loc, inp);
  result.validate_equal(builder, loc, expected);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO
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
  // auto order = builder.create<BigInt::DefOp>(loc, order_bits, 5, true);  // TODO: Or get from a parameter to this call?  // TODO
  auto order = builder.create<BigInt::DefOp>(loc, bits, 5, true);  // TODO: Or get from a parameter to this call?
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 6, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 7, true);


  // TODO: Basic sanity test section (DELETE ME)
  auto xP_diff = builder.create<BigInt::SubOp>(loc, xP, xP);
  auto yP_diff = builder.create<BigInt::SubOp>(loc, yP, yP);
  auto scale_diff = builder.create<BigInt::SubOp>(loc, scale, scale);
  auto xArb_diff = builder.create<BigInt::SubOp>(loc, xArb, xArb);
  auto yArb_diff = builder.create<BigInt::SubOp>(loc, yArb, yArb);
  auto order_diff = builder.create<BigInt::SubOp>(loc, order, order);
  auto xR_diff = builder.create<BigInt::SubOp>(loc, xR, xR);
  auto yR_diff = builder.create<BigInt::SubOp>(loc, yR, yR);
  builder.create<BigInt::EqualZeroOp>(loc, xP_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yP_diff);
  builder.create<BigInt::EqualZeroOp>(loc, scale_diff);
  builder.create<BigInt::EqualZeroOp>(loc, xArb_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yArb_diff);
  builder.create<BigInt::EqualZeroOp>(loc, order_diff);
  builder.create<BigInt::EqualZeroOp>(loc, xR_diff);
  builder.create<BigInt::EqualZeroOp>(loc, yR_diff);
  // TODO: End of sanity test section

  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt inp(xP, yP, curve, order);
  AffinePt arb(xArb, yArb, curve, order);
  AffinePt expected(xR, yR, curve, order);
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
  // auto order_bits = bits + 1;  // TODO
  auto order_bits = bits;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt inp(xP, yP, curve, order);
  AffinePt expected(xR, yR, curve, order);
  auto result = neg(builder, loc, inp);
  result.validate_equal(builder, loc, expected);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO
}

void makeECAffineSubtractTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  auto order_bits = bits + 1;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 5, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 6, true);
  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt lhs(xP, yP, curve, order);
  AffinePt rhs(xQ, yQ, curve, order);
  AffinePt expected(xR, yR, curve, order);
  auto result = sub(builder, loc, lhs, rhs);
  result.validate_equal(builder, loc, expected);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO
}

void makeECAffineValidatePointsEqualTest(
    mlir::OpBuilder builder,
    mlir::Location loc,
    size_t bits,
    APInt prime,
    APInt curve_a,
    APInt curve_b
) {
  // auto order_bits = bits + 1;  // Skip for this test, causing [11, 0] instead of [11] and debugging that doesn't seem critical
  auto order_bits = bits;  // TODO: See above, this is ok for now
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true);  // TODO: Or get from a parameter to this call?
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt lhs(xP, yP, curve, order);
  AffinePt rhs(xQ, yQ, curve, order);
  lhs.validate_equal(builder, loc, rhs);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO
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
  auto order_bits = bits;
  auto xP = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto yP = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto order = builder.create<BigInt::DefOp>(loc, order_bits, 2, true); // TODO: Or get from a parameter to this call?
  auto xQ = builder.create<BigInt::DefOp>(loc, bits, 3, true);
  auto yQ = builder.create<BigInt::DefOp>(loc, bits, 4, true);
  auto xR = builder.create<BigInt::DefOp>(loc, bits, 5, true);
  auto yR = builder.create<BigInt::DefOp>(loc, bits, 6, true);

  // TODO: Empty tests to make the compiler happy
  auto order_TODO = builder.create<BigInt::SubOp>(loc, order, order);
  builder.create<BigInt::EqualZeroOp>(loc, order_TODO);
  // END TODO

  auto curve = std::make_shared<WeierstrassCurve>(curve_a, curve_b, prime);
  AffinePt lhs(xP, yP, curve, order);
  AffinePt rhs(xQ, yQ, curve, order);
  AffinePt expected(xR, yR, curve, order);
  auto result = add(builder, loc, lhs, rhs);
  // iterate from 1 because the first repition was already done
  for (size_t rp = 1; rp < reps; rp++) {
    result = add(builder, loc, result, rhs);
  }
  result.validate_equal(builder, loc, expected);
}

} // namespace zirgen::BigInt
