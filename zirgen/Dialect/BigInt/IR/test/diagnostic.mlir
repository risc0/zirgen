// RUN: zirgen-opt %s -split-input-file -verify-diagnostics

// TODO: are the intended semantics for `min_bits` that the value must be positive? Or is it a bound on absolute value? Status quo is "must be positive"
// TODO: Add verifier that at least one of `max_neg` and `min_bits` must be zero

// TODO: Test the following:
// For `add`:
//  - `coeffs` is max of the input coeffs
//  - `max_pos` is the sum of the input `max_pos`s
//  - `max_neg` is the sum of the input `max_neg`s
//  - If both inputs are nonnegative, `min_bits` is max of input `min_bits`s
//  - If either input may be negative, `min_bits` is 0
// For `sub` (A - B):
//  - `coeffs` is max of the input coeffs
//  - `max_pos` is A's `max_pos` plus B's `max_neg`
//  - `max_neg` is A's `max_neg` plus B's `max_pos`
//  - Probably: just set `min_bits` to 0 (TODO but could be more precise)
// For `mul`:
//  - `coeffs` is the sum of the input coeffs minus 1 [TODO: Confirm no carries]
//  - `max_pos` is the max of the product of the `max_pos` and the product of the `max_neg`
//  - `max_neg` is the max of the two mixed products (of one `max_pos` and one `max_neg`)
//  - If both inputs are nonnegative, `min_bits` is the sum of input `min_bits`s
//  - If either input may be negative, `min_bits` is zero
// For [nondets]:
//  - In general, nondets will only return nonnegative answers
//  - In general, nondets will return values with normalized coeffs (and therefore potentially more coeffs than if unnormalized)
//    - The max possible overall value can be computed as `max_pos` (of the unnormalized form) times the sum from i=0..coeffs of 256^i
//      - Then 1 + the floor of log_256 of this value is the number of coeffs
//  - So in normalized form `max_pos = 255` and `max_neg = 0`
// For `nondet_quot`:
//  - For `coeffs`:
//    - Compute the max overall value from the numerator by the algorithm from the general nondets section
//    - Divide this by `2^min_bits` of the denominator
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0 (might be clever tricks in restrictive circumstances, but IMO shouldn't bother)
// For `nondet_rem`:
//  - For `coeffs`:
//    - Compute the max overall value from the denominator - 1 by the algorithm from the general nondets section
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0 (might be clever tricks in restrictive circumstances, but IMO shouldn't bother)
// For `nondet_inv_mod`:
//  - For `coeffs`:
//    - Compute the max overall value from the modulus - 1 by the algorithm from the general nondets section
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0 (might be clever tricks in restrictive circumstances, but IMO shouldn't bother)
// For `modular_inv`:
//  - Same as `nondet_inv_mod`
// For `reduce`:
//  - Same as `nondet_rem`

func.func @good_add_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  return
}

// -----

func.func @good_add_with_min_bits() {
  // Primary rules tested:
  //  - [%3] If both `add` inputs are nonnegative, `min_bits` is max of input `min_bits`s
  //  - [%5, %6] If either input to `add` may be negative, `min_bits` is 0
  // This is calculating 7 + 8 [in %3] and 8 + (0 - 7) [in %5 and %6]
  %0 = bigint.const 0 : i8 -> <1, 255, 0, 0>
  %1 = bigint.const 7 : i8 -> <1, 255, 0, 3>
  %2 = bigint.const 8 : i8 -> <1, 255, 0, 4>
  %3 = bigint.add %1 : <1, 255, 0, 3>, %2 : <1, 255, 0, 4> -> <1, 510, 0, 4>
  %4 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 3> -> <1, 255, 255, 0>
  %5 = bigint.add %2 : <1, 255, 0, 4>, %4 : <1, 255, 255, 0> -> <1, 510, 255, 0>
  %6 = bigint.add %4 : <1, 255, 255, 0>, %2 : <1, 255, 0, 4> -> <1, 510, 255, 0>
  return
}

// -----

func.func @bad_add_max_pos() {
  // Primary rules tested:
  //  - [%2] `add`'s `max_pos` is the sum of the input `max_pos`s
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  // expected-error@+2 {{op inferred type(s)}}
  // expected-error@+1 {{failed to infer returned types}}
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @good_sub_max_pos_max_neg() {
  // Primary rules tested:
  //  - [%3] For A - B: `max_pos` is A's `max_pos` plus B's `max_neg`
  //  - [%4] For A - B: `max_neg` is A's `max_neg` plus B's `max_pos`
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 255, 0>
  %3 = bigint.sub %0 : <1, 255, 0, 0>, %2 : <1, 255, 255, 0> -> <1, 510, 255, 0>
  %4 = bigint.sub %2 : <1, 255, 255, 0>, %3 : <1, 510, 255, 0> -> <1, 510, 765, 0>
  return
}

// -----

func.func @bad_sub_max_pos() {
  // Primary rules tested:
  //  - [%3] For A - B: `max_pos` is A's `max_pos` plus B's `max_neg`
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 255, 0>
  // expected-error@+2 {{op inferred type(s)}}
  // expected-error@+1 {{failed to infer returned types}}
  %3 = bigint.sub %0 : <1, 255, 0, 0>, %2 : <1, 255, 255, 0> -> <1, 255, 255, 0>
  return
}

// -----

func.func @bad_sub_max_neg() {
  // Primary rules tested:
  //  - [%4] For A - B: `max_neg` is A's `max_neg` plus B's `max_pos`
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 255, 0>
  %3 = bigint.sub %0 : <1, 255, 0, 0>, %2 : <1, 255, 255, 0> -> <1, 510, 255, 0>
  // expected-error@+2 {{op inferred type(s)}}
  // expected-error@+1 {{failed to infer returned types}}
  %4 = bigint.sub %2 : <1, 255, 255, 0>, %3 : <1, 510, 255, 0> -> <1, 510, 510, 0>
  return
}

// -----

func.func @good_sub_unique_nonzero_maxs() {
  // Primary rules tested:
  //  - [%9] For A - B: `max_pos` is A's `max_pos` plus B's `max_neg`
  //  - [%9] For A - B: `max_neg` is A's `max_neg` plus B's `max_pos`
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.def 8, 2, true -> <1, 255, 0, 0>
  %3 = bigint.def 8, 3, true -> <1, 255, 0, 0>
  %4 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 65025, 0, 0>
  %5 = bigint.add %2 : <1, 255, 0, 0>, %3 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %6 = bigint.sub %4 : <1, 65025, 0, 0>, %5 : <1, 510, 0, 0> -> <1, 65025, 510, 0>
  %7 = bigint.sub %6 : <1, 65025, 510, 0>, %0 : <1, 255, 0, 0> -> <1, 65025, 765, 0>
  %8 = bigint.sub %5 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 255, 0>
  %9 = bigint.sub %7 : <1, 65025, 765, 0>, %8 : <1, 510, 255, 0> -> <1, 65280, 1275, 0>
  return
}
