// RUN: zirgen-opt %s -split-input-file -verify-diagnostics

// Type inference for `add`:
//
//  - `coeffs` is max of the input coeffs
//  - `max_pos` is the sum of the input `max_pos`s
//  - `max_neg` is the sum of the input `max_neg`s
//  - If both inputs are nonnegative, `min_bits` is max of input `min_bits`s
//  - If either input may be negative, `min_bits` is 0

func.func @add_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  return
}

// -----

func.func @add_coeff_count() {
  // Primary rules tested:
  //  - [%2, %3] `coeffs` is max of the input coeffs
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.add %1 : <8, 255, 0, 0>, %0 : <3, 255, 0, 0> -> <8, 510, 0, 0>
  return
}

// -----

func.func @add_multisize() {
  // Primary rules tested:
  //  - [%7, %8] `max_pos` is the sum of the input `max_pos`s
  //  - [%7, %8] `max_neg` is the sum of the input `max_neg`s
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.add %0 : <1, 255, 0, 0>, %2 : <1, 510, 0, 0> -> <1, 765, 0, 0>
  %4 = bigint.sub %3 : <1, 765, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 765, 255, 0>
  %5 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 65025, 0, 0>
  %6 = bigint.sub %5 : <1, 65025, 0, 0>, %2 : <1, 510, 0, 0> -> <1, 65025, 510, 0>
  %7 = bigint.add %4 : <1, 765, 255, 0>, %6 : <1, 65025, 510, 0> -> <1, 65790, 765, 0>
  %8 = bigint.add %6 : <1, 65025, 510, 0>, %4 : <1, 765, 255, 0> -> <1, 65790, 765, 0>
  return
}

// -----

func.func @add_min_bits() {
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

// Type inference for `sub` (A - B):
//
//  - `coeffs` is max of the input coeffs
//  - `max_pos` is A's `max_pos` plus B's `max_neg`
//  - `max_neg` is A's `max_neg` plus B's `max_pos`
//  - just set `min_bits` to 0

func.func @sub_coeff_count() {
  // Primary rules tested:
  //  - [%2, %3] `coeffs` is max of the input coeffs
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.sub %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 255, 255, 0>
  %3 = bigint.sub %1 : <8, 255, 0, 0>, %0 : <3, 255, 0, 0> -> <8, 255, 255, 0>
  return
}

// -----

func.func @sub_max_pos_max_neg() {
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

func.func @sub_multisize() {
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

// -----

func.func @sub_min_bits() {
  // Primary rules tested:
  //  - just set `min_bits` to 0 [This could be more complicated, but we don't bother]
  %0 = bigint.const 0 : i8 -> <1, 255, 0, 0>
  %1 = bigint.const 7 : i8 -> <1, 255, 0, 3>
  %2 = bigint.const 8 : i8 -> <1, 255, 0, 4>
  %3 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 3> -> <1, 255, 255, 0>
  %4 = bigint.sub %1 : <1, 255, 0, 3>, %0 : <1, 255, 0, 0> -> <1, 255, 255, 0>
  %5 = bigint.sub %2 : <1, 255, 0, 4>, %1 : <1, 255, 0, 3> -> <1, 255, 255, 0>
  %6 = bigint.sub %1 : <1, 255, 0, 3>, %2 : <1, 255, 0, 4> -> <1, 255, 255, 0>
  return
}

// -----

// Type inference for `mul`:
//
//  - `coeffs` is the sum of the input coeffs minus 1
//  - `max_pos` is the smaller `coeffs` value from the two inputs times
//     the max of the product of the `max_pos` and the product of the `max_neg`
//  - `max_neg` is the smaller `coeffs` value from the two inputs times
//     the max of the two mixed products (of one `max_pos` and one `max_neg`)
//  - If both inputs are nonnegative, `min_bits` is the sum of input `min_bits`s minus 1
//  - If either input may be negative or zero, `min_bits` is zero

func.func @mul_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 65025, 0, 0>
  return
}

// -----

func.func @mul_coeff_count() {
  // Primary rules tested:
  //  - [%2, %3] `coeffs` is the sum of the input coeffs minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.mul %1 : <8, 255, 0, 0>, %0 : <3, 255, 0, 0> -> <10, 195075, 0, 0>
  return
}

// -----

func.func @mul_multisize() {
  // Primary rules tested:
  //  - [%8 - %11] `max_pos` is the smaller `coeffs` value from the two inputs times
  //    the max of the product of the `max_pos` and the product of the `max_neg`
  //  - [%8 - %11] `max_neg` is the smaller `coeffs` value from the two inputs times
  //    the max of the two mixed products (of one `max_pos` and one `max_neg`)
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.add %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <8, 765, 0, 0>
  %4 = bigint.sub %3 : <8, 765, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 765, 255, 0>
  %5 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 65025, 0, 0>
  %6 = bigint.sub %5 : <8, 65025, 0, 0>, %2 : <8, 510, 0, 0> -> <8, 65025, 510, 0>
  %7 = bigint.sub %2 : <8, 510, 0, 0>, %5 : <8, 65025, 0, 0> -> <8, 510, 65025, 0>
  %8 = bigint.mul %4 : <8, 765, 255, 0>, %6 : <8, 65025, 510, 0> -> <15, 397953000, 132651000, 0>
  %9 = bigint.mul %6 : <8, 65025, 510, 0>, %4 : <8, 765, 255, 0> -> <15, 397953000, 132651000, 0>
  %10 = bigint.mul %4 : <8, 765, 255, 0>, %7 : <8, 510, 65025, 0> -> <15, 132651000, 397953000, 0>
  %11 = bigint.mul %7 : <8, 510, 65025, 0>, %4 : <8, 765, 255, 0> -> <15, 132651000, 397953000, 0>
  return
}

// -----

func.func @mul_min_bits() {
  // Primary rules tested:
  //  - [%3] If both inputs are nonnegative, `min_bits` is the sum of input `min_bits`s minus 1
  //  - [%5, %6] If either input may be negative, `min_bits` is zero
  // This is calculating 7 + 8 [in %3] and 8 + (0 - 7) [in %5 and %6]
  %0 = bigint.const 0 : i8 -> <1, 255, 0, 0>
  %1 = bigint.const 7 : i8 -> <1, 255, 0, 3>
  %2 = bigint.const 8 : i8 -> <1, 255, 0, 4>
  %3 = bigint.mul %1 : <1, 255, 0, 3>, %2 : <1, 255, 0, 4> -> <1, 65025, 0, 6>
  %4 = bigint.sub %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 3> -> <1, 255, 255, 0>
  %5 = bigint.mul %2 : <1, 255, 0, 4>, %4 : <1, 255, 255, 0> -> <1, 65025, 65025, 0>
  %6 = bigint.mul %4 : <1, 255, 255, 0>, %2 : <1, 255, 0, 4> -> <1, 65025, 65025, 0>
  return
}

// -----

// For nondets generally:
//  - Nondets will only return nonnegative answers
//    - In many cases, they cannot give correct answers when negative inputs are provided
//    - Negative inputs are still allowed (for cases such as when a developer knows more than the type system)
//    - Nondets should always be appropriately constrained, including failing if necessary on negative inputs
//      - ReduceOp and InvOp come with constraints built in
//  - Nondets will return values with normalized coeffs (and therefore potentially more coeffs than if unnormalized)
//    - The max possible overall value can be computed as `max_pos` (of the unnormalized form) times the sum from i=0..coeffs of 256^i
//      - Then 1 + the floor of log_256 of this value is the number of coeffs
//  - So in normalized form `max_pos = 255` and `max_neg = 0`
//  - `min_bits` is zero to avoid a semi-hidden responsibility for checking that the input is in bounds

// Type inference for `nondet_quot`:
//
//  - For `coeffs`:
//    - Compute the max overall value from the numerator by the algorithm from the general nondets section
//    - Divide this by `2^(min_bits - 1)` of the denominator
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0

func.func @nondet_quot_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.nondet_quot %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_oversized_num() {
  // Primary rules tested:
  //  - [%3] Compute the max overall value from the numerator
  //  - [%3] Return values with normalized coeffs (potentially more coeffs than if unnormalized)
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.nondet_quot %2 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_multibyte_num() {
  // Primary rules tested:
  //  - [%3] Compute the max overall value from the numerator
  //  - [%3] Return values with normalized coeffs (potentially more coeffs than if unnormalized)
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_quot %2 : <8, 510, 0, 0>, %0 : <3, 255, 0, 0> -> <9, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_multibyte_num2() {
  // Primary rules tested:
  //  - [%3] Compute the max overall value from the numerator
  //  - [%3] Return values with normalized coeffs (potentially more coeffs than if unnormalized)
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.nondet_quot %2 : <10, 195075, 0, 0>, %0 : <3, 255, 0, 0> -> <12, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_multibyte_denom() {
  // Primary rules tested:
  //  - [%3] Compute the max overall value from the numerator
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_quot %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_1bit_denom() {
  // Primary rules tested:
  //  - [%2] Compute the max overall value from the numerator
  //  - [%2] As part of computing Coeffs, divide this by `2^(min_bits - 1)` of the denominator
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 1 : i8 -> <1, 255, 0, 1>
  %2 = bigint.nondet_quot %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 1> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_8bit_denom() {
  // Primary rules tested:
  //  - [%2] Compute the max overall value from the numerator
  //  - [%2] As part of computing Coeffs, divide this by `2^(min_bits - 1)` of the denominator
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.nondet_quot %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 8> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_9bit_denom() {
  // Primary rules tested:
  //  - [%2] Compute the max overall value from the numerator
  //  - [%2] As part of computing Coeffs, divide this by `2^(min_bits - 1)` of the denominator
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %2 = bigint.nondet_quot %0 : <3, 255, 0, 0>, %1 : <2, 255, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_9bit_1coeff_denom() {
  // Primary rules tested:
  //  - [%4] Compute the max overall value from the numerator
  //  - [%4] As part of computing Coeffs, divide this by `2^(min_bits - 1)` of the denominator
  //  - [%4] Return values with normalized coeffs (potentially more coeffs than if unnormalized)
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.const 2 : i8 -> <1, 255, 0, 2>
  %3 = bigint.mul %1 : <1, 255, 0, 8>, %2 : <1, 255, 0, 2> -> <1, 65025, 0, 9>
  %4 = bigint.nondet_quot %0 : <3, 255, 0, 0>, %3 : <1, 65025, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_num_minbits() {
  // Primary rules tested:
  //  - [%2] `min_bits` of `nondet_quot` result is always 0
  %0 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %1 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %2 = bigint.nondet_quot %0 : <2, 255, 0, 9>, %1 : <1, 255, 0, 0> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_ignore_negatives() {
  %0 = bigint.def 16, 0, true -> <2, 255, 0, 0>
  %1 = bigint.def 32, 0, true -> <4, 255, 0, 0>
  %2 = bigint.sub %0 : <2, 255, 0, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 255, 0>
  %3 = bigint.sub %2 : <4, 255, 255, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 510, 0>
  %4 = bigint.nondet_quot %3 : <4, 255, 510, 0>, %0 : <2, 255, 0, 0> -> <4, 255, 0, 0>
  %5 = bigint.nondet_quot %0 : <2, 255, 0, 0>, %3 : <4, 255, 510, 0> -> <2, 255, 0, 0>
  %6 = bigint.nondet_quot %2 : <4, 255, 255, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %7 = bigint.nondet_quot %3 : <4, 255, 510, 0>, %2 : <4, 255, 255, 0> -> <4, 255, 0, 0>
  return
}

// -----

// Type inference for `nondet_rem`:
//
//  - For `coeffs`:
//    - Compute the max overall value from the denominator - 1 by the algorithm from the general nondets section
//    - Compute the max overall value from the numerator by the algorithm from the general nondets section
//    - Choose the smaller of these two numbers
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0 (might be clever tricks in restrictive circumstances, but IMO shouldn't bother)
//
// We also test the `reduce` op here as it should produce the exact same type

func.func @nondet_rem_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.nondet_rem %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  %3 = bigint.reduce %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_oversized_num() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.nondet_rem %2 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  %4 = bigint.reduce %2 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_oversized_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the numerator
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.nondet_rem %1 : <1, 255, 0, 0>, %2 : <1, 510, 0, 0> -> <1, 255, 0, 0>
  %4 = bigint.reduce %1 : <1, 255, 0, 0>, %2 : <1, 510, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_multibyte_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_rem %2 : <8, 510, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  %4 = bigint.reduce %2 : <8, 510, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_multibyte_denom2() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.nondet_rem %2 : <10, 195075, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  %4 = bigint.reduce %2 : <10, 195075, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_multibyte_denom3() {
  // Primary rules tested:
  //  - Compute the max overall value from the numerator
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_rem %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <1, 255, 0, 0>
  %4 = bigint.reduce %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_multibyte_denom4() {
  // Primary rules tested:
  //  - Compute the max overall value from the numerator
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.nondet_rem %0 : <3, 255, 0, 0>, %2 : <10, 195075, 0, 0> -> <3, 255, 0, 0>
  %4 = bigint.reduce %0 : <3, 255, 0, 0>, %2 : <10, 195075, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_1bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 1 : i8 -> <1, 255, 0, 1>
  %2 = bigint.nondet_rem %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 1> -> <1, 255, 0, 0>
  %3 = bigint.reduce %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 1> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_8bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.nondet_rem %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 8> -> <1, 255, 0, 0>
  %3 = bigint.reduce %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 8> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_9bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %2 = bigint.nondet_rem %0 : <3, 255, 0, 0>, %1 : <2, 255, 0, 9> -> <2, 255, 0, 0>
  %3 = bigint.reduce %0 : <3, 255, 0, 0>, %1 : <2, 255, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_9bit_1coeff_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.const 2 : i8 -> <1, 255, 0, 2>
  %3 = bigint.mul %1 : <1, 255, 0, 8>, %2 : <1, 255, 0, 2> -> <1, 65025, 0, 9>
  %4 = bigint.nondet_rem %0 : <3, 255, 0, 0>, %3 : <1, 65025, 0, 9> -> <2, 255, 0, 0>
  %5 = bigint.reduce %0 : <3, 255, 0, 0>, %3 : <1, 65025, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_num_minbits() {
  // Primary rules tested:
  //  - `min_bits` is 0
  %0 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %1 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %2 = bigint.nondet_rem %0 : <2, 255, 0, 9>, %1 : <3, 255, 0, 0> -> <2, 255, 0, 0>
  %3 = bigint.reduce %0 : <2, 255, 0, 9>, %1 : <3, 255, 0, 0> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_rem_coeff_carry() {
  // Primary rules tested:
  //  - `min_bits` is 0
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 16, 0, true -> <2, 255, 0, 0>
  %2 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %3 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <2, 255, 0, 0> -> <2, 65025, 0, 0>
  // Note: although the maximum value of BigInt<2, 65025, 0, 0> would fit into a BigInt<3, 255, 0, 0>,
  // the type inference system approximates to the next highest power of 2 (minus 1, so 65535 in this case).
  // a BigInt<2, 65535, 0, 0> would not fit into a BigInt<3, 255, 0, 0>, so we use BigInt<4, 255, 0, 0>
  // here instead, even though that results in an unused (always 0) coefficient for BigInt<2, 65025, 0, 0>.
  // See `getMaxPosBits` and its comments for more details.
  %4 = bigint.nondet_rem %2 : <8, 255, 0, 0>, %3 : <2, 65025, 0, 0> -> <4, 255, 0, 0>
  %5 = bigint.reduce %2 : <8, 255, 0, 0>, %3 : <2, 65025, 0, 0> -> <4, 255, 0, 0>
  return
}

// -----

func.func @nondet_quot_ignore_negatives() {
  %0 = bigint.def 16, 0, true -> <2, 255, 0, 0>
  %1 = bigint.def 32, 0, true -> <4, 255, 0, 0>
  %2 = bigint.sub %0 : <2, 255, 0, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 255, 0>
  %3 = bigint.sub %2 : <4, 255, 255, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 510, 0>
  %4 = bigint.nondet_rem %3 : <4, 255, 510, 0>, %0 : <2, 255, 0, 0> -> <2, 255, 0, 0>
  %5 = bigint.nondet_rem %0 : <2, 255, 0, 0>, %3 : <4, 255, 510, 0> -> <2, 255, 0, 0>
  %6 = bigint.nondet_rem %2 : <4, 255, 255, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %7 = bigint.nondet_rem %3 : <4, 255, 510, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %8 = bigint.reduce %3 : <4, 255, 510, 0>, %0 : <2, 255, 0, 0> -> <2, 255, 0, 0>
  %9 = bigint.reduce %0 : <2, 255, 0, 0>, %3 : <4, 255, 510, 0> -> <2, 255, 0, 0>
  %10 = bigint.reduce %2 : <4, 255, 255, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %11 = bigint.reduce %3 : <4, 255, 510, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  return
}

// -----

// Type inference for `nondet_inv`:
//
//  - For `coeffs`:
//    - Compute the max overall value from the denominator - 1 by the algorithm from the general nondets section
//    - Compute the coeffs from this number by the algorithm from the general nondets section
//  - `max_pos` is 255
//  - `max_neg` is 0
//  - `min_bits` is 0
//
// We also test the `inv` op here as it should produce the exact same type

func.func @nondet_inv_basic() {
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.nondet_inv %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  %3 = bigint.inv %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_oversized_num() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.nondet_inv %2 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  %4 = bigint.inv %2 : <1, 510, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_oversized_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 8, 1, true -> <1, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <1, 255, 0, 0> -> <1, 510, 0, 0>
  %3 = bigint.nondet_inv %1 : <1, 255, 0, 0>, %2 : <1, 510, 0, 0> -> <2, 255, 0, 0>
  %4 = bigint.inv %1 : <1, 255, 0, 0>, %2 : <1, 510, 0, 0> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_multibyte_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_inv %2 : <8, 510, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  %4 = bigint.inv %2 : <8, 510, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_multibyte_denom2() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.nondet_inv %2 : <10, 195075, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  %4 = bigint.inv %2 : <10, 195075, 0, 0>, %0 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_multibyte_denom3() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.add %0 : <1, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <8, 510, 0, 0>
  %3 = bigint.nondet_inv %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <9, 255, 0, 0>
  %4 = bigint.inv %0 : <1, 255, 0, 0>, %2 : <8, 510, 0, 0> -> <9, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_multibyte_denom4() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %2 = bigint.mul %0 : <3, 255, 0, 0>, %1 : <8, 255, 0, 0> -> <10, 195075, 0, 0>
  %3 = bigint.nondet_inv %0 : <3, 255, 0, 0>, %2 : <10, 195075, 0, 0> -> <12, 255, 0, 0>
  %4 = bigint.inv %0 : <3, 255, 0, 0>, %2 : <10, 195075, 0, 0> -> <12, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_1bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 1 : i8 -> <1, 255, 0, 1>
  %2 = bigint.nondet_inv %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 1> -> <1, 255, 0, 0>
  %3 = bigint.inv %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 1> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_8bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.nondet_inv %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 8> -> <1, 255, 0, 0>
  %3 = bigint.inv %0 : <3, 255, 0, 0>, %1 : <1, 255, 0, 8> -> <1, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_9bit_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %2 = bigint.nondet_inv %0 : <3, 255, 0, 0>, %1 : <2, 255, 0, 9> -> <2, 255, 0, 0>
  %3 = bigint.inv %0 : <3, 255, 0, 0>, %1 : <2, 255, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_9bit_1coeff_denom() {
  // Primary rules tested:
  //  - Compute the max overall value from the denominator max value minus 1
  //  - `min_bits` is 0
  %0 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %1 = bigint.const 200 : i8 -> <1, 255, 0, 8>
  %2 = bigint.const 2 : i8 -> <1, 255, 0, 2>
  %3 = bigint.mul %1 : <1, 255, 0, 8>, %2 : <1, 255, 0, 2> -> <1, 65025, 0, 9>
  %4 = bigint.nondet_inv %0 : <3, 255, 0, 0>, %3 : <1, 65025, 0, 9> -> <2, 255, 0, 0>
  %5 = bigint.inv %0 : <3, 255, 0, 0>, %3 : <1, 65025, 0, 9> -> <2, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_num_minbits() {
  // Primary rules tested:
  //  - `min_bits` is 0
  %0 = bigint.const 300 : i16 -> <2, 255, 0, 9>
  %1 = bigint.def 24, 0, true -> <3, 255, 0, 0>
  %2 = bigint.nondet_inv %0 : <2, 255, 0, 9>, %1 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  %3 = bigint.inv %0 : <2, 255, 0, 9>, %1 : <3, 255, 0, 0> -> <3, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_coeff_carry() {
  // Primary rules tested:
  //  - `min_bits` is 0
  %0 = bigint.def 8, 0, true -> <1, 255, 0, 0>
  %1 = bigint.def 16, 0, true -> <2, 255, 0, 0>
  %2 = bigint.def 64, 1, true -> <8, 255, 0, 0>
  %3 = bigint.mul %0 : <1, 255, 0, 0>, %1 : <2, 255, 0, 0> -> <2, 65025, 0, 0>
  // Note: although the maximum value of BigInt<2, 65025, 0, 0> would fit into a BigInt<3, 255, 0, 0>,
  // the type inference system approximates to the next highest power of 2 (minus 1, so 65535 in this case).
  // a BigInt<2, 65535, 0, 0> would not fit into a BigInt<3, 255, 0, 0>, so we use BigInt<4, 255, 0, 0>
  // here instead, even though that results in an unused (always 0) coefficient for BigInt<2, 65025, 0, 0>.
  // See `getMaxPosBits` and its comments for more details.
  %4 = bigint.nondet_inv %2 : <8, 255, 0, 0>, %3 : <2, 65025, 0, 0> -> <4, 255, 0, 0>
  %5 = bigint.inv %2 : <8, 255, 0, 0>, %3 : <2, 65025, 0, 0> -> <4, 255, 0, 0>
  return
}

// -----

func.func @nondet_inv_ignore_negatives() {
  %0 = bigint.def 16, 0, true -> <2, 255, 0, 0>
  %1 = bigint.def 32, 0, true -> <4, 255, 0, 0>
  %2 = bigint.sub %0 : <2, 255, 0, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 255, 0>
  %3 = bigint.sub %2 : <4, 255, 255, 0>, %1 : <4, 255, 0, 0> -> <4, 255, 510, 0>
  %4 = bigint.nondet_inv %3 : <4, 255, 510, 0>, %0 : <2, 255, 0, 0> -> <2, 255, 0, 0>
  %5 = bigint.nondet_inv %0 : <2, 255, 0, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %6 = bigint.nondet_inv %2 : <4, 255, 255, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %7 = bigint.inv %3 : <4, 255, 510, 0>, %0 : <2, 255, 0, 0> -> <2, 255, 0, 0>
  %8 = bigint.inv %0 : <2, 255, 0, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  %9 = bigint.inv %2 : <4, 255, 255, 0>, %3 : <4, 255, 510, 0> -> <4, 255, 0, 0>
  return
}
