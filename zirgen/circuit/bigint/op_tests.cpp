// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/bigint/op_tests.h"

namespace zirgen::BigInt {

void makeConstZeroTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto inp = builder.create<BigInt::DefOp>(
      loc, bits, 0, true); // Ignored, but not allowed to have no-input BigInt op

  mlir::Type zeroType = builder.getIntegerType(8, false); // unsigned 8 bit
  auto zeroAttr = builder.getIntegerAttr(zeroType, 0);    // value 0
  auto zero = builder.create<BigInt::ConstOp>(loc, zeroAttr);

  builder.create<BigInt::EqualZeroOp>(loc, zero);
}

void makeConstOneTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 0, true);

  mlir::Type oneType = builder.getIntegerType(8, false); // unsigned 8 bit
  auto oneAttr = builder.getIntegerAttr(oneType, 1);     // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  auto diff = builder.create<BigInt::SubOp>(loc, one, expected);

  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeConstTwoByteTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 0, true);

  mlir::Type twobyteType = builder.getIntegerType(16, false);     // unsigned 16 bit
  auto twobyteAttr = builder.getIntegerAttr(twobyteType, 0x1234); // value 0x1234
  auto twobyte = builder.create<BigInt::ConstOp>(loc, twobyteAttr);

  auto diff = builder.create<BigInt::SubOp>(loc, twobyte, expected);

  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeConstAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  size_t const_bits = 16;
  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto expected = builder.create<BigInt::DefOp>(loc, std::max(bits, const_bits) + 1, 1, true);

  // Construct constant
  mlir::Type fortysevensType = builder.getIntegerType(const_bits);
  auto fortysevensAttr = builder.getIntegerAttr(fortysevensType, 0x4747); // value 0x4747
  auto fortysevens = builder.create<BigInt::ConstOp>(loc, fortysevensAttr);

  auto result = builder.create<BigInt::AddOp>(loc, inp, fortysevens);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeConstMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  size_t const_bits = 16;

  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto expected = builder.create<BigInt::DefOp>(loc, bits + const_bits, 1, true);

  // Construct constant
  mlir::Type theconstType = builder.getIntegerType(const_bits);
  auto theconstAttr = builder.getIntegerAttr(theconstType, 0x5432); // value 0x5432
  auto theconst = builder.create<BigInt::ConstOp>(loc, theconstAttr);

  auto result = builder.create<BigInt::MulOp>(loc, theconst, inp);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeConstAddAltTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  size_t const_bits = 16;

  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto expected = builder.create<BigInt::DefOp>(loc, std::max(bits, const_bits) + 1, 1, true);

  // Construct constant
  mlir::Type theconstType = builder.getIntegerType(const_bits);
  auto theconstAttr = builder.getIntegerAttr(theconstType, 0x5432); // value 0x5432
  auto theconst = builder.create<BigInt::ConstOp>(loc, theconstAttr);

  auto result = builder.create<BigInt::AddOp>(loc, theconst, inp);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto lhs = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto rhs = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto expected = builder.create<BigInt::DefOp>(loc, bits + 1, 2, true);

  auto result = builder.create<BigInt::AddOp>(loc, lhs, rhs);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeSubTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto lhs = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto rhs = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  auto result = builder.create<BigInt::SubOp>(loc, lhs, rhs);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto lhs = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto rhs = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto expected = builder.create<BigInt::DefOp>(loc, 2 * bits, 2, true);

  auto result = builder.create<BigInt::MulOp>(loc, lhs, rhs);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

void makeReduceTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto lhs = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto rhs = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  auto result = builder.create<BigInt::ReduceOp>(loc, lhs, rhs);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

} // namespace zirgen::BigInt
