// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/MLIRContext.h"
#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"

#include <gtest/gtest.h>

using namespace testing;
using namespace zirgen;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;

struct TypeTest : public Test {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::MLIRContext* ctx;

  TypeTest() {
    mlir::DialectRegistry registry;
    registry.insert<zirgen::Zll::ZllDialect>();
    registry.insert<zirgen::ZStruct::ZStructDialect>();
    registry.insert<zirgen::Zhlt::ZhltDialect>();

    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    ctx = context.get();
  }
};

// LCS(NonReg, Val) = Val
TEST_F(TypeTest, LCSRegVal) {
  EXPECT_EQ(getLeastCommonSuper({getNondetRegType(ctx), getValType(ctx)}), getValType(ctx));
}

// LCS(Val, Array<Val, 3>) = Component
TEST_F(TypeTest, LCSValArray) {
  EXPECT_EQ(getLeastCommonSuper({getValType(ctx), ArrayType::get(ctx, getValType(ctx), 3)}),
            getComponentType(ctx));
}

// Is NondetReg coercible to Component?
TEST_F(TypeTest, CoerceRegComponent) {
  EXPECT_TRUE(isCoercibleTo(getNondetRegType(ctx), getComponentType(ctx)));
}

// Is NondetReg coercible to Val?
TEST_F(TypeTest, CoerceNondetRegVal) {
  EXPECT_TRUE(isCoercibleTo(getNondetRegType(ctx), getValType(ctx)));
}
