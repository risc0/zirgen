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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "zirgen/Dialect/BigInt/Bytecode/decode.h"
#include "zirgen/Dialect/BigInt/Bytecode/encode.h"
#include "zirgen/Dialect/BigInt/Bytecode/file.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/circuit/bigint/op_tests.h"
#include "zirgen/circuit/bigint/rsa.h"
#include "llvm/ADT/APInt.h"

#include <gtest/gtest.h>

using namespace zirgen;

using ZType = std::array<uint32_t, 4>;

struct BibcTest : public testing::Test {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::MLIRContext* ctx;
  mlir::ModuleOp module;

  BibcTest() {
    mlir::DialectRegistry registry;
    registry.insert<BigInt::BigIntDialect>();
    registry.insert<mlir::func::FuncDialect>();

    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    ctx = context.get();

    auto loc = mlir::UnknownLoc::get(ctx);
    module = mlir::ModuleOp::create(loc);
  }

  mlir::func::FuncOp makeFunc(std::string name, mlir::OpBuilder& builder) {
    auto loc = mlir::UnknownLoc::get(ctx);
    builder.setInsertionPointToEnd(&module.getBodyRegion().front());
    auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto out = builder.create<mlir::func::FuncOp>(loc, name, funcType);
    builder.setInsertionPointToEnd(out.addEntryBlock());
    builder.create<mlir::func::ReturnOp>(loc);
    builder.setInsertionPointToStart(builder.getInsertionBlock());
    return out;
  }

  mlir::func::FuncOp recycle(mlir::func::FuncOp inFunc) {
    // Encode this function into BIBC structure
    auto prog = BigInt::Bytecode::encode(inFunc);
    // Write it out into a buffer
    size_t bytes = BigInt::Bytecode::tell(*prog);
    auto buf = std::make_unique<uint8_t[]>(bytes);
    BigInt::Bytecode::write(*prog, buf.get(), bytes);
    // Drop the old bytecode structure and create a fresh one
    prog.reset(new BigInt::Bytecode::Program);
    // Read the contents of the buffer back in
    BigInt::Bytecode::read(*prog, buf.get(), bytes);
    // Decode the bytecode back into MLIR operations
    return BigInt::Bytecode::decode(module, *prog);
  }

  void lower() {
    // Lower the inverse and reduce ops to simpler, executable ops
    mlir::PassManager pm(ctx);
    pm.enableVerifier(true);
    pm.addPass(zirgen::BigInt::createLowerReducePass());
    pm.addPass(zirgen::BigInt::createLowerInvPass());
    if (failed(pm.run(module))) {
      llvm::errs() << "an internal validation error occurred:\n";
      module.print(llvm::errs());
      std::exit(1);
    }
  }

  void AB(mlir::func::FuncOp func, llvm::ArrayRef<llvm::APInt> inputs, ZType& A, ZType& B) {
    A = BigInt::eval(func, inputs).z;
    func = recycle(func);
    B = BigInt::eval(func, inputs).z;
  }
};

std::vector<llvm::APInt> apints(std::vector<std::string> args) {
  std::vector<llvm::APInt> out;
  out.resize(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    // each hex digit represents one nibble, 4 bits
    unsigned bits = args[i].size() * 4;
    out[i] = llvm::APInt(bits, args[i], 16);
  }
  return out;
}

TEST_F(BibcTest, Add8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("add_8", builder);
  BigInt::makeAddTest(builder, func.getLoc(), 8);

  auto inputs = apints({"1", "2", "3"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Add16) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("add_16", builder);
  BigInt::makeAddTest(builder, func.getLoc(), 16);

  auto inputs = apints({"1", "2", "3"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Add128) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("add_128", builder);
  BigInt::makeAddTest(builder, func.getLoc(), 128);

  auto inputs = apints({"1", "2", "3"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Mul8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("mul_8", builder);
  BigInt::makeMulTest(builder, func.getLoc(), 8);

  auto inputs = apints({"5", "7", "23"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Mul16) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("mul_16", builder);
  BigInt::makeMulTest(builder, func.getLoc(), 16);

  auto inputs = apints({"5", "7", "23"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Mul128) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("mul_128", builder);
  BigInt::makeMulTest(builder, func.getLoc(), 128);

  auto inputs = apints({"5", "7", "23"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, RSA256) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("rsa_256", builder);
  BigInt::makeRSA(builder, func.getLoc(), 256);
  lower();

  llvm::APInt N(64, 101);
  llvm::APInt S(64, 32766);
  auto M = BigInt::RSA(N, S);
  std::vector<llvm::APInt> inputs = {N, S, M};

  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, RSA3072) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("rsa_3072", builder);
  BigInt::makeRSA(builder, func.getLoc(), 3072);
  lower();

  llvm::APInt N(64, 22764235167642101);
  llvm::APInt S(64, 10116847215);
  auto M = BigInt::RSA(N, S);
  std::vector<llvm::APInt> inputs = {N, S, M};

  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
