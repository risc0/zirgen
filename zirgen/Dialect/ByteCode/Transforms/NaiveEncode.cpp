// Copyright 2025 RISC Zero, Inc.
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
#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Bufferize.h"
#include "zirgen/Dialect/ByteCode/Transforms/Executor.h"

using namespace mlir;

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_GENEXECUTOR
#define GEN_PASS_DEF_PRINTNAIVEBC
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

struct GenExecutorPass : public impl::GenExecutorBase<GenExecutorPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    funcOp.insertArgument(funcOp.getArguments().size(),
                          EncodedType::get(&getContext()),
                          /*argAttrs=*/DictionaryAttr(),
                          funcOp.getLoc());

    Block* block = &funcOp.getBody().front();
    NaiveBufferize bufferize;
    ExecuteOp execOp = buildExecutor(
        funcOp.getLoc(), &funcOp.getBody(), /*encoded=*/block->getArguments().back(), bufferize);

    Operation* terminator = nullptr;
    if (block->mightHaveTerminator()) {
      terminator = block->getTerminator();
      terminator->remove();
    }
    block->clear();

    OpBuilder builder = OpBuilder::atBlockBegin(block);
    builder.insert(execOp);
    if (terminator)
      builder.insert(terminator);
  }
};

struct PrintNaiveBCPass : public impl::PrintNaiveBCBase<PrintNaiveBCPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    OpBuilder builder(&getContext());
    GetEncodedOp getOp =
        builder.create<GetEncodedOp>(funcOp.getLoc(), builder.getStringAttr("unused"));
    NaiveBufferize bufferize;
    ExecuteOp execOp =
        buildExecutor(funcOp.getLoc(), &funcOp.getBody(), /*encoded=*/getOp, bufferize);

    EncodeOptions encodeOpts;
    encodeOpts.outputText = true;
    EncodedAttr encoded = encodeByteCode(&funcOp.getBody(), execOp, bufferize, encodeOpts);
    if (!encoded) {
      llvm::errs() << "Unable to encode byte code\n";
      signalPassFailure();
      return;
    }
    llvm::outs() << "START " << funcOp.getSymName() << "\n";
    for (auto intInfo : execOp.getIntKinds().getAsRange<IntKindInfoAttr>())
      llvm::outs() << "INT-KIND " << getNameForIntKind(intInfo.getIntKind()) << " bits "
                   << intInfo.getEncodedBits() << "\n";
    for (auto tempBuf : encoded.getTempBufs())
      llvm::outs() << "BUF " << getNameForIntKind(tempBuf.getIntKind()) << " size "
                   << tempBuf.getSize() << "\n";
    llvm::outs() << encoded.getEncoded();
    llvm::outs() << "END " << funcOp.getSymName() << "\n";
    execOp->erase();
    getOp->erase();
  }
};

} // namespace zirgen::ByteCode
