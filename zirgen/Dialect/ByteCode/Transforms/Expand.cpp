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

#define GEN_PASS_DEF_EXPANDBC
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

namespace {

struct Expander {
  Expander(ExecutorOp executeOp, EncodedAttr encoded, Region& oldBody, Region& newBody)
      : executeOp(executeOp), encoded(encoded), newBody(newBody), builder(&newBody) {
    for (auto arg : oldBody.getArguments()) {
      newBody.addArgument(arg.getType(), arg.getLoc());
    }
    mapper.map(oldBody.getArguments(), newBody.getArguments());
    builder.createBlock(&newBody);

    byteCode = encoded.getEncoded();

    for (auto intName : encoded.getIntNames()) {
      intNameBits[intName.getName()] = intName.getEncodedBits();
    }
  }

  size_t decodeInt(StringRef intName) {
    assert(intNameBits.contains(intName));
    size_t bits = intNameBits[intName];
    assert((bits % 8) == 0);
    size_t result = 0;
    for (size_t i = 0; i != bits / 8; ++i) {
      result += uint8_t(byteCode.front()) << (i * 8);
      byteCode = byteCode.drop_front(1);
    }
    return result;
  }

  void expand() {
    while (!byteCode.empty()) {
      expandOne();
    }
  }

  void expandOne() {
    size_t keyIndex = byteCode.decodeInt("DispatchKey");
    auto& arm = executeOp.getArm(key);
    SmallVector<size_t> decodedInts;
    for (Operation* op : llvm::make_pointer_range(arm.front())) {
      if (auto decodeOp = dyn_cast<DecodeOp>(op)) {
        decodedInts.push_back(decodeInt(decodeOp.getIntName()));
        continue;
      }
      if (auto yieldOp = dyn_cast<YieldOp>(op)) {
      }
    }
  }

  ExecutorOp executeOp;
  EncodedAttr byteCode;
  Region& newBody;
  OpBuilder builder;
  IRMapping mapper;

  DenseMap<Type, DenseMap</*index=*/size_t, Value>> values;
  llvm::StringMap<size_t> intNameBits;
} // namespace

struct ExpandBCPass : public impl::ExpandBCBase<ExpandBCPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    GetEncodedOp getEncodedOp = funcOp.getBody().getOps<GetEncodedOp>().front();
    ExecutorOp executeOp = funcOp.getBody().getOps<GetEncodedOp>().front();
    DefineEncodedOp defineOp = SymbolTable::lookupNearestSymbolFrom<DefineEncodedOp>(
        getEncodedOp, getEncodedOp.getTarget());
    if (!defineOp) {
      getEncodedOp.emitError("Unable to find byte code definition");
      return signalPassFailure();
    }

    EncodedAttr byteCode = defineOp.getEncoded();

    Region newBody(funcOp);
    Expander expander(executeOp, byteCode, func.getBody(), newBody);
    expander.expand();
    func.getBody().takeBody(newBody)
  }
};

} // namespace zirgen::ByteCode
