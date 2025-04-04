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

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"

namespace zirgen::Zhlt {

using namespace mlir;

// Given a ComponentOp for an entry point to the circuit, generate operations at
// the builder's current insertion point to bind the appropriate layout symbols
// and pass them into a call to an aspect of that entry point.
template <typename ConstructLikeOp>
LogicalResult bindLayoutsForEntryPoint(ComponentOp entryPoint,
                        OpBuilder& builder,
                        ZStruct::BufferAnalysis& bufferAnalysis) {
  using ComponentLikeOp = typename ConstructLikeOp::FuncOpTy;
  Location loc = entryPoint.getLoc();
  auto callee = entryPoint.getAspect<ComponentLikeOp>();
  if (!callee) {
    entryPoint.emitError() << "Unable to find " << ComponentLikeOp::getOperationName() << " for entry point " << entryPoint.getName();
    return failure();
  }

  llvm::SmallVector<Value> args;
  for (auto execArg : callee.getBody().front().getArguments()) {
    auto [constOp, bufferDesc] = bufferAnalysis.getLayoutAndBufferForArgument(execArg);
    if (!constOp) {
      callee.emitError() << "Unable to find a value for argument " << execArg
                          << " to top-level step for component " << entryPoint.getName();
      return failure();
    }
    auto getBufferOp = builder.create<ZStruct::GetBufferOp>(
        loc, bufferDesc.getType(), bufferDesc.getName());
    args.push_back(
        builder.create<ZStruct::BindLayoutOp>(loc,
                                              constOp.getType(),
                                              SymbolRefAttr::get(constOp.getSymNameAttr()),
                                              getBufferOp));
  }

  mlir::FunctionType funcType = callee.getFunctionType();
  builder.create<ConstructLikeOp>(loc,
                                  builder.getAttr<FlatSymbolRefAttr>(callee.getSymName()),
                                  funcType,
                                  args,
                                  callee.getInputSegmentSizes(),
                                  callee.getResultSegmentSizes());
  return success();
}

} // namespace zirgen::Zhlt {
