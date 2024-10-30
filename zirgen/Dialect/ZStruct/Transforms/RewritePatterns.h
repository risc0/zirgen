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

// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/TypeSwitch.h"

// #include "zirgen/Dialect/ZStruct/IR/Types.h"
// #include "zirgen/Dialect/ZStruct/IR/ZStruct.h"

#include "mlir/IR/PatternMatch.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"

namespace zirgen::ZStruct {

struct UnrollMaps : public mlir::OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(MapOp op, mlir::PatternRewriter& rewriter) const;
};

struct UnrollReduces : public mlir::OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ReduceOp op, mlir::PatternRewriter& rewriter) const;
};

inline void getUnrollPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
  patterns.insert<UnrollMaps>(ctx);
  patterns.insert<UnrollReduces>(ctx);
}

// Convert switch statements to if statements.
struct SplitSwitchArms : public mlir::OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(SwitchOp op, mlir::PatternRewriter& rewriter) const;
};

} // namespace zirgen::ZStruct
