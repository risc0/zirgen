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

// Analyses what buffers are used by a module.

#pragma once

#include "mlir/Pass/AnalysisManager.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace zirgen::ZStruct {

// This class provides utilities to get buffer and layout information
// for zirgen ZHLT modules.
class BufferAnalysis {
public:
  BufferAnalysis(mlir::ModuleOp op);

  // Returns the operation defining the layout that we would provide to
  // the given top-level block argument.
  std::pair<ZStruct::GlobalConstOp, Zll::BufferDescAttr>
  getLayoutAndBufferForArgument(mlir::BlockArgument layoutArg);
};

} // namespace zirgen::ZStruct
