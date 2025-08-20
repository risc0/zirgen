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

#pragma once

#include "zirgen/Main/Target.h"

namespace zirgen {

std::unique_ptr<llvm::raw_ostream> openOutput(llvm::StringRef filename);

void emitPoly(mlir::ModuleOp mod, mlir::StringRef circuitName, llvm::StringRef protocolInfo);

void emitTarget(const zirgen::CodegenTarget& target,
                mlir::ModuleOp mod,
                mlir::ModuleOp stepFuncs,
                const zirgen::codegen::CodegenOptions& opts,
                unsigned stepSplitCount = 1);

} // namespace zirgen
