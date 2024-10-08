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

#pragma once

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Zhlt {

class EmitZhlt {
public:
  EmitZhlt(mlir::ModuleOp module, codegen::CodegenEmitter& cg)
      : module(module), cg(cg), ctx(module.getContext()) {}
  virtual ~EmitZhlt() = default;

  mlir::LogicalResult emitDefs() {
    if (failed(doValType()))
      return mlir::failure();

    if (failed(doBuffers()))
      return mlir::failure();

    return mlir::success();
  }

protected:
  virtual mlir::LogicalResult emitBufferList(llvm::ArrayRef<Zll::BufferDescAttr> bufs) {
    return mlir::success();
  }

  mlir::ModuleOp module;
  codegen::CodegenEmitter& cg;
  mlir::MLIRContext* ctx;

private:
  // Declares "Val" to be a type alias to the appropriete field element.
  mlir::LogicalResult doValType();

  // Provide buffers and sizes
  mlir::LogicalResult doBuffers();
};

std::unique_ptr<EmitZhlt> getEmitter(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg);

// Generates code for a ZHLT module, including extern traits, type definitions, etc.
mlir::LogicalResult emitModule(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg);

void addCppSyntax(codegen::CodegenOptions& opts);
void addRustSyntax(codegen::CodegenOptions& opts);

} // namespace zirgen::Zhlt
