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

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Main/Main.h"

namespace zirgen {

struct Template {
  std::string header = "";
  std::string footer = "";
};

class CodegenTarget {
public:
  CodegenTarget(zirgen::Zll::CircuitNameAttr circuitName) : circuitName(circuitName) {}

  // Returns the extension of a declaration file (e.g. `.h` or .`cuh`).
  // May return the same extension as getImplExtension if the target doesn't have a separate
  // declaration file (like `.rs`).
  virtual llvm::StringRef getDeclExtension() const = 0;

  // Returns the extension of an implementation file (e.g. `.rs` or `.cpp` or `.cu`).
  virtual llvm::StringRef getImplExtension() const = 0;

  virtual Template getDefsTemplate() const { return Template{}; }
  virtual Template getTypesTemplate() const { return Template{}; }
  virtual Template getLayoutDeclTemplate() const { return Template{}; }
  virtual Template getLayoutTemplate() const { return Template{}; }
  virtual Template getStepDeclTemplate() const { return Template{}; }
  virtual Template getStepTemplate() const { return Template{}; }

protected:
  // TODO: Remove this mutable when MLIR generates const getter methods on CircuitNameAttr.
  mutable zirgen::Zll::CircuitNameAttr circuitName;
};

struct CppCodegenTarget : public CodegenTarget {
  using CodegenTarget::CodegenTarget;

  llvm::StringRef getDeclExtension() const override;
  llvm::StringRef getImplExtension() const override;
  Template getStepDeclTemplate() const override;
  Template getStepTemplate() const override;
};

struct RustCodegenTarget : public CodegenTarget {
  using CodegenTarget::CodegenTarget;

  llvm::StringRef getDeclExtension() const override;
  llvm::StringRef getImplExtension() const override;
};

struct CudaCodegenTarget : public CodegenTarget {
  using CodegenTarget::CodegenTarget;

  llvm::StringRef getDeclExtension() const override;
  llvm::StringRef getImplExtension() const override;
  Template getStepDeclTemplate() const override;
  Template getStepTemplate() const override;
};

} // namespace zirgen
