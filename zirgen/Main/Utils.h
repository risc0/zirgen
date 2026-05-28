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

// Emission helpers called by gen_zirgen.cpp after the main pass pipeline.
// emitPoly handles the validity polynomial outputs (eval_check, poly_ext, taps,
// info); emitTarget handles per-language (Rust/C++/CUDA) defs, types, layout,
// and step-function outputs.

#include "zirgen/Main/Target.h"

namespace zirgen {

std::unique_ptr<llvm::raw_ostream> openOutput(llvm::StringRef filename);

// Runs the MakePolynomial + taps passes and writes all validity polynomial outputs
// (validity.ir, poly_ext.rs, taps.rs, eval_check_*.cu, rust_poly_fp_*.cpp, etc.)
// to the directory specified by --output-dir.
void emitPoly(mlir::ModuleOp mod, mlir::StringRef circuitName, llvm::StringRef protocolInfo);

// Emits defs, types, layout (decl + impl), and step functions for the given
// CodegenTarget (Rust, C++, or CUDA). stepFuncs is the module produced by
// makeStepFuncs(); stepSplitCount controls how many output files step functions
// are split across.
void emitTarget(const zirgen::CodegenTarget& target,
                mlir::ModuleOp mod,
                mlir::ModuleOp stepFuncs,
                const zirgen::codegen::CodegenOptions& opts,
                unsigned stepSplitCount = 1);

} // namespace zirgen
