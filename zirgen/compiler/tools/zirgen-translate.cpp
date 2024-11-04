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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace zirgen;

enum Stage { exec, verify_mem, verify_bytes, compute_accum, verify_accum };

llvm::cl::opt<std::string>
    funcName("function",
             llvm::cl::desc("The name of the Zirgen MLIR function to generate code for"),
             llvm::cl::init(""));

llvm::cl::opt<Stage>
    stage("stage",
          llvm::cl::desc("The name of the stage to generate for the given function"),
          llvm::cl::values(clEnumVal(exec, "generate step_exec"),
                           clEnumVal(verify_mem, "generate step_verify_mem"),
                           clEnumVal(verify_bytes, "generate step_verify_bytes"),
                           clEnumVal(compute_accum, "generate step_compute_accum"),
                           clEnumVal(verify_accum, "generate step_verify_accum")));

int main(int argc, char** argv) {
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  zirgen::Zll::registerPasses();

  mlir::TranslateFromMLIRRegistration rustStepReg(
      "zirgen-to-rust-step",
      "generate rust_step_{stage}.cpp from Zirgen MLIR",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        if (!func) {
          llvm::errs() << "Function " << funcName << " not present.\n";
          return mlir::failure();
        }
        auto rust = zirgen::createRustStreamEmitter(output);
        switch (stage) {
        case exec:
          rust->emitStepFunc("exec", func);
          break;
        case verify_mem:
          rust->emitStepFunc("verify_mem", func);
          break;
        case verify_bytes:
          rust->emitStepFunc("verify_bytes", func);
          break;
        case compute_accum:
          rust->emitStepFunc("compute_accum", func);
          break;
        case verify_accum:
          rust->emitStepFunc("verify_accum", func);
          break;
        }
        return mlir::success();
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect>();
      });

  mlir::TranslateFromMLIRRegistration rustCodegen(
      "rust-codegen",
      "",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        codegen::CodegenOptions opts = codegen::getRustCodegenOpts();
        if (!funcName.empty()) {
          auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
          if (!func) {
            llvm::errs() << "Function " << funcName << " not present.\n";
            return mlir::failure();
          }
          return codegen::translateCodegen(func, opts, output);
        } else {
          return codegen::translateCodegen(module, opts, output);
        }
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect, ZStruct::ZStructDialect>();
      });

  mlir::TranslateFromMLIRRegistration cppCodegen(
      "cpp-codegen",
      "",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        codegen::CodegenOptions opts = codegen::getCppCodegenOpts();
        if (!funcName.empty()) {
          auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
          if (!func) {
            llvm::errs() << "Function " << funcName << " not present.\n";
            return mlir::failure();
          }
          return codegen::translateCodegen(func, opts, output);
        } else {
          return codegen::translateCodegen(module, opts, output);
        }
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect, ZStruct::ZStructDialect>();
      });

  mlir::TranslateFromMLIRRegistration rustPolyFpReg(
      "zirgen-to-rust-poly-fp",
      "generate rust_poly_fp.cpp from Zirgen MLIR",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        auto rust = zirgen::createRustStreamEmitter(output);
        rust->emitPolyFunc("poly_fp", func, /*idx=*/0, /*num split=*/1);
        return mlir::success();
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect>();
      });

  mlir::TranslateFromMLIRRegistration rustPolyExtReg(
      "zirgen-to-rust-poly-ext",
      "generate poly_ext.rs from Zirgen MLIR",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        auto rust = zirgen::createRustStreamEmitter(output);
        rust->emitPolyExtFunc(func);
        return mlir::success();
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect>();
      });

  mlir::TranslateFromMLIRRegistration rustTapsReg(
      "zirgen-to-rust-taps",
      "generate taps.rs from Zirgen MLIR",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        auto rust = zirgen::createRustStreamEmitter(output);
        rust->emitTaps(func);
        return mlir::success();
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect>();
      });

  mlir::TranslateFromMLIRRegistration rustInfoReg(
      "zirgen-to-rust-info",
      "generate info.rs from Zirgen MLIR",
      [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        auto rust = zirgen::createRustStreamEmitter(output);
        setModuleAttr(func, Zll::ProtocolInfoAttr::get(module.getContext(), "zirgen-translate"));
        rust->emitInfo(func);
        return mlir::success();
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::func::FuncDialect, Zll::ZllDialect>();
      });

  return failed(mlir::mlirTranslateMain(argc, argv, "Zirgen Translation Testing Tool"));
}
