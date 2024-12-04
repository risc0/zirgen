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

#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;
namespace cl = llvm::cl;

namespace zirgen {
namespace codegen {

namespace {

void addCommonSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<StringAttr>(
      [](CodegenEmitter& cg, auto strAttr) { cg.emitEscapedString(strAttr); });
  opts.addLiteralSyntax<IntegerAttr>(
      [](CodegenEmitter& cg, auto intAttr) { cg << intAttr.getValue().getZExtValue(); });
}

void addCppSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << "Val(" << elems[0] << ")";
    } else if (elems.size() == 4) {
      cg << "ExtVal(";
      cg.interleaveComma(elems);
      cg << ")";
    }
  });
}

void addRustSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << "Val::new(" << elems[0] << ")";
    } else {
      cg << "ExtVal::new(";
      cg.interleaveComma(elems, [&](auto elem) { cg << "Val::new(" << elem << ")"; });
      cg << ")";
    }
  });
}

} // namespace

CodegenOptions getRustCodegenOpts() {
  static codegen::RustLanguageSyntax kRust;
  codegen::CodegenOptions opts(&kRust);
  addCommonSyntax(opts);
  addRustSyntax(opts);
  ZStruct::addRustSyntax(opts);
  Zhlt::addRustSyntax(opts);
  return opts;
}

CodegenOptions getCppCodegenOpts() {
  static codegen::CppLanguageSyntax kCpp;
  codegen::CodegenOptions opts(&kCpp);
  addCommonSyntax(opts);
  addCppSyntax(opts);
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}

CodegenOptions getCudaCodegenOpts() {
  static codegen::CudaLanguageSyntax kCuda;
  codegen::CodegenOptions opts(&kCuda);
  addCommonSyntax(opts);
  addCppSyntax(opts);
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}

} // namespace codegen

namespace {

void optimizeSimple(ModuleOp module) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

void optimizeSplit(ModuleOp module, unsigned stage, const StageOptions& opts) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(Zll::createSplitStagePass(stage));
  if (opts.addExtraPasses) {
    opts.addExtraPasses(opm);
  }
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

void optimizePoly(ModuleOp module, const EmitCodeOptions& opts) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(Zll::createMakePolynomialPass());
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  opm.addPass(Zll::createComputeTapsPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

llvm::StringRef getOutputDir() {
  if (!codegenCLOptions.isConstructed()) {
    throw(std::runtime_error("codegen command line options must be registered"));
  }

  return codegenCLOptions->outputDir;
}

} // namespace

llvm::ManagedStatic<CodegenCLOptions> codegenCLOptions;

class FileEmitter {
public:
  FileEmitter(StringRef path) : path(path) {}

  void emitIR(const std::string& fn, Operation* op) {
    auto ofs = openOutputFile(fn + ".ir");
    op->print(*ofs.get());
  }

  void emitRustStep(const std::string& stage, func::FuncOp func) {
    auto ofs = openOutputFile("rust_step_" + stage + ".cpp");
    createRustStreamEmitter(*ofs)->emitStepFunc(stage, func);
  }

  void emitGpuStep(const std::string& stage, const std::string& suffix, func::FuncOp func) {
    auto ofs = openOutputFile("step_" + stage + suffix);
    createGpuStreamEmitter(*ofs, suffix)->emitStepFunc(stage, func);
  }

  void emitPolyFunc(const std::string& fn, func::FuncOp func) {
    if (codegenCLOptions->validitySplitCount > 1) {
      for (size_t i : llvm::seq(size_t(codegenCLOptions->validitySplitCount))) {
        auto ofs = openOutputFile("rust_" + fn + "_" + std::to_string(i) + ".cpp");
        createRustStreamEmitter(*ofs)->emitPolyFunc(
            fn, func, i, size_t(codegenCLOptions->validitySplitCount));
      }
    } else {
      auto ofs = openOutputFile("rust_" + fn + ".cpp");
      createRustStreamEmitter(*ofs)->emitPolyFunc(fn, func, /*split part=*/0, /*num splits=*/1);
    }
  }

  void emitPolyEdslFunc(func::FuncOp func) {
    auto ofs = openOutputFile("poly_edsl.cpp");
    createCppStreamEmitter(*ofs)->emitPoly(func);
  }

  void emitPolyExtFunc(func::FuncOp func) {
    auto ofs = openOutputFile("poly_ext.rs");
    createRustStreamEmitter(*ofs)->emitPolyExtFunc(func);
  }

  void emitTapsCpp(func::FuncOp func) {
    auto ofs = openOutputFile("taps.cpp");
    createCppStreamEmitter(*ofs)->emitTaps(func);
  }

  void emitTaps(func::FuncOp func) {
    auto ofs = openOutputFile("taps.rs");
    createRustStreamEmitter(*ofs)->emitTaps(func);
  }

  void emitInfo(func::FuncOp func) {
    auto ofs = openOutputFile("info.rs");
    createRustStreamEmitter(*ofs)->emitInfo(func);
  }

  void emitHeader(func::FuncOp func) {
    auto ofs = openOutputFile("impl.h");
    createCppStreamEmitter(*ofs)->emitHeader(func);
  }

  void
  emitEvalCheck(const std::string& suffix, const std::string& headerSuffix, func::FuncOp func) {
    if (codegenCLOptions->validitySplitCount > 1) {
      for (size_t i : llvm::seq(size_t(codegenCLOptions->validitySplitCount))) {
        auto ofs = openOutputFile("eval_check_" + std::to_string(i) + suffix);
        createGpuStreamEmitter(*ofs, suffix)
            ->emitPoly(func, i, size_t(codegenCLOptions->validitySplitCount));
      }
    } else {
      auto ofs = openOutputFile("eval_check" + suffix);
      createGpuStreamEmitter(*ofs, suffix)->emitPoly(func, /*split part=*/0, /*num splits=*/1);
    }

    auto ofs = openOutputFile("eval_check" + headerSuffix);
    createGpuStreamEmitter(*ofs, suffix)->emitPoly(func, 0, 0, /*declsOnly=*/true);
  }

  void emitAllLayouts(mlir::ModuleOp op) {
    emitLayout(op, codegen::getRustCodegenOpts(), ".rs.inc");
    emitLayout(op, codegen::getCppCodegenOpts(), ".cpp.inc");
    emitLayout(op, codegen::getCudaCodegenOpts(), ".cu.inc");
  }

  void emitLayout(mlir::ModuleOp op, const codegen::CodegenOptions& opts, StringRef suffix) {
    auto ofs = openOutputFile(("layout" + suffix).str());
    codegen::CodegenEmitter emitter(opts, ofs.get(), op->getContext());
    op.walk([&](ZStruct::GlobalConstOp constOp) {
      emitter.emitTypeDefs(constOp);
      constOp.emitGlobal(emitter);
    });
  }

private:
  std::string path;

  std::unique_ptr<llvm::raw_ostream> openOutputFile(const std::string& name) {
    std::string filename = path + "/" + name;
    std::error_code ec;
    auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
    if (ec) {
      throw std::runtime_error("Unable to open file: " + filename);
    }
    return ofs;
  }
};

void registerCodegenCLOptions() {
  *codegenCLOptions;
}

void emitCode(ModuleOp module, const EmitCodeOptions& opts) {
  FileEmitter emitter(getOutputDir());
  optimizeSimple(module);
  emitter.emitAllLayouts(module);

  auto stepsAttr = Zll::lookupModuleAttr<Zll::StepsAttr>(module);

  llvm::StringSet seenStages;

  for (auto [stageIndex, stage] : llvm::enumerate(stepsAttr.getSteps())) {
    auto stageOpts = opts.stages.lookup(stage);
    seenStages.insert(stage.strref());
    auto stageName = stage.str();

    auto moduleCopy = dyn_cast<ModuleOp>(module->clone());
    optimizeSplit(moduleCopy, stageIndex, stageOpts);
    moduleCopy.walk([&](func::FuncOp func) {
      std::string outputFile;
      if (stageOpts.outputFile.empty())
        outputFile = stageName;
      else
        outputFile = stageOpts.outputFile;

      emitter.emitRustStep(outputFile, func);
      if (stageName == "compute_accum" || stageName == "verify_accum") {
        emitter.emitGpuStep(outputFile, ".metal", func);
      }
      emitter.emitGpuStep(outputFile, ".cu", func);
    });
  }

  for (auto k : opts.stages.keys()) {
    if (!seenStages.contains(k)) {
      llvm::errs() << "Options specified for stage " << k << " but no stage " << k << " seen\n";
      exit(1);
    }
  }

  optimizePoly(module, opts);
  module.walk([&](func::FuncOp func) {
    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func);
    emitter.emitEvalCheck(".cu", ".cuh", func);
    emitter.emitEvalCheck(".metal", ".h", func);
    emitter.emitPolyEdslFunc(func);
    emitter.emitHeader(func);
    emitter.emitTapsCpp(func);
  });
}

void emitCodeZirgenPoly(ModuleOp module, StringRef outputDir) {
  FileEmitter emitter(outputDir);

  // Inline everything, since everything else expects there to be a single function left.
  PassManager pm(module.getContext());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }

  // Save as IR so we can generate predicates to verify the validity polynomial.
  emitter.emitIR("validity", module);

  module.walk([&](func::FuncOp func) {
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func);
    emitter.emitTapsCpp(func);
  });

  // Split up functions for poly_fp
  pm.clear();
  pm.addPass(Zll::createBalancedSplitPass(/*maxOps=*/1000));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to balanced split");
  }

  module.walk([&](func::FuncOp func) {
    if (SymbolTable::getSymbolVisibility(func) == SymbolTable::Visibility::Private)
      return;

    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitEvalCheck(".cu", ".cuh", func);
  });
}

std::string escapeString(llvm::StringRef str) {
  std::string out = "\"";
  for (size_t i = 0; i < str.size(); i++) {
    unsigned char c = str[i];
    if (' ' <= c and c <= '~' and c != '\\' and c != '"') {
      out.push_back(c);
    } else {
      out.push_back('\\');
      switch (c) {
      case '"':
        out.push_back('"');
        break;
      case '\\':
        out.push_back('\\');
        break;
      case '\t':
        out.push_back('t');
        break;
      case '\r':
        out.push_back('r');
        break;
      case '\n':
        out.push_back('n');
        break;
      default:
        char const* const hexdig = "0123456789ABCDEF";
        out.push_back('x');
        out.push_back(hexdig[c >> 4]);
        out.push_back(hexdig[c & 0xF]);
      }
    }
  }
  out.push_back('"');
  return out;
}

} // namespace zirgen
