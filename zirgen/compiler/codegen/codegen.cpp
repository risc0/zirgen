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

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;
namespace cl = llvm::cl;

namespace zirgen {

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
  if (opts.addExtraPolyPasses)
    opts.addExtraPolyPasses(pm);
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(Zll::createMakePolynomialPass());
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  opm.addPass(Zll::createComputeTapsPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

struct CodegenCLOptions {
  cl::list<std::string> outputFiles{
      cl::Positional, cl::OneOrMore, cl::desc("files in output directory")};
};

static llvm::ManagedStatic<CodegenCLOptions> clOptions;

llvm::StringRef getOutputDir() {
  std::string path;
  if (!clOptions.isConstructed()) {
    throw(std::runtime_error("codegen command line options must be registered"));
  }

  if (clOptions->outputFiles.empty()) {
    llvm::errs() << "At least one file in the output directory must be specified\n";
    exit(1);
  }

  llvm::StringRef outputDir = llvm::StringRef(clOptions->outputFiles[0]).rsplit('/').first;
  if (outputDir.empty())
    return ".";
  return outputDir;
}

} // namespace

class FileEmitter {
public:
  FileEmitter(StringRef path, const EmitCodeOptions& emitCodeOpts)
      : path(path), emitCodeOpts(emitCodeOpts) {}

  void emitRustStep(const std::string& stage, func::FuncOp func) {
    auto ofs = openOutputFile("rust_step_" + stage + ".cpp");
    createRustStreamEmitter(*ofs)->emitStepFunc(stage, func);
  }

  void emitGpuStep(const std::string& stage, const std::string& suffix, func::FuncOp func) {
    auto ofs = openOutputFile("step_" + stage + suffix);
    createGpuStreamEmitter(*ofs, suffix)->emitStepFunc(stage, func);
  }

  void emitPolyFunc(const std::string& fn, func::FuncOp func) {
    auto ofs = openOutputFile("rust_" + fn + ".cpp");
    createRustStreamEmitter(*ofs)->emitPolyFunc(fn, func);
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

  void emitInfo(func::FuncOp func, ProtocolInfo info) {
    auto ofs = openOutputFile("info.rs");
    createRustStreamEmitter(*ofs)->emitInfo(func, info);
  }

  void emitHeader(func::FuncOp func) {
    auto ofs = openOutputFile("impl.h");
    createCppStreamEmitter(*ofs)->emitHeader(func);
  }

  void emitEvalCheck(const std::string& suffix, func::FuncOp func) {
    auto ofs = openOutputFile("eval_check" + suffix);
    createGpuStreamEmitter(*ofs, suffix)->emitPoly(func);
  }

  void emitAllLayouts(mlir::ModuleOp op) {
    static codegen::RustLanguageSyntax kRust;
    emitLayout(op, &kRust, ".rs.inc");

    static codegen::CppLanguageSyntax kCpp;
    emitLayout(op, &kCpp, ".cpp.inc");

    static codegen::CudaLanguageSyntax kCuda;
    emitLayout(op, &kCuda, ".cu.inc");
  }

  void emitLayout(mlir::ModuleOp op, codegen::LanguageSyntax* lang, StringRef suffix) {
    codegen::CodegenOptions opts;
    opts.lang = lang;
    opts.zkpLayoutCompat = emitCodeOpts.zkpLayoutCompat;
    auto ofs = openOutputFile(("layout" + suffix).str());
    codegen::CodegenEmitter emitter(opts, ofs.get(), op->getContext());
    op.walk([&](ZStruct::GlobalConstOp constOp) {
      emitter.emitTypeDefs(constOp);
      constOp.emitGlobal(emitter);
    });
  }

private:
  std::string path;
  const EmitCodeOptions& emitCodeOpts;

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
  *clOptions;
}

void emitCode(ModuleOp module, const EmitCodeOptions& opts) {
  FileEmitter emitter(getOutputDir(), opts);
  optimizeSimple(module);
  emitter.emitAllLayouts(module);

  auto circuitDef = Zll::CircuitDefOp::lookupInModule(module);

  llvm::StringSet<> seenStages;

  for (auto stage : llvm::enumerate(circuitDef.getSteps())) {
    auto step = llvm::cast<Zll::StepDefAttr>(stage.value());
    auto stageOpts = opts.stages.lookup(step.getName());
    seenStages.insert(step.getName());

    auto moduleCopy = dyn_cast<ModuleOp>(module->clone());

    if (opts.splitUsingBarrier)
      optimizeSplit(moduleCopy, stage.index(), stageOpts);
    auto func = moduleCopy.lookupSymbol<func::FuncOp>(step.getStepFunc());
    if (!func) {
      llvm::errs() << "Unable to find step function " << step.getStepFunc() << "\n";
      exit(1);
    }
    std::string outputFile;
    if (stageOpts.outputFile.empty())
      outputFile = step.getName();
    else
      outputFile = stageOpts.outputFile;
    emitter.emitRustStep(outputFile, func);
    if (outputFile == "compute_accum" || outputFile == "verify_accum") {
      emitter.emitGpuStep(outputFile, ".metal", func);
    }
    emitter.emitGpuStep(outputFile, ".cu", func);
  }

  for (auto k : opts.stages.keys()) {
    if (!seenStages.contains(k)) {
      llvm::errs() << "Options specified for stage " << k << " but no stage " << k << " seen\n";
      exit(1);
    }
  }

  optimizePoly(module, opts);
  ProtocolInfo protocolInfo;
  strncpy(protocolInfo.data(), circuitDef.getCircuitInfo().data(), PROTOCOL_INFO_LEN);
  protocolInfo[PROTOCOL_INFO_LEN] = '\0';
  module.walk([&](func::FuncOp func) {
    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func, protocolInfo);
    emitter.emitEvalCheck(".cu", func);
    emitter.emitEvalCheck(".metal", func);
    emitter.emitPolyEdslFunc(func);
    emitter.emitHeader(func);
    emitter.emitTapsCpp(func);
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
