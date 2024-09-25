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
  opts.addLiteralHandler<IntegerAttr>(
      [](CodegenEmitter& cg, auto intAttr) { cg << intAttr.getValue().getZExtValue(); });
  opts.addLiteralHandler<StringAttr>(
      [](CodegenEmitter& cg, auto strAttr) { cg.emitEscapedString(strAttr); });
}

void addCppSyntax(CodegenOptions& opts) {
  opts.addLiteralHandler<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << "Val(" << elems[0] << ")";
    } else {
      cg << "Val" << elems.size() << "{";
      cg.interleaveComma(elems);
      cg << "}";
    }
  });
}

void addRustSyntax(CodegenOptions& opts) {
  opts.addLiteralHandler<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
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

void optimizePoly(ModuleOp module) {
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

struct CodegenCLOptions {
  cl::opt<std::string> outputDir{
      "output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required};
};

static llvm::ManagedStatic<CodegenCLOptions> clOptions;

llvm::StringRef getOutputDir() {
  if (!clOptions.isConstructed()) {
    throw(std::runtime_error("codegen command line options must be registered"));
  }

  return clOptions->outputDir;
}

} // namespace

class FileEmitter {
public:
  FileEmitter(StringRef path) : path(path) {}

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
  *clOptions;
}

void emitCode(ModuleOp module, const EmitCodeOptions& opts) {
  FileEmitter emitter(getOutputDir());
  optimizeSimple(module);
  emitter.emitAllLayouts(module);
  for (auto stage : llvm::enumerate(opts.stages)) {
    auto moduleCopy = dyn_cast<ModuleOp>(module->clone());
    optimizeSplit(moduleCopy, stage.index(), stage.value());
    moduleCopy.walk([&](func::FuncOp func) {
      auto stage_name = stage.value().name;
      emitter.emitRustStep(stage.value().name, func);
      if (stage_name == "compute_accum" || stage_name == "verify_accum") {
        emitter.emitGpuStep(stage.value().name, ".metal", func);
      }
      emitter.emitGpuStep(stage.value().name, ".cu", func);
    });
  }
  optimizePoly(module);
  module.walk([&](func::FuncOp func) {
    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func, opts.info);
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
