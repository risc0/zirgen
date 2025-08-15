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

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/Passes.h"

#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/Main/Utils.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace mlir;
using namespace zirgen::codegen;

namespace zirgen {

std::unique_ptr<llvm::raw_ostream> openOutput(StringRef filename) {
  std::string path = (codegenCLOptions->outputDir + "/" + filename).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(path, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + path);
  }
  return ofs;
}

void emitPoly(ModuleOp mod, StringRef circuitName, StringRef protocolInfo) {
  // mod.print(llvm::outs());
  ModuleOp funcMod = mod.cloneWithoutRegions();
  OpBuilder builder(funcMod.getContext());
  builder.createBlock(&funcMod->getRegion(0));

  // Convert functions to func::FuncOp, since that's what the edsl
  // codegen knows how to deal with
  mod.walk([&](zirgen::Zhlt::CheckFuncOp funcOp) {
    llvm::outs() << "copy!\n";
    auto newFuncOp = builder.create<func::FuncOp>(funcOp.getLoc(),
                                                  builder.getStringAttr(circuitName),
                                                  TypeAttr::get(funcOp.getFunctionType()),
                                                  funcOp.getSymVisibilityAttr(),
                                                  funcOp.getArgAttrsAttr(),
                                                  funcOp.getResAttrsAttr());
    IRMapping mapping;
    newFuncOp.getBody().getBlocks().clear();
    funcOp.getBody().cloneInto(&newFuncOp.getBody(), mapping);
  });
  // funcMod.print(llvm::outs());

  zirgen::Zll::setModuleAttr(funcMod, builder.getAttr<zirgen::Zll::ProtocolInfoAttr>(protocolInfo));

  mlir::PassManager pm(mod.getContext());
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    exit(1);
  }
  {
    auto& opm = pm.nest<mlir::func::FuncOp>();
    // opm.addPass(zirgen::ZStruct::createInlineLayoutPass());
    opm.addPass(zirgen::ZStruct::createBuffersToArgsPass());
    opm.addPass(Zll::createMakePolynomialPass());
    opm.addPass(createCanonicalizerPass());
    opm.addPass(createCSEPass());
    opm.addPass(Zll::createComputeTapsPass());
  }

  //  pm.addPass(createPrintIRPass());

  if (failed(pm.run(funcMod))) {
    llvm::errs() << "an internal compiler error occurred while optimizing poly for this module:\n";
    funcMod.print(llvm::errs());
    exit(1);
  }

  emitCodeZirgenPoly(funcMod, codegenCLOptions->outputDir);

  // TODO: modularize generating the validity stuff
  auto rustOpts = codegen::getRustCodegenOpts();
  rustOpts.addFuncContextArgument<func::FuncOp>("ctx: &ValidityCtx");
  rustOpts.addCallContextArgument<Zll::GetOp, Zll::SetOp>("ctx");
  CodegenEmitter rustCg(rustOpts, mod.getContext());
  auto os = openOutput("validity.rs.inc");
  CodegenEmitter::StreamOutputGuard guard(rustCg, os.get());
  rustCg.emitModule(funcMod);
}

void emitDefs(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  auto emitZhlt = Zhlt::getEmitter(mod, cg);
  if (emitZhlt->emitDefs().failed()) {
    llvm::errs() << "Failed to emit circuit definitions to " << filename << "\n";
    exit(1);
  }
  *os << tmpl.footer;
}

void emitTypes(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  cg.emitTypeDefs(mod);
  *os << tmpl.footer;
}

template <typename... OpT>
void emitOps(CodegenEmitter& cg,
             ModuleOp mod,
             const Twine& filename,
             const Template& tmpl,
             size_t splitPart = 0,
             size_t numSplit = 1) {
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  size_t funcIdx = 0;
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      if ((funcIdx % numSplit) == splitPart) {
        cg.emitTopLevel(&op);
      }
      ++funcIdx;
    }
  }
  *os << tmpl.footer;
}

template <typename... OpT>
void emitOpDecls(CodegenEmitter& cg, ModuleOp mod, const Twine& filename, const Template& tmpl) {
  auto os = openOutput(filename.str());
  *os << tmpl.header;
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      cg.emitTopLevelDecl(&op);
    }
  }
  *os << tmpl.footer;
}

void emitTarget(const CodegenTarget& target,
                ModuleOp mod,
                ModuleOp stepFuncs,
                const CodegenOptions& opts,
                unsigned stepSplitCount) {
  CodegenEmitter cg(opts, mod.getContext());
  auto declExt = target.getDeclExtension();
  auto implExt = target.getImplExtension();

  emitDefs(cg, mod, "defs." + implExt + ".inc", target.getDefsTemplate());
  emitTypes(cg, mod, "types." + declExt + ".inc", target.getTypesTemplate());

  if (implExt != declExt) {
    emitOpDecls<ZStruct::GlobalConstOp>(
        cg, mod, "layout." + declExt + ".inc", target.getLayoutDeclTemplate());
  }
  emitOps<ZStruct::GlobalConstOp>(
      cg, mod, "layout." + implExt + ".inc", target.getLayoutTemplate());

  if (implExt != declExt) {
    emitOpDecls<Zhlt::StepFuncOp>(cg, stepFuncs, "steps." + declExt, target.getStepDeclTemplate());
  }
  if (stepSplitCount == 1) {
    emitOps<Zhlt::StepFuncOp>(cg, stepFuncs, "steps." + implExt, target.getStepTemplate());
  } else {
    for (size_t i = 0; i != stepSplitCount; ++i) {
      emitOps<Zhlt::StepFuncOp>(cg,
                                stepFuncs,
                                "steps_" + std::to_string(i) + "." + implExt,
                                target.getStepTemplate(),
                                i,
                                stepSplitCount);
    }
  }
}

} // namespace zirgen
