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

#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Zhlt {
namespace {

using namespace mlir;
using namespace zirgen::codegen;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

struct RustEmitZhlt : public EmitZhlt {
  using EmitZhlt::EmitZhlt;

  LogicalResult emitBufferList(ArrayRef<BufferDescAttr> bufs) override {
    cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineBufferList")) << "!{\n";

    cg << "all: [";
    for (auto desc : bufs)
      cg << CodegenIdent<IdentKind::Var>(desc.getName()) << ",";
    cg << "],\n";

    cg << "rows: [";
    for (auto desc : bufs)
      if (desc.getType().getKind() != BufferKind::Global)
        cg << CodegenIdent<IdentKind::Var>(desc.getName()) << ",";
    cg << "],\n";

    cg << "taps: [";
    for (auto desc : bufs)
      if (desc.getRegGroupId())
        cg << CodegenIdent<IdentKind::Var>(desc.getName()) << ",";
    cg << "],\n";

    cg << "globals: [";
    for (auto desc : bufs)
      if (desc.isGlobal())
        cg << CodegenIdent<IdentKind::Var>(desc.getName()) << ",";
    cg << "],}\n";

    for (auto desc : bufs) {
      auto name = CodegenIdent<IdentKind::Var>(desc.getName());
      if (desc.getRegGroupId()) {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineTapBuffer")) << "!{" << name
           << ", /*count=*/" << desc.getRegCount() << ", /*groupId=*/" << *desc.getRegGroupId()
           << "}\n";
      } else if (desc.getKind() == BufferKind::Global) {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineGlobalBuffer")) << "!{" << name
           << ", /*count=*/" << desc.getRegCount() << "}\n";
      } else {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineBuffer")) << "!{" << name
           << ", /*count=*/" << desc.getRegCount() << "}\n";
      }
    }
    return success();
  }
};

struct CppEmitZhlt : public EmitZhlt {
  using EmitZhlt::EmitZhlt;

  LogicalResult emitBufferList(ArrayRef<BufferDescAttr> bufs) override {
    for (auto desc : bufs) {
      auto name = cg.getStringAttr("regCount_" + desc.getName().str());
      cg << "constexpr size_t " << CodegenIdent<IdentKind::Const>(name) << " = "
         << desc.getRegCount() << ";\n";
    }
    return success();
  }
};

} // namespace

LogicalResult EmitZhlt::doValType() {
  DenseSet<Zll::FieldAttr> fields;

  AttrTypeWalker typeWalker;
  typeWalker.addWalk([&](ValType ty) { fields.insert(ty.getField()); });

  module.walk([&](Operation* op) {
    for (Type ty : op->getOperandTypes()) {
      typeWalker.walk(ty);
    }
    for (Type ty : op->getResultTypes()) {
      typeWalker.walk(ty);
    }
  });

  if (fields.empty()) {
    fields.insert(Zll::getDefaultField(ctx));
  }

  if (fields.size() > 1) {
    return emitError(UnknownLoc::get(ctx), "Ambiguous circuit field");
  }

  auto field = *fields.begin();

  cg.emitInvokeMacro(
      cg.getStringAttr("setField"),
      {codegen::CodegenIdent<codegen::IdentKind::Type>(cg.getStringAttr(field.getName()))});
  cg << ";\n";
  return success();
}

LogicalResult EmitZhlt::doBuffers() {
  auto bufs = Zll::lookupModuleAttr<BuffersAttr>(module);
  if (failed(emitBufferList(bufs.getBuffers())))
    return failure();

  return success();
}

std::unique_ptr<EmitZhlt> getEmitter(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg) {
  std::unique_ptr<EmitZhlt> impl;

  switch (cg.getLanguageKind()) {
  case LanguageKind::Cpp:
    impl = std::make_unique<CppEmitZhlt>(module, cg);
    break;
  case LanguageKind::Rust:
    impl = std::make_unique<RustEmitZhlt>(module, cg);
    break;
  }

  assert(impl && "Unknown language kind");

  return impl;
}

LogicalResult emitModule(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg) {
  auto emitter = getEmitter(module, cg);
  if (failed(emitter->emitDefs()))
    return failure();
  cg.emitModule(module);
  return success();
}

void addCppSyntax(codegen::CodegenOptions& opts) {
  opts.addFuncContextArgument<CheckFuncOp, ExecFuncOp, BackFuncOp, StepFuncOp>("ExecContext& ctx");
  opts.addFuncContextArgument<ValidityTapsFuncOp>("ValidityTapsContext& ctx");
  opts.addFuncContextArgument<ValidityRegsFuncOp>("ValidityRegsContext& ctx");

  opts.addCallContextArgument<ExecCallOp, BackCallOp, StepCallOp>("ctx");
}

void addRustSyntax(codegen::CodegenOptions& opts) {
  opts.addFuncContextArgument<CheckFuncOp, ExecFuncOp, BackFuncOp, StepFuncOp>(
      "ctx: &'a ExecContext");
  opts.addFuncContextArgument<ValidityTapsFuncOp>("ctx: &'a ValidityTapsContext");
  opts.addFuncContextArgument<ValidityRegsFuncOp>("ctx: &'a ValidityRegsContext");

  opts.addCallContextArgument<ExecCallOp, BackCallOp, StepCallOp>("ctx");
}

} // namespace zirgen::Zhlt
