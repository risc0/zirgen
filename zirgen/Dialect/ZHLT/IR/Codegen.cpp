// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"

namespace zirgen::Zhlt {
namespace {

using namespace mlir;
using namespace zirgen::codegen;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

class EmitZhltBase {
public:
  EmitZhltBase(ModuleOp module, CodegenEmitter& cg)
      : module(module), cg(cg), ctx(module.getContext()), bufferAnalysis(module) {}
  virtual ~EmitZhltBase() = default;

  LogicalResult emit() {
    if (failed(doValType()))
      return failure();

    if (failed(doBuffers()))
      return failure();

    cg.emitModule(module);

    return success();
  }

protected:
  virtual LogicalResult emitBufferList(ArrayRef<BufferDesc> bufs) { return success(); }

  ModuleOp module;
  CodegenEmitter& cg;
  MLIRContext* ctx;
  BufferAnalysis bufferAnalysis;

private:
  // Declares "Val" to be a type alias to the appropriete field element.
  LogicalResult doValType();

  // Provide buffers and sizes
  LogicalResult doBuffers();
};

struct RustEmitZhlt : public EmitZhltBase {
  using EmitZhltBase::EmitZhltBase;

  LogicalResult emitBufferList(ArrayRef<BufferDesc> bufs) override {
    cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineBufferList")) << "!{\n";

    cg << "all: [";
    for (auto desc : bufs)
      cg << CodegenIdent<IdentKind::Var>(desc.name) << ",";
    cg << "],\n";

    cg << "rows: [";
    for (auto desc : bufs)
      if (!desc.global)
        cg << CodegenIdent<IdentKind::Var>(desc.name) << ",";
    cg << "],\n";

    cg << "taps: [";
    for (auto desc : bufs)
      if (desc.regGroupId)
        cg << CodegenIdent<IdentKind::Var>(desc.name) << ",";
    cg << "],\n";

    cg << "globals: [";
    for (auto desc : bufs)
      if (desc.global)
        cg << CodegenIdent<IdentKind::Var>(desc.name) << ",";
    cg << "],}\n";

    for (auto desc : bufs) {
      auto name = CodegenIdent<IdentKind::Var>(desc.name);
      if (desc.regGroupId) {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineTapBuffer")) << "!{" << name
           << ", /*count=*/" << desc.regCount << ", /*groupId=*/" << *desc.regGroupId << "}\n";
      } else if (desc.kind == BufferKind::Global) {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineGlobalBuffer")) << "!{" << name
           << ", /*count=*/" << desc.regCount << "}\n";
      } else {
        cg << CodegenIdent<IdentKind::Macro>(cg.getStringAttr("defineBuffer")) << "!{" << name
           << ", /*count=*/" << desc.regCount << "}\n";
      }
    }
    return success();
  }
};

struct CppEmitZhlt : public EmitZhltBase {
  using EmitZhltBase::EmitZhltBase;

  LogicalResult emitBufferList(ArrayRef<BufferDesc> bufs) override {
    for (auto desc : bufs) {
      auto name = cg.getStringAttr("regCount_" + desc.name.str());
      cg << "constexpr size_t " << CodegenIdent<IdentKind::Const>(name) << " = " << desc.regCount
         << ";\n";
    }
    return success();
  }
};

LogicalResult EmitZhltBase::doValType() {
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

LogicalResult EmitZhltBase::doBuffers() {
  if (failed(emitBufferList(bufferAnalysis.getAllBuffers())))
    return failure();

  return success();
}

} // namespace

LogicalResult emitModule(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg) {
  std::unique_ptr<EmitZhltBase> impl;

  switch (cg.getLanguageKind()) {
  case LanguageKind::Cpp:
    impl = std::make_unique<CppEmitZhlt>(module, cg);
    break;
  case LanguageKind::Rust:
    impl = std::make_unique<RustEmitZhlt>(module, cg);
    break;
  }

  assert(impl && "Unknown language kind");

  return impl->emit();
}

} // namespace zirgen::Zhlt
