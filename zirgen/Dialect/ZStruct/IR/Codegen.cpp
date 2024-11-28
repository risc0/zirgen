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

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"

namespace zirgen::ZStruct {

void addCppSyntax(codegen::CodegenOptions& opts) {
  opts.addLiteralSyntax<RefAttr>([](codegen::CodegenEmitter& cg, RefAttr refAttr) {
    cg << "/*offset=*/" << refAttr.getIndex();
  });
}

void addRustSyntax(codegen::CodegenOptions& opts) {
  opts.addLiteralSyntax<RefAttr>([](codegen::CodegenEmitter& cg, RefAttr refAttr) {
    cg << "&Reg{offset: " << refAttr.getIndex() << "}";
  });

  opts.addOpSyntax<LookupOp>([](codegen::CodegenEmitter& cg, LookupOp op) {
    if (llvm::isa<LayoutType>(op.getBase().getType())) {
      cg << "(" << op.getBase() << ".map(|c| c."
         << codegen::CodegenIdent<codegen::IdentKind::Field>(op.getMemberAttr()) << "))";
    } else {
      cg << op.getBase() << "."
         << codegen::CodegenIdent<codegen::IdentKind::Field>(op.getMemberAttr());
    }
  });

  opts.addOpSyntax<SubscriptOp>([](codegen::CodegenEmitter& cg, SubscriptOp op) {
    if (llvm::isa<LayoutArrayType>(op.getBase().getType())) {
      cg << "(" << op.getBase() << ".map(|c| c[to_usize(" << op.getIndex() << ")]))";
    } else {
      cg << op.getBase() << "[to_usize(" << op.getIndex() << ")]";
    }
  });

  // Load and store methods need to calculate per-cycle buffer offsets
  // for non-global buffers, so they need to pass in the current
  // context
  opts.addOpSyntax<LoadOp>([](codegen::CodegenEmitter& cg, LoadOp op) {
    auto elemType = op.getRef().getType().getElement();

    cg << op.getRef() << ".load";
    if (op->getAttr("unchecked"))
      cg << "_unchecked";
    if (elemType.getExtended())
      cg << "_ext::<ExtVal>";
    cg << "(ctx, " << op.getDistance() << ")";
  });

  opts.addOpSyntax<StoreOp>([](codegen::CodegenEmitter& cg, StoreOp op) {
    auto elemType = op.getRef().getType().getElement();

    cg << op.getRef() << ".store";
    if (elemType.getExtended()) {
      cg << "_ext";
    }
    cg << "(ctx, " << op.getVal() << ")";
  });
}

} // namespace zirgen::ZStruct
