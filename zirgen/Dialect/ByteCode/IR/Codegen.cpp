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

#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ByteCode/IR/Codegen.h"

using namespace mlir;
using namespace zirgen::codegen;
using namespace zirgen::Zll;

namespace zirgen::ByteCode {

namespace {

EmitPart getLocalAccessor(Attribute intKind) {
  return [=](CodegenEmitter& cg) {
    cg << cg.getIdent<IdentKind::Var>(getNameForIntKind(intKind)) << "[readBits<"
       << cg.getIdent<IdentKind::Const>(getNameForIntKind(intKind) + "Bits") << ">(execPos, ";
    cg.emitEscapedString(getNameForIntKind(intKind));
    cg << ")]";
  };
}

void addCppSyntaxImpl(codegen::CodegenOptions& opts, bool isCuda) {
#if 0
  opts.addFuncContextArgument<func::FuncOp>("size_t cycle");
  opts.addFuncContextArgument<func::FuncOp>("size_t steps");
  opts.addOpSyntax<DefineEncodedOp>([isCuda](codegen::CodegenEmitter& cg, DefineEncodedOp op) {
    EncodedAttr encoded = op.getEncoded();
    for (auto tempBuf : encoded.getTempBufs()) {
      cg << "static constexpr size_t "
         << cg.getIdent<IdentKind::Const>("num" + getNameForIntKind(tempBuf.getIntKind())) << " = "
         << tempBuf.getSize() << ";\n";
    }
    cg << "static constexpr size_t kEncodedSize = " << encoded.getEncoded().size() << ";\n";
    if (isCuda)
      cg << "__device__ ";
    cg << "static constexpr unsigned char kEncoded[] = {";
    cg.interleaveComma(
        llvm::map_range(encoded.getEncoded(), [](unsigned char c) { return int(c); }));
    cg << "}\n";
  });

  opts.markStatementOp<YieldOp>();
  opts.addOpSyntax<YieldOp>([](codegen::CodegenEmitter& cg, YieldOp op) {
    for (auto [result, intKind] : llvm::zip_equal(op.getOperands(), op.getIntKinds())) {
      cg << "   if (kDebug) debugOut(" << result << ");\n";
      cg << getLocalAccessor(intKind) << " = " << result << ";\n";
    }
  });

  opts.addOpSyntax<DecodeOp>([](codegen::CodegenEmitter& cg, DecodeOp op) {
    cg << "readBits<" << cg.getIdent<IdentKind::Const>(getNameForIntKind(op.getIntKind()) + "Bits")
       << ">(execPos, ";
    cg.emitEscapedString(getNameForIntKind(op.getIntKind()));
    cg << ")";
  });
  opts.addOpSyntax<LoadTemporaryOp>([](codegen::CodegenEmitter& cg, LoadTemporaryOp op) {
    cg << "debugIn(" << getLocalAccessor(op.getIntKind()) << ")";
  });

  opts.addOpSyntax<WrappedOp>([](codegen::CodegenEmitter& cg, WrappedOp op) {
    cg << CodegenIdent<IdentKind::Func>(op.getWrappedOpNameAttr()) << "(";
    cg.interleaveComma(op.getOperands());
    cg << ")";
  });

  opts.addOpSyntax<ExitOp>([](codegen::CodegenEmitter& cg, ExitOp op) {
    assert(op.getNumOperands() <= 1);
    cg << "return ";
    if (op.getNumOperands())
      cg << op.getOperand(0);
  });

  opts.markStatementOp<GetEncodedOp>();
  opts.addOpSyntax<GetEncodedOp>([](codegen::CodegenEmitter& cg, GetEncodedOp op) {});

  opts.markStatementOp<func::ReturnOp>();
  opts.addOpSyntax<func::ReturnOp>([](codegen::CodegenEmitter& cg, func::ReturnOp op) {
    cg << "assert(false && \"byte code execution loop should never end\");";
  });

  opts.markStatementOp<ExecutorOp>();
  opts.addOpSyntax<ExecutorOp>([=](codegen::CodegenEmitter& cg, ExecutorOp op) {
    for (auto intInfo : op.getIntKinds().getAsRange<IntKindInfoAttr>()) {
      cg << "constexpr size_t "
         << cg.getIdent<IdentKind::Const>(getNameForIntKind(intInfo.getIntKind()) + "Bits") << " = "
         << intInfo.getEncodedBits() << ";\n";
    }
    cg << "size_t mask = steps - 1;\n";
    cg << "const uint8_t* execPos = kEncoded;\n";

    // TODO: Gather these declarations from the bufferizer:
    cg << "Fp localFp[kNumLocalFp];\n";
    cg << "FpExt localFpExt[kNumLocalFpExt];\n";

    cg << "for (;;) {\n";
    cg << "switch(readBits<kDispatchKeyBits>(execPos, \"DispatchKey\")) {\n";

    for (auto [idx, arm] : llvm::enumerate(llvm::make_pointer_range(op.getArms()))) {
      cg << "case " << idx << ": {\n";
      auto key = op.getArmDispatchKey(idx);
      auto& os = *cg.getOutputStream();
      os << "// " << key.getOperationName() << "\n";

      if (!key.getOperandTypes().empty()) {
        os << "// : ";
        interleaveComma(key.getOperandTypes(), os);
        os << "\n";
      }
      if (key.getIntKinds().size()) {
        os << "// also decode: ";
        interleaveComma(key.getIntKinds(), os);
        os << "\n";
      }
      if (!key.getResultTypes().empty()) {
        os << "//  -> ";
        interleaveComma(key.getResultTypes(), os);
        os << "\n";
      }

      os << "  if (kDebug) printf(\"%u: %s\\n\", " << idx << "u, \"" << key.getOperationName()
         << "\");\n";

      cg.emitBlock(arm->front());
      cg << "}; break;";
    }
    cg << "default: assert(false && \"unknown bytecode\");\n";
    cg << "} // switch(dispatchKey)\n";
    cg << "} // for(;;)\n";
  });
#endif
}

} // namespace

void addCppSyntax(codegen::CodegenOptions& opts) {
  addCppSyntaxImpl(opts, /*cuda=*/false);
}

void addCudaSyntax(codegen::CodegenOptions& opts) {
  addCppSyntaxImpl(opts, /*cuda=*/true);
}

} // namespace zirgen::ByteCode
