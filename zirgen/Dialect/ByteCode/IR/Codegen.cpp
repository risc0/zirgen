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

void addCppSyntaxImpl(codegen::CodegenOptions& opts, bool isCuda) {
  opts.markStatementOp<YieldOp>();
  opts.addOpSyntax<YieldOp>([](codegen::CodegenEmitter& cg, YieldOp op) {
    for (auto result : op.getVals()) {
      cg << "   tempStore<" << cg.getTypeName(result.getType()) << ">(fpBuffer, prog.decode(), "
         << result << ");\n";
    }
  });

  opts.addOpSyntax<DecodeOp>(
      [](codegen::CodegenEmitter& cg, DecodeOp op) { cg << "prog.decode()"; });
  opts.addOpSyntax<LoadOp>([](codegen::CodegenEmitter& cg, LoadOp op) {
    cg << "tempLoad<" << cg.getTypeName(op.getType()) << ">(fpBuffer, prog.decode())";
  });

  opts.addOpSyntax<WrappedOp>([](codegen::CodegenEmitter& cg, WrappedOp op) {
    cg << CodegenIdent<IdentKind::Func>(op.getWrappedOpNameAttr()) << "(";
    cg.interleaveComma(llvm::concat<Value>(op.getVals(), op.getIntArgs()));
    cg << ")";
  });

  opts.markStatementOp<ExitOp>();
  opts.addOpSyntax<ExitOp>([](codegen::CodegenEmitter& cg, ExitOp op) {
    assert(op.getNumOperands() <= 1);
    cg << "return ";
    cg.interleaveComma(op.getVals());
    cg << ";\n";
  });

  opts.markStatementOp<ExecutorOp>();
  opts.addFuncContextArgument<ExecutorOp>("size_t cycle");
  opts.addFuncContextArgument<ExecutorOp>("size_t steps");
  opts.addOpSyntax<ExecutorOp>([=](codegen::CodegenEmitter& cg, ExecutorOp op) {
    SmallVector<CodegenIdent<IdentKind::Var>> argNames;
    for (DictionaryAttr argAttrDict : op.getArgAttrs()->getAsRange<DictionaryAttr>()) {
      argNames.push_back(argAttrDict.getAs<StringAttr>("zirgen.argName"));
    }
    cg << "template<typename ProgT>\n";
    cg.emitFunc(op,
                op.getFunctionType(),
                argNames,
                /*body=*/[&]() {
                  cg << "size_t mask = steps - 1;\n";
                  cg << "Fp fpBuffer[ProgT::getFpBufferSize()];\n";
                  cg << "ProgT prog;\n";
                  cg << "prog.reset();\n";
                  cg << "for (;;) {\n";
                  cg << "size_t dispatchKey = prog.decode();\n";
                  cg << "switch(dispatchKey) {\n";

                  for (auto [idx, arm] : llvm::enumerate(llvm::make_pointer_range(op.getArms()))) {
                    cg << "case " << idx << ": {\n";

                    cg.emitBlock(arm->front());
                    cg << "}; break;";
                  }
                  cg << "default: assert(false && \"unknown bytecode\");\n";
                  cg << "} // switch(dispatchKey)\n";
                  cg << "} // for(;;)\n";
                });
  });

  opts.addOpSyntax<GetArgumentOp>([=](codegen::CodegenEmitter& cg, GetArgumentOp op) {
    cg << CodegenIdent<IdentKind::Var>(op.getArgNameAttr());
  });

  opts.markStatementOp<EncodedBlockOp>();
  opts.addOpSyntax<EncodedBlockOp>([=](codegen::CodegenEmitter& cg, EncodedBlockOp op) {
    auto encodedOps = op.getBody().front().getOps<EncodedOp>();
    size_t totLen = 0;
    for (EncodedOp op : encodedOps)
      totLen += op.size();
    auto encodedConstName = cg.getIdent<IdentKind::Const>("encoded_" + op.getSymName());
    if (isCuda)
      cg << "  __device__ ";
    cg << "static constexpr uint32_t " << encodedConstName << "[" << totLen << "] = {";
    cg.interleaveComma(encodedOps, [&](EncodedOp encoded) {
      cg << "\n";
      cg.emitLoc(encoded.getLoc());
      cg.interleaveComma(encoded.getEncoded());
    });
    cg << "\n};\n";

    CodegenIdent<IdentKind::Type> symName = op.getSymNameAttr();
    cg << "struct " << symName << " : public EncodedProgBase<" << symName << "> {\n";
    if (isCuda)
      cg << "  __device__";
    cg << "  void reset() { curPos = " << encodedConstName << "; }\n";
    for (auto tempBuf : op.getTempBufs()->getAsRange<TempBufAttr>()) {
      if (isCuda)
        cg << "  __device__ ";
      cg << " static constexpr size_t "
         << cg.getIdent<IdentKind::Func>("get_" + tempBuf.getBufName().strref() + "_size")
         << "() { return " << tempBuf.getSize() << "; }\n";
    }
    cg << "};\n";
  });
}

} // namespace

void addCppSyntax(codegen::CodegenOptions& opts) {
  addCppSyntaxImpl(opts, /*cuda=*/false);
}

void addCudaSyntax(codegen::CodegenOptions& opts) {
  addCppSyntaxImpl(opts, /*cuda=*/true);
}

} // namespace zirgen::ByteCode
