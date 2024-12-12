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

#include "zirgen/circuit/verify/circom/circom.h"
#include "zirgen/compiler/zkp/poseidon_254.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen::snark {

constexpr uint64_t P = 15 * (1 << 27) + 1;

CircomGenerator::CircomGenerator(std::ostream& outs) : outs(outs) {}

void CircomGenerator::emit(mlir::func::FuncOp func, bool encodeOutput) {
  // Get the main block from the function
  Block& block = func.front();
  // First, compute size of IOP
  size_t iopSize = 0;
  block.walk([&](Iop::ReadOp op) { iopSize += op.getOuts().size(); });
  // Emit the fixed header
  emitHeader();
  // Output the iop input signal
  outs << "  signal input iop[" << iopSize << "];\n";
  // Check if the function has an output buffer, and if so add it
  size_t outSize = 0;
  if (auto btype = dyn_cast<BufferType>(block.getArgument(0).getType())) {
    if (btype.getKind() == BufferKind::Global) {
      outSize = btype.getSize();
      if (encodeOutput) {
        assert(outSize % 8 == 0);
        outs << "  signal out_tmp[" << outSize << "];\n";
        outs << "  signal output out[" << outSize / 8 << "];\n";
        outs << "  signal output codeRoot;\n";
      } else {
        outs << "  signal output out_tmp[" << outSize << "];\n";
      }
    }
  }
  if (outSize == 0) {
    outs << "  signal output final_iop[3];\n";
  }
  outs << "  component iop_init = iop_init_impl();\n";
  curIop = "iop_init.iop";

  for (Operation& origOp : func.front().without_terminator()) {
    TypeSwitch<Operation*>(&origOp)
        .Case<ConstOp>([&](ConstOp op) { emit(op); })
        .Case<NegOp>([&](NegOp op) { emit(op); })
        .Case<AddOp>([&](AddOp op) { emitBinary('+', origOp); })
        .Case<SubOp>([&](SubOp op) { emitBinary('-', origOp); })
        .Case<MulOp>([&](MulOp op) { emitBinary('*', origOp); })
        .Case<Iop::CommitOp>([&](Iop::CommitOp op) { emit(op); })
        .Case<Iop::ReadOp>([&](Iop::ReadOp op) { emit(op); })
        .Case<Iop::RngBitsOp>([&](Iop::RngBitsOp op) { emit(op); })
        .Case<Iop::RngValOp>([&](Iop::RngValOp op) { emit(op); })
        .Case<EqualZeroOp>([&](EqualZeroOp op) { emit(op); })
        .Case<HashAssertEqOp>([&](HashAssertEqOp op) { emit(op); })
        .Case<SelectOp>([&](SelectOp op) { emit(op); })
        .Case<NormalizeOp>([&](NormalizeOp op) { emit(op); })
        .Case<SetGlobalOp>([&](SetGlobalOp op) {
          assert(op.getBuf() == block.getArgument(0));
          outs << "  out_tmp[" << op.getOffset() << "] <== " << signal[op.getIn()] << ";\n";
        })
        .Case<SetGlobalDigestOp>([&](SetGlobalDigestOp op) {
          // The code root is the only digest val that is set in globals.
          assert(op.getBuf() == block.getArgument(1));
          outs << "  codeRoot <== " << signal[op.getIn()] << ";\n";
        })
        .Default([&](Operation* op) {
          // Generic op handler
          std::string opName = op->getName().stripDialect().str();
          if (op->getNumResults() != 1) {
            llvm::errs() << "BAD THING: " << *op << "\n";
            throw std::runtime_error("YIKES");
          }
          assert(op->getNumResults() == 1);
          size_t id = idCount++;
          size_t inCount = op->getNumOperands();
          outs << "  component comp_" << id << " = " << opName << "_impl(" << inCount << ");\n";
          for (size_t i = 0; i < inCount; i++) {
            outs << "  comp_" << id << ".in[" << i << "] <== " << signal[op->getOperand(i)]
                 << ";\n";
          }
          signal[op->getResult(0)] = "comp_" + std::to_string(id) + ".out";
        });
  }
  if (encodeOutput) {
    for (size_t i = 0; i < outSize / 8; i++) {
      P254 mul = 1;
      outs << "  out[" << i << "] <== 0 ";
      for (size_t j = 0; j < 8; j++) {
        outs << "+ " << mul.toString() << " * out_tmp[" << i * 8 + j << "] ";
        mul = mul * (1 << 16);
      }
      outs << ";\n";
    }
  }
  if (outSize == 0) {
    outs << "  final_iop <== " << curIop << ";\n";
  }
  emitFooter();
}

void CircomGenerator::emitHeader() {
  outs << "pragma circom 2.0.4;\n\n";
  outs << "include \"risc0.circom\";\n\n";

  outs << "template Verify() {\n";
}

void CircomGenerator::emitFooter() {
  outs << "}\n\n";
  outs << "component main = Verify();\n";
}

void CircomGenerator::emit(ConstOp op) {
  signal[op.getOut()] = std::to_string(op.getCoefficients()[0]);
}

void CircomGenerator::emit(NegOp op) {
  size_t id = idCount++;
  outs << "  signal local_" << id << ";\n";
  outs << "  local_" << id << " <== -" << signal[op.getIn()] << ";\n";
  signal[op.getOut()] = "local_" + std::to_string(id);
}

void CircomGenerator::emitBinary(char symbol, Operation& op) {
  size_t id = idCount++;
  outs << "  signal local_" << id << ";\n";
  outs << "  local_" << id << " <== " << signal[op.getOperand(0)] << " " << symbol << " "
       << signal[op.getOperand(1)] << ";\n";
  // outs << "  log(\"  " << symbol << " : \", local_" << id << ");\n";
  signal[op.getResult(0)] = "local_" + std::to_string(id);
}

void CircomGenerator::emit(Iop::CommitOp op) {
  size_t id = idCount++;
  outs << "  component comp_" << id << " = iop_commit_impl();\n";
  outs << "  comp_" << id << ".old_iop <== " << curIop << ";\n";
  outs << "  comp_" << id << ".digest <== " << signal[op.getDigest()] << ";\n";
  curIop = "comp_" + std::to_string(id) + ".new_iop";
}

void CircomGenerator::emit(Iop::ReadOp op) {
  for (size_t i = 0; i < op.getOuts().size(); i++) {
    signal[op.getOuts()[i]] = "iop[" + std::to_string(iopOffset + i) + "]";
  }
  iopOffset += op.getOuts().size();
}

void CircomGenerator::emit(Iop::RngBitsOp op) {
  size_t id = idCount++;
  outs << "  component comp_" << id << " = iop_rng_bits_impl(" << op.getBits() << ");\n";
  outs << "  comp_" << id << ".old_iop <== " << curIop << ";\n";
  signal[op.getOut()] = "comp_" + std::to_string(id) + ".rng_out";
  curIop = "comp_" + std::to_string(id) + ".new_iop";
}

void CircomGenerator::emit(Iop::RngValOp op) {
  size_t id = idCount++;
  outs << "  component comp_" << id << " = iop_rng_val_impl();\n";
  outs << "  comp_" << id << ".old_iop <== " << curIop << ";\n";
  signal[op.getOut()] = "comp_" + std::to_string(id) + ".rng_out";
  curIop = "comp_" + std::to_string(id) + ".new_iop";
}

void CircomGenerator::emit(EqualZeroOp op) {
  outs << "  " << signal[op.getIn()] << " === 0;\n";
}

void CircomGenerator::emit(HashAssertEqOp op) {
  outs << "  " << signal[op.getLhs()] << " === " << signal[op.getRhs()] << ";\n";
}

void CircomGenerator::emit(SelectOp op) {
  size_t id = idCount++;
  outs << "  component comp_" << id << " = select_impl(" << op.getElems().size() << ");\n";
  for (size_t i = 0; i < op.getElems().size(); i++) {
    outs << "  comp_" << id << ".elems[" << i << "] <== " << signal[op.getElems()[i]] << ";\n";
  }
  outs << "  comp_" << id << ".idx <== " << signal[op.getIdx()] << ";\n";
  signal[op.getOut()] = "comp_" + std::to_string(id) + ".out";
}

void CircomGenerator::emit(NormalizeOp op) {
  size_t id = idCount++;
  outs << "  component comp_" << id << " = normalize_impl(" << op.getBits() << ");\n";
  BigInt low = op.getLow();
  BigInt addP = 0;
  if (low < 0) {
    addP = (BigInt(P - 1) - low) / P * P;
  }
  outs << "  comp_" << id << ".in <== " << signal[op.getIn()] << " + " << addP.toStr() << ";\n";
  signal[op.getOut()] = "comp_" + std::to_string(id) + ".out";
}

} // namespace zirgen::snark
