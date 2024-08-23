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

#include "zirgen/circuit/recursion/micro.h"

#include "zirgen/components/fpext.h"

namespace zirgen::recursion {

void MicroOpImpl::set(MicroInst inst, Val writeAddr, Reg extraPrev, size_t extraBack) {
  decode->set(inst->opcode);
  std::array<Val, kExtSize> operands;
  operands[0] = inst->operands[0]->get();
  operands[1] = inst->operands[1]->get();
  operands[2] = inst->operands[2]->get();
  operands[3] = 0;

  IF(decode->at(size_t(MicroOpcode::CONST))) {
    XLOG("%u> CONST: %e", writeAddr, operands);
    in0->doNOP();
    in1->doNOP();
    out->doWrite(writeAddr, operands);
  }
  IF(decode->at(size_t(MicroOpcode::ADD))) {
    FpExt a = in0->doRead(operands[0]);
    FpExt b = in1->doRead(operands[1]);
    out->doWrite(writeAddr, (a + b).getElems());
    XLOG("%u> ADD: %e + %e -> %e", writeAddr, a.getElems(), b.getElems(), out->data());
  }
  IF(decode->at(size_t(MicroOpcode::SUB))) {
    FpExt a = in0->doRead(operands[0]);
    FpExt b = in1->doRead(operands[1]);
    out->doWrite(writeAddr, (a - b).getElems());
    XLOG("%u> SUB: %e - %e -> %e", writeAddr, a.getElems(), b.getElems(), out->data());
  }
  IF(decode->at(size_t(MicroOpcode::MUL))) {
    FpExt a = in0->doRead(operands[0]);
    FpExt b = in1->doRead(operands[1]);
    out->doWrite(writeAddr, (a * b).getElems());
    XLOG("%u> MUL: %e * %e -> %e", writeAddr, a.getElems(), b.getElems(), out->data());
  }
  IF(decode->at(size_t(MicroOpcode::INV)) * operands[1]) {
    FpExt a = in0->doRead(operands[0]);
    in1->doNOP();
    NONDET { out->doWrite(writeAddr, inv(a).getElems()); }
    XLOG("INV: %e -> %e", a.getElems(), out->data());
    eq(FpExt(Val(1)), FpExt(in0->data()) * FpExt(out->data()));
  }
  IF(decode->at(size_t(MicroOpcode::INV)) * (1 - operands[1])) {
    in0->doRead(operands[0]);
    in1->doNOP();
    Val input = in0->data()[0];
    NONDET {
      extra->set(inv(input));
      out->doWrite(writeAddr, {1 - extra * input, 0, 0, 0});
    }
    Val maybeInv = extra;
    Val isZero = out->data()[0];
    // isZero is a bit
    eqz(isZero * (1 - isZero));
    // If it's nonzero, input has an inverse
    eq(maybeInv * input, 1 - isZero);
    // Either isZero is false, or input is zero
    eqz(isZero * input);
    XLOG("%u> IS_ZERO: %e -> %e", writeAddr, in0->data(), out->data());
  }
  IF(decode->at(size_t(MicroOpcode::EQ))) {
    FpExt a = in0->doRead(operands[0]);
    FpExt b = in1->doRead(operands[1]);
    out->doWrite(writeAddr, (a - b).getElems());
    XLOG("%u> EQ: %e == %e", writeAddr, a.getElems(), b.getElems());
    eq(FpExt(Val(0)), FpExt(out->data()));
  }
  IF(decode->at(size_t(MicroOpcode::READ_IOP_HEADER))) {
    XLOG("%u> READ_IOP_HEADER: %u %u", writeAddr, operands[0], operands[1]);
    in0->doNOP();
    in1->doNOP();
    out->doWrite(writeAddr, {0, 0, 0, 0});
    NONDET { auto vals = doExtern("readIOPHeader", "", 0, {operands[0], operands[1]}); }
  }
  IF(decode->at(size_t(MicroOpcode::READ_IOP_BODY))) {
    in0->doNOP();
    in1->doNOP();
    NONDET {
      auto vals = doExtern("readIOPBody", "", kExtSize, {operands[0], operands[1], operands[2]});
      std::array<Val, kExtSize> asArr;
      for (size_t i = 0; i < asArr.size(); i++) {
        asArr[i] = vals[i];
      }
      out->doWrite(writeAddr, asArr);
    }
    XLOG("%u> READ_IOP_BODY: %u %u -> %e", writeAddr, operands[0], operands[1], out->data());
    eqz(operands[0] * out->data()[1]);
    eqz(operands[1] * out->data()[2]);
    eqz(operands[1] * out->data()[3]);
  }
  IF(decode->at(size_t(MicroOpcode::MIX_RNG))) {
    XLOG("%u> MIX_RNG: %u, %u, %u", writeAddr, operands[0], operands[1], operands[2]);
    in0->doRead(operands[0]);
    in1->doRead(operands[1]);
    XLOG("  in0=[%x %x], in1=[%x %x]",
         in0->data()[0],
         in0->data()[1],
         in1->data()[0],
         in1->data()[1]);
    Val val = operands[2] * UNCHECKED_BACK(extraBack, extraPrev->get());
    XLOG("  prev_val = %u", val);
    val = val * (1 << 16) + in0->data()[1];
    val = val * (1 << 16) + in0->data()[0];
    val = val * (1 << 16) + in1->data()[1];
    val = val * (1 << 16) + in1->data()[0];
    XLOG("  val = %u", val);
    extra->set(val);
    out->doWrite(writeAddr, {val, 0, 0, 0});
  }
  IF(decode->at(size_t(MicroOpcode::SELECT))) {
    in0->doRead(operands[0]);
    in1->doRead(operands[1] + operands[2] * in0->data()[0]);
    out->doWrite(writeAddr, in1->data());
    XLOG("%u> SELECT, idx = %u, start = %u, step = %u, idx = %u, writing %e to %u",
         writeAddr,
         operands[0],
         operands[1],
         operands[2],
         in0->data()[0],
         in1->data(),
         writeAddr);
  }
  IF(decode->at(size_t(MicroOpcode::EXTRACT))) {
    XLOG("%u> EXTRACT: %e", writeAddr, operands);
    in0->doRead(operands[0]);
    in1->doNOP();
    Val val = operands[1] * operands[2] * in0->data()[3] +
              operands[1] * (1 - operands[2]) * in0->data()[2] +
              (1 - operands[1]) * operands[2] * in0->data()[1] +
              (1 - operands[1]) * (1 - operands[2]) * in0->data()[0];
    out->doWrite(writeAddr, {val, 0, 0, 0});
  }
}

MicroOpsImpl::MicroOpsImpl(Code code, WomHeader header) : body(Label("wom_body"), header, 9, 4) {
  for (size_t i = 0; i < 3; i++) {
    ops.emplace_back(Label("op", i));
  }
}

void MicroOpsImpl::set(Code code, Val writeAddr) {
  MicroInsts insts = code->inst->at<size_t(OpType::MICRO)>();
  for (size_t i = 0; i < 3; i++) {
    Reg extraPrev = ops[(i + 2) % 3]->extra;
    size_t extraBack = (i == 0 ? 1 : 0);
    ops[i]->set(insts->insts[i], writeAddr + i, extraPrev, extraBack);
  }
}

} // namespace zirgen::recursion
