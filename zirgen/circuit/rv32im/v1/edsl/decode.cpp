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

#include "zirgen/circuit/rv32im/v1/edsl/decode.h"

namespace zirgen::rv32im_v1 {

void DecoderImpl::set(U32Val inst) {
  NONDET {
    f7_6->set((inst.bytes[3] & 0x80) / (1 << 7));
    f7_45->set((inst.bytes[3] & 0x60) / (1 << 5));
    f7_3->set((inst.bytes[3] & 0x10) / (1 << 4));
    f7_2->set((inst.bytes[3] & 0x08) / (1 << 3));
    f7_01->set((inst.bytes[3] & 0x06) / (1 << 1));
    rs2_4->set((inst.bytes[3] & 0x01) / (1 << 0));
    rs2_3->set((inst.bytes[2] & 0x80) / (1 << 7));
    rs2_12->set((inst.bytes[2] & 0x60) / (1 << 5));
    rs2_0->set((inst.bytes[2] & 0x10) / (1 << 4));
    rs1_34->set((inst.bytes[2] & 0x0C) / (1 << 2));
    rs1_12->set((inst.bytes[2] & 0x03) / (1 << 0));
    rs1_0->set((inst.bytes[1] & 0x80) / (1 << 7));
    func3_2->set((inst.bytes[1] & 0x40) / (1 << 6));
    func3_01->set((inst.bytes[1] & 0x30) / (1 << 4));
    rd_34->set((inst.bytes[1] & 0x0C) / (1 << 2));
    rd_12->set((inst.bytes[1] & 0x03) / (1 << 0));
    rd_0->set((inst.bytes[0] & 0x80) / (1 << 7));
    opcode_->set(inst.bytes[0] & 0x7f);
  }
  eq(inst.bytes[3], func7() * 0x02 + rs2_4);
  eq(inst.bytes[2], (rs2_3 * 0x08 + rs2_12 * 0x02 + rs2_0) * 0x10 + rs1_34 * 0x04 + rs1_12);
  eq(inst.bytes[1], rs1_0 * 0x80 + func3() * 0x10 + rd_34 * 0x04 + rd_12);
  eq(inst.bytes[0], rd_0 * 0x80 + opcode_);
}

Val DecoderImpl::rs1() {
  return rs1_34 * 0x08 + rs1_12 * 0x02 + rs1_0;
}

Val DecoderImpl::rs2() {
  return rs2_4 * 0x10 + (rs2_3 * 0x08 + rs2_12 * 0x02 + rs2_0);
}

Val DecoderImpl::rd() {
  return rd_34 * 0x08 + rd_12 * 0x02 + rd_0;
}

Val DecoderImpl::func3() {
  return func3_2 * 0x04 + func3_01;
}

Val DecoderImpl::func7() {
  return f7_6 * 0x40 + func7Low();
}

Val DecoderImpl::func7Low() {
  return f7_45 * 0x10 + f7_3 * 0x08 + f7_2 * 0x04 + f7_01;
}

Val DecoderImpl::opcode() {
  return opcode_;
}

U32Val DecoderImpl::immR() {
  return {0x00, 0x00, 0x00, 0x00};
}

U32Val DecoderImpl::immI() {
  return {
      f7_2 * 0x80 + f7_01 * 0x20 + rs2(),
      0xf8 * f7_6 + f7_45 * 0x02 + f7_3,
      0xff * f7_6,
      0xff * f7_6,
  };
}

U32Val DecoderImpl::immS() {
  return {
      f7_2 * 0x80 + f7_01 * 0x20 + rd(),
      0xf8 * f7_6 + f7_45 * 0x02 + f7_3,
      0xff * f7_6,
      0xff * f7_6,
  };
}

U32Val DecoderImpl::immB() {
  return {
      f7_2 * 0x80 + f7_01 * 0x20 + rd_34 * 0x08 + rd_12 * 0x02,
      0xf0 * f7_6 + rd_0 * 0x08 + f7_45 * 0x02 + f7_3,
      0xff * f7_6,
      0xff * f7_6,
  };
}

U32Val DecoderImpl::immU() {
  return {
      0x00,
      rs1_0 * 0x80 + func3() * 0x10,
      (rs2_3 * 0x08 + rs2_12 * 0x02 + rs2_0) * 0x10 + rs1_34 * 0x04 + rs1_12,
      func7() * 0x02 + rs2_4,
  };
}

U32Val DecoderImpl::immJ() {
  return {
      f7_2 * 0x80 + f7_01 * 0x20 + rs2() - rs2_0,
      rs1_0 * 0x80 + func3() * 0x10 + rs2_0 * 0x08 + f7_45 * 0x02 + f7_3,
      0xf0 * f7_6 + rs1_34 * 0x04 + rs1_12,
      0xff * f7_6,
  };
}

} // namespace zirgen::rv32im_v1
