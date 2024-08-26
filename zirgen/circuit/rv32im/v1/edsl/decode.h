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

#pragma once

#include "zirgen/components/u32.h"

namespace zirgen::rv32im_v1 {

class DecoderImpl : public CompImpl<DecoderImpl> {
public:
  void set(U32Val inst);

  Val rs1();
  Val rs2();
  Val rd();
  Val func3();
  Val func7();
  Val opcode();
  U32Val immR();
  U32Val immI();
  U32Val immS();
  U32Val immB();
  U32Val immU();
  U32Val immJ();

private:
  Val func7Low();

  // TODO: Full decomposition isn't requried in all / most case, but it's easier this way
  Twit f7_01;
  Bit f7_2;
  Bit f7_3;
  Twit f7_45;
  Bit f7_6;
  Bit rs2_0;
  Twit rs2_12;
  Bit rs2_3;
  Bit rs2_4;
  Bit rs1_0;
  Twit rs1_12;
  Twit rs1_34;
  Twit func3_01;
  Bit func3_2;
  Bit rd_0;
  Twit rd_12;
  Twit rd_34;
  Reg opcode_;
};

using Decoder = Comp<DecoderImpl>;

} // namespace zirgen::rv32im_v1
