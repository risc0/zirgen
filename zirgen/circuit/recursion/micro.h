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

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

struct MicroOpImpl : public CompImpl<MicroOpImpl> {
  void set(MicroInst inst, Val writeAddr, Reg extraPrev, size_t extraBack);

  MicroInst inst = Label("inst");
  OneHot<MICRO_OPCODE_COUNT> decode = {Label("decode"),
                                       Labels({"constop",
                                               "add",
                                               "sub",
                                               "mul",
                                               "inv",
                                               "eq",
                                               "read_iop_header",
                                               "read_iop_body",
                                               "mix_rng",
                                               "select",
                                               "extract"})};
  WomReg in0 = Label("in0");
  WomReg in1 = Label("in1");
  WomReg out = Label("out");
  Reg extra = Label("extra");
};
using MicroOp = Comp<MicroOpImpl>;

struct MicroOpsImpl : public CompImpl<MicroOpsImpl> {
  MicroOpsImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

  WomBody body;
  std::vector<MicroOp> ops;
};
using MicroOps = Comp<MicroOpsImpl>;

} // namespace zirgen::recursion
