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

#include "zirgen/circuit/rv32im/v1/platform/page_table.h"
#include "zirgen/compiler/edsl/component.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

static constexpr uint32_t kInputRegs = 4;

class TopImpl;
using Top = Comp<TopImpl>;

Val getPageIndex(Val addr);
Val getPageAddr(Val pageIndex);

// A page index is represented in 18-bits
struct PageIndexDiffReg {
  ByteReg low;
  ByteReg mid;
  Twit high;

  void set(Val diff);
  void setZero();
};

class PageFaultCycleImpl : public CompImpl<PageFaultCycleImpl> {
public:
  PageFaultCycleImpl(RamHeader ramHeader);
  void set(Top top);

  RamPass ram;
  Reg stateOut;
  Reg stateIn;

  Bit isRead;
  Bit isDone;
  Reg pageIndex;
  PageIndexDiffReg ltBound;
  PageIndexDiffReg gtBound;
  Reg repeat;
  Reg indexOffset;
  IsZero isRootIndex;
};

using PageFaultCycle = Comp<PageFaultCycleImpl>;

} // namespace zirgen::rv32im_v1
