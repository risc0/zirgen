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

#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/reg.h"

namespace zirgen::rv32im_v1 {

struct GlobalDigestImpl : public CompImpl<GlobalDigestImpl> {
  GlobalDigestImpl() {
    for (size_t i = 0; i < kDigestWords; i++) {
      words.emplace_back(Label("word", i), "out");
    }
  }

  std::vector<U32Reg> words;
};
using GlobalDigest = Comp<GlobalDigestImpl>;

struct SystemStateImpl : public CompImpl<SystemStateImpl> {
  SystemStateImpl() : pc(Label("pc"), "out"), imageId(Label("image_id")) {}

  U32Reg pc;
  GlobalDigest imageId;
};
using SystemState = Comp<SystemStateImpl>;

struct GlobalImpl : public CompImpl<GlobalImpl> {
  GlobalImpl()
      : input(Label("input"))
      , pre(Label("pre"))
      , post(Label("post"))
      , sysExitCode(Label("sys_exit_code"), "out")
      , userExitCode(Label("user_exit_code"), "out")
      , output(Label("output")) {}

  // Input state
  GlobalDigest input;
  SystemState pre;

  // Output state
  SystemState post;
  Reg sysExitCode;
  Reg userExitCode;
  GlobalDigest output;
};
using Global = Comp<GlobalImpl>;

} // namespace zirgen::rv32im_v1
