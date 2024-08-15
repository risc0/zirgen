// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
