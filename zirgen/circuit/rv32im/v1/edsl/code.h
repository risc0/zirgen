// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/mux.h"

namespace zirgen::rv32im_v1 {

namespace StepType {

constexpr size_t BYTES_INIT = 0;
constexpr size_t BYTES_SETUP = 1;
constexpr size_t RAM_INIT = 2;
constexpr size_t RAM_LOAD = 3;
constexpr size_t RESET = 4;
constexpr size_t BODY = 5;
constexpr size_t RAM_FINI = 6;
constexpr size_t BYTES_FINI = 7;
constexpr size_t COUNT = 8;

} // namespace StepType

struct EmptyInfoImpl : public CompImpl<EmptyInfoImpl> {};
using EmptyInfo = Comp<EmptyInfoImpl>;

struct SetupInfoImpl : public CompImpl<SetupInfoImpl> {
  SetupInfoImpl();

  Reg isLastSetup;
};
using SetupInfo = Comp<SetupInfoImpl>;

struct RamLoadInfoImpl : public CompImpl<RamLoadInfoImpl> {
  RamLoadInfoImpl();

  Reg startAddr;
  std::vector<Reg> data;
};
using RamLoadInfo = Comp<RamLoadInfoImpl>;

struct ResetInfoImpl : public CompImpl<ResetInfoImpl> {
  ResetInfoImpl();

  Reg isFirst;
  Reg isInit;
  Reg isOutput;
  Reg isFini;
};
using ResetInfo = Comp<ResetInfoImpl>;

struct CodeImpl : public CompImpl<CodeImpl> {
public:
  CodeImpl();

  Reg cycle;
  OneHot<StepType::COUNT> stepType;
  Mux<EmptyInfo, SetupInfo, EmptyInfo, RamLoadInfo, ResetInfo, EmptyInfo, EmptyInfo, EmptyInfo>
      stepInfo;
};
using Code = Comp<CodeImpl>;

std::vector<uint64_t> writeCode(size_t cycles);

} // namespace zirgen::rv32im_v1
