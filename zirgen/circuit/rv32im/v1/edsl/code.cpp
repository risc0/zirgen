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

#include "zirgen/circuit/rv32im/v1/edsl/code.h"

#include "zirgen/components/bytes.h"

namespace zirgen::rv32im_v1 {

SetupInfoImpl::SetupInfoImpl() : isLastSetup("code") {}

RamLoadInfoImpl::RamLoadInfoImpl() : startAddr("code") {
  for (size_t i = 0; i < kRamLoadStepIOCount * 2; i++) {
    data.emplace_back("code");
  }
}

ResetInfoImpl::ResetInfoImpl()
    : isFirst("code"), isInit("code"), isOutput("code"), isFini("code") {}

CodeImpl::CodeImpl() : cycle("code"), stepType("code", false), stepInfo(stepType) {}

static constexpr uint32_t kShaK[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static constexpr uint32_t kShaInit[] = {
    0x67e6096a,
    0x85ae67bb,
    0x72f36e3c,
    0x3af54fa5,
    0x7f520e51,
    0x8c68059b,
    0xabd9831f,
    0x19cde05b,
};

std::vector<uint64_t> writeCode(size_t cycles) {
  std::map<uint32_t, uint32_t> data;

  // Setup 'k' for SHA
  for (size_t i = 0; i < kShaKSize; i++) {
    data[kShaKOffset + i] = kShaK[i];
  }

  // Setup SHA-256 Init
  for (size_t i = 0; i < kDigestWords; i++) {
    data[kShaInitOffset + i] = kShaInit[i];
  }

  // Setup SHA-256 Init
  for (size_t i = 0; i < kDigestWords; i++) {
    data[kZerosOffset + i] = 0;
  }

  // Zero fill data as needed to be divisible by kRamLoadStepIOCount
  auto it = data.begin();
  while (it != data.end()) {
    // Pick the next natural value.
    size_t addr = it->first;
    // Make sure at least kRamLoadStepIOCount entries exist past it
    for (size_t i = 1; i < kRamLoadStepIOCount; i++) {
      if (!data.count(addr + i)) {
        data[addr + i] = 0;
      }
    }
    // Walk over the kRamLoadStepIOCount values (which now definitely exist)
    for (size_t i = 0; i < kRamLoadStepIOCount; i++) {
      it++;
    }
  }

  size_t setupCount = BytesSetupImpl::setupCount(kSetupStepRegs);
  size_t infoOffset = 1 + StepType::COUNT;

  // Verify that we have enough space
  size_t minSpace = 1 +                                 // BYTES_INIT
                    setupCount +                        // BYTES_SETUP
                    1 +                                 // RAM_INIT
                    data.size() / kRamLoadStepIOCount + // RamLoad
                    2 +                                 // Reset (isInit)
                    2 +                                 // Reset (!isInit)
                    1 +                                 // RamFini
                    1 +                                 // BytesFini
                    kZKCycles;

  if (cycles < minSpace) {
    llvm::errs() << "Setup count = " << setupCount << "\n";
    llvm::errs() << "data.size() = " << data.size() << "\n";
    llvm::errs() << "cycles = " << cycles << "\n";
    throw std::runtime_error("Not enough space in writeCode");
  }

  // Create buffer to return
  std::vector<uint64_t> code(cycles * kCodeSize);
  size_t cycle = 0;
  auto set = [&](size_t pos, uint64_t val) { code[cycle * kCodeSize + pos] = val; };

  // Write cycle everywhere (except zk area)
  for (size_t i = 0; i < cycles - kZKCycles; i++) {
    set(0, i);
    cycle++;
  }
  // Init
  cycle = 0;
  set(1 + StepType::BYTES_INIT, 1);
  cycle++;
  // Setup
  for (size_t i = 0; i < setupCount; i++) {
    set(1 + StepType::BYTES_SETUP, 1);
    if (i == setupCount - 1) {
      set(infoOffset, 1);
    }
    cycle++;
  }
  // RAM_INIT
  set(1 + StepType::RAM_INIT, 1);
  cycle++;
  // Ram load
  it = data.begin();
  while (it != data.end()) {
    set(1 + StepType::RAM_LOAD, 1);
    set(infoOffset, it->first);
    for (size_t i = 0; i < kRamLoadStepIOCount; i++) {
      set(infoOffset + 1 + 2 * i, it->second & 0xffff);
      set(infoOffset + 1 + 2 * i + 1, it->second >> 16);
      it++;
    }
    cycle++;
  }
  // RESET(init)
  set(1 + StepType::RESET, 1);
  set(infoOffset, 1);     // isFirst
  set(infoOffset + 1, 1); // isInit
  cycle++;
  set(1 + StepType::RESET, 1);
  set(infoOffset, 0);     // isFirst
  set(infoOffset + 1, 1); // isInit
  cycle++;
  // Body
  for (size_t i = 0; i < cycles - minSpace; i++) {
    set(1 + StepType::BODY, 1);
    cycle++;
  }
  // RESET(fini)
  set(1 + StepType::RESET, 1);
  set(infoOffset, 1);     // isFirst
  set(infoOffset + 2, 1); // isOutput
  cycle++;
  set(1 + StepType::RESET, 1);
  set(infoOffset, 0);     // isFirst
  set(infoOffset + 2, 1); // isOutput
  cycle++;
  set(1 + StepType::RESET, 1);
  set(infoOffset, 1);     // isFirst
  set(infoOffset + 3, 1); // isFini
  cycle++;
  set(1 + StepType::RESET, 1);
  set(infoOffset, 0);     // isFirst
  set(infoOffset + 3, 1); // isFini
  cycle++;
  // Ram Fini
  set(1 + StepType::RAM_FINI, 1);
  cycle++;
  // Bytes Fini
  set(1 + StepType::BYTES_FINI, 1);
  cycle++;
  return code;
}

} // namespace zirgen::rv32im_v1
