// Copyright 2025 RISC Zero, Inc.
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

#include <set>

#include "zirgen/Dialect/Zll/Analysis/TapsAnalysis.h"
#include "zirgen/circuit/verify/fri.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen::verify {

constexpr size_t kCheckSize = kBabyBearExtSize * kInvRate;

class CircuitInterface {
public:
  virtual ~CircuitInterface() {}
  virtual const Zll::TapSet& get_taps() const = 0;
  virtual Val compute_poly(llvm::ArrayRef<Val> u,
                           llvm::ArrayRef<Val> out,
                           llvm::ArrayRef<Val> accumMix,
                           Val polyMix,
                           Val z) const = 0;
  virtual size_t out_size() const = 0;
  virtual size_t mix_size() const = 0;
  virtual ProtocolInfo get_circuit_info() const = 0;
};

struct VerifyInfo {
  std::vector<Val> out;
  DigestVal outDigest;
  DigestVal codeRoot;
};

// Returns verified code root.
VerifyInfo verify(ReadIopVal& iop, size_t po2, const CircuitInterface& circuit);

VerifyInfo verifyRecursion(ReadIopVal& allowedRoot,
                           std::vector<ReadIopVal> seals,
                           std::vector<ReadIopVal> alloweds,
                           const CircuitInterface& circuit);

struct GroupInfoV3 {
  size_t globalCount;
  size_t mixCount;
};

class CircuitInterfaceV3 {
public:
  virtual ~CircuitInterfaceV3() {}
  virtual const Zll::TapSet& getTaps() const = 0;
  virtual const llvm::ArrayRef<GroupInfoV3> getGroupInfo() const = 0;
  virtual Val computePolyExt(llvm::ArrayRef<Val> u,
                             llvm::ArrayRef<Val> out,
                             llvm::ArrayRef<Val> accumMix,
                             Val polyMix,
                             Val z) const = 0;
};

// Returns verified code root.
std::vector<Val> verifyV3(ReadIopVal& iop, size_t po2, const CircuitInterfaceV3& circuit);

} // namespace zirgen::verify
