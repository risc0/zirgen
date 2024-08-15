// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
                           Val polyMix) const = 0;
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

} // namespace zirgen::verify
