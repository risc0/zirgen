// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "zkp.h"

namespace zirgen {

constexpr size_t kBabyBearExtSize = 4;
constexpr uint64_t kBabyBearP = 15 * (1 << 27) + 1;
constexpr uint64_t kBabyBearToMontgomery = 268435454;
constexpr uint64_t kBabyBearFromMontgomery = 943718400;

inline uint32_t toMontgomery(uint32_t in) {
  return uint64_t(in) * kBabyBearToMontgomery % kBabyBearP;
}
inline uint32_t fromMontgomery(uint32_t in) {
  return uint64_t(in) * kBabyBearFromMontgomery % kBabyBearP;
}

} // namespace zirgen
