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
