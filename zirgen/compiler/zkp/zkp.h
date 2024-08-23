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

namespace zirgen {

// Generic constants
constexpr size_t kWordSize = sizeof(uint32_t);
constexpr size_t kDigestWords = 8;
constexpr size_t kBlockSize = kDigestWords * 2;
constexpr size_t kDigestBytes = kDigestWords * kWordSize;

// zkp parameters
constexpr size_t kQueries = 50;    // TODO bits of TODO security
constexpr size_t kZKCycles = 1994; // TODO: would be nice to calculate this in code

} // namespace zirgen
