// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
