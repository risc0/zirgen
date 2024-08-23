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

/// Protocol info strings for the proof system and circuits. Used to seed the
/// Fiat-Shamir transcript and provide domain seperation between different
/// protocol and circuit versions.

#pragma once

#include <array>

namespace zirgen {

const uint32_t PROTOCOL_INFO_LEN = 16;

// NOTE: Length of the array is +1 for the null terminator.
typedef std::array<char, PROTOCOL_INFO_LEN + 1> ProtocolInfo;

const ProtocolInfo PROOF_SYSTEM_INFO = {"RISC0_STARK:v1__"};

const ProtocolInfo RECURSION_CIRCUIT_INFO = {"RECURSION:rev1v1"};

const ProtocolInfo FIBONACCI_CIRCUIT_INFO = {"FIBONACCI:rev1v1"};

const ProtocolInfo RV32IM_CIRCUIT_INFO = {"RV32IM:rev1v1___"};

} // namespace zirgen
