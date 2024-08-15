// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
