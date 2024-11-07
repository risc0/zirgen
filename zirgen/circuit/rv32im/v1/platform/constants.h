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

// Generic constants

namespace zirgen::rv32im_v1 {

constexpr size_t kWordSize = sizeof(uint32_t);
constexpr size_t kDigestWords = 8;
constexpr size_t kBlockSize = kDigestWords * 2;
constexpr size_t kDigestBytes = kDigestWords * kWordSize;

// Size of each of the buffers
constexpr size_t kCodeSize = 16;
constexpr size_t kSystemStateSize = kWordSize + kDigestBytes;
constexpr size_t kInOutSize = kSystemStateSize * 2 + kDigestBytes * 2 + 2;
constexpr size_t kDataSize = 226;
constexpr size_t kMixSize = 40;
constexpr size_t kAccumSize = 52;

constexpr size_t kSetupStepRegs = 84;
constexpr size_t kRamLoadStepIOCount = 3;

constexpr size_t kBytesNeededPerRamIO = 3;
constexpr size_t kTwitsNeededPerRamIO = 1;

constexpr size_t kNumBodyBytes = 34;
constexpr size_t kNumBodyTwits = 20;

constexpr size_t kMinorMuxSize = 8;

namespace MajorType {

constexpr size_t kCompute0 = 0;
constexpr size_t kCompute1 = 1;
constexpr size_t kCompute2 = 2;
constexpr size_t kMemIO = 3;
constexpr size_t kMultiply = 4;
constexpr size_t kDivide = 5;
constexpr size_t kVerifyAnd = 6;
constexpr size_t kVerifyDivide = 7;
constexpr size_t kECall = 8;
constexpr size_t kShaInit = 9;
constexpr size_t kShaLoad = 10;
constexpr size_t kShaMain = 11;
constexpr size_t kPageFault = 12;
constexpr size_t kECallCopyIn = 13;
constexpr size_t kBigInt = 14;
constexpr size_t kBigInt2 = 15;
constexpr size_t kHalt = 16;
constexpr size_t kMuxSize = 17;

} // namespace MajorType

// Machine ECalls, all user ecalls jump to dispatch address
namespace ECallType {

constexpr size_t kHalt = 0;
constexpr size_t kInput = 1;
constexpr size_t kSoftware = 2;
constexpr size_t kSha = 3;
constexpr size_t kBigInt = 4;
constexpr size_t kUser = 5;
constexpr size_t kBigInt2 = 6;
constexpr size_t kMuxSize = 7;

} // namespace ECallType

namespace HaltType {

constexpr size_t kTerminate = 0;
constexpr size_t kPause = 1;
constexpr size_t kSystemSplit = 2;
constexpr size_t kMuxSize = 3;

} // namespace HaltType

constexpr size_t kPageSize = 1024;

// Memory addresses are 28 bits long (for byte based addresses) or 26 bits long (for word based
// addresses). The area of memory with the top two bits == 1 (i.e. the top 1/4 of memory) is
// reserved for system memory.
constexpr size_t kSystemAddr = 0x0C000000 / kWordSize;

// Registers live in the beginning of system memory
constexpr size_t kRegisterOffset = kSystemAddr;
constexpr size_t kRegisterSize = 32;
constexpr size_t kUserRegisterOffset = kSystemAddr - kRegisterSize;

// Memory addresses of registers by name
namespace RegAddr {

constexpr size_t kZero = kRegisterOffset + 0;

constexpr size_t kT0 = kRegisterOffset + 5;
constexpr size_t kT1 = kRegisterOffset + 6;
constexpr size_t kT2 = kRegisterOffset + 7;

constexpr size_t kA0 = kRegisterOffset + 10;
constexpr size_t kA1 = kRegisterOffset + 11;
constexpr size_t kA2 = kRegisterOffset + 12;
constexpr size_t kA3 = kRegisterOffset + 13;
constexpr size_t kA4 = kRegisterOffset + 14;
constexpr size_t kA5 = kRegisterOffset + 15;
constexpr size_t kA6 = kRegisterOffset + 16;
constexpr size_t kA7 = kRegisterOffset + 17;

} // namespace RegAddr

// Memory addresses of registers by name (user version)
namespace UserRegAddr {

constexpr size_t kZero = kUserRegisterOffset + 0;

constexpr size_t kT0 = kUserRegisterOffset + 5;
constexpr size_t kT1 = kUserRegisterOffset + 6;
constexpr size_t kT2 = kUserRegisterOffset + 7;

constexpr size_t kA0 = kUserRegisterOffset + 10;
constexpr size_t kA1 = kUserRegisterOffset + 11;
constexpr size_t kA2 = kUserRegisterOffset + 12;
constexpr size_t kA3 = kUserRegisterOffset + 13;
constexpr size_t kA4 = kUserRegisterOffset + 14;
constexpr size_t kA5 = kUserRegisterOffset + 15;
constexpr size_t kA6 = kUserRegisterOffset + 16;
constexpr size_t kA7 = kUserRegisterOffset + 17;

} // namespace UserRegAddr

// Constants for the BigInt arithmatic circuit.
namespace BigInt {

// BigInt width that is handled by this multiplier.
constexpr size_t kBitWidth = 256;
constexpr size_t kByteWidth = BigInt::kBitWidth / 8;
constexpr size_t kWordWidth = BigInt::kByteWidth / kWordSize;

// Resource allocations.
constexpr size_t kIoSize = 4;
constexpr size_t kBytesSize = 16;
constexpr size_t kMulBufferSize = 32;
constexpr size_t kCarryHiSize = 8;
constexpr size_t kMulInSize = kMulBufferSize / 2;
constexpr size_t kMulOutSize = kMulBufferSize;

// Timing information.
constexpr size_t kCyclesPerStage = 2;
constexpr size_t kStages = 5;

} // namespace BigInt

namespace PolyOp {
// Every poly op begins with 'newPoly = poly + newData'
// Then what happens next is:
// For all memory types
constexpr size_t kEnd = 0;     // poly' = newPoly, term' = term, tot' = tot
constexpr size_t kOpShift = 1; // poly' = newPoly * x^16, term' = term, tot' = tot
// For 'read/write'
constexpr size_t kOpSetTerm = 2; // poly' = 0, term' = newPoly, tot' = tot
constexpr size_t kOpAddTot = 3;  // poly' = 0, term' = 1, tot' = tot + coeff * newPoly * term
// For 'check'
constexpr size_t kOpCarry1 = 4; // poly' = newPoly * 256, term' = term, tot' = tot
constexpr size_t kOpCarry2 = 5; // poly' = newPoly * 64, term' = term, tot' = tot
constexpr size_t kOpEqz =
    6; // poly' = 0, term' = 1, tot' = 0, assert tot + (z - 256) * newPoly == 0
} // namespace PolyOp

constexpr size_t kUserPC = 0x0BFFFF00;
constexpr size_t kECallEntry = 0x0BFFFF04;
constexpr size_t kTrapEntry = 0x0BFFFF08;

// Location of page table. This contains a merkle tree based on the SHA256 hash of each page in
// memory.
constexpr size_t kPageTableAddr = 0x0D000000 / kWordSize;

// SHA-256 mixing constant; 64 words.
constexpr size_t kShaKOffset = 0x0D700000 / kWordSize;
constexpr size_t kShaKSize = 64;
// SHA-256 Initialization vector: 8 words.
constexpr size_t kShaInitOffset = kShaKOffset + kShaKSize;
// 8 words of zeros (used in system split halt)
constexpr size_t kZerosOffset = kShaInitOffset + kDigestWords;

// Maximum degree the circuit is allowed to be.
constexpr size_t kMaxDegree = 5;

// Number of words in each cycle received using the SOFTWARE ecall
constexpr size_t kIoChunkWords = 4;

} // namespace zirgen::rv32im_v1
