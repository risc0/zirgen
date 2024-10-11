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

#include <array>
#include <stdexcept>
#include "file.h"

namespace zirgen::BigInt::Bytecode {

/*
BIGINT BYTECODE FILE FORMAT
All values are little-endian.
Table elements are all 64-bit aligned and therefore so are sections.

HEADER: 24 bytes
4 | 62 69 62 63   Magic "bibc"
4 | 01 00 00 00   Version 1
4 | number of types
4 | number of inputs
4 | number of constants
4 | number of operations

TYPE TABLE: 32 bytes each
8 | coeffs
8 | maxPos
8 | maxNeg
8 | minBits

INPUT TABLE: 16 bytes each
8 | label
4 | bitWidth
4 | minBits

CONSTANT TABLE: 8 bytes each
8 | word

OPERATIONS: 8 bytes each
 4 bits: opcode
 12 bits: type T
 24 bits: operand A
 24 bits: operand B

OPCODES:
  0: Eqz -  A
  1: Def - TA
  2: Con - TAB
  8: Add - TAB
  9: Sub - TAB
  A: Mul - TAB
  B: Rem - TAB
  C: Quo - TAB
  D: Inv - TAB
*/

IOException::IOException(const char* file, const char* func, int line, const char* msg)
    : std::runtime_error::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                        ", check failed in " + std::string(func) + ": " +
                                        std::string(msg)) {}

#define check(cond)                                                                                \
  if (cond)                                                                                        \
  throw IOException(__FILE__, __FUNCTION__, __LINE__, #cond)

namespace {

void writeU32(uint32_t value, FILE *stream) {
  std::array<uint8_t, 4> buf;
  buf[0] = (value >> 0x00) & 0xFFU;
  buf[1] = (value >> 0x08) & 0xFFU;
  buf[2] = (value >> 0x10) & 0xFFU;
  buf[3] = (value >> 0x18) & 0xFFU;
  size_t writ = fwrite(buf.data(), buf.size(), 1, stream);
  check(ferror(stream) || !writ);
}

void writeU64(uint64_t value, FILE *stream) {
  std::array<uint8_t, 8> buf;
  buf[0] = (value >> 0x00) & 0xFFU;
  buf[1] = (value >> 0x08) & 0xFFU;
  buf[2] = (value >> 0x10) & 0xFFU;
  buf[3] = (value >> 0x18) & 0xFFU;
  buf[4] = (value >> 0x20) & 0xFFU;
  buf[5] = (value >> 0x28) & 0xFFU;
  buf[6] = (value >> 0x30) & 0xFFU;
  buf[7] = (value >> 0x38) & 0xFFU;
  size_t writ = fwrite(buf.data(), buf.size(), 1, stream);
  check(ferror(stream) || !writ);
}

void writeHeader(const Program &p, FILE *stream) {
  writeU32(0x63626962, stream);
  writeU32(1, stream);
  writeU32(p.inputs.size(), stream);
  writeU32(p.types.size(), stream);
  writeU32(p.constants.size(), stream);
  writeU32(p.ops.size(), stream);
}

void writeType(const Type &t, FILE *stream) {
  writeU64(t.coeffs, stream);
  writeU64(t.maxPos, stream);
  writeU64(t.maxNeg, stream);
  writeU64(t.minBits, stream);
}

void writeInput(const Input &i, FILE *stream) {
  writeU64(i.label, stream);
  writeU32(i.bitWidth, stream);
  writeU32(i.minBits, stream);
}

void writeOp(const Op &o, FILE *stream) {
  // Pack operation struct fields into a single 64-bit word
  uint64_t w = 0;
  check(static_cast<uint8_t>(o.code) >= 0x10);
  w |= static_cast<uint64_t>(static_cast<uint8_t>(o.code) & 0x0F) << 60;
  check(o.type >= 0x1000);
  w |= static_cast<uint64_t>(o.type & 0x0FFF) << 48;
  check(o.operandA >= 0x01000000);
  w |= static_cast<uint64_t>(o.operandA & 0x00FFFFFF) << 24;
  check(o.operandB >= 0x01000000);
  w |= static_cast<uint64_t>(o.operandB & 0x00FFFFFF) << 0;
  writeU64(w, stream);
}

} // namespace

void write(const Program &p, FILE *stream) {
  writeHeader(p, stream);
  // inputs referenced through 24-bit operand
  check(p.inputs.size() > 0x00FFFFFF);
  for (auto &i: p.inputs) {
    writeInput(i, stream);
  }
  // types referenced through 12-bit operand
  check(p.types.size() > 0x00000FFF);
  for (auto &t: p.types) {
    writeType(t, stream);
  }
  // constants referenced through 24-bit operand
  check(p.constants.size() > 0x00FFFFFF);
  for (uint64_t c: p.constants) {
    writeU64(c, stream);
  }
  // op results referenced through 24-bit operand
  check(p.ops.size() > 0x00FFFFFF);
  for (auto &o: p.ops) {
    writeOp(o, stream);
  }
}

void read(Program &p, FILE *stream) {
  p.clear();
  throw std::runtime_error("not yet implemented");
}

} // namespace zirgen::BigInt::Bytecode
