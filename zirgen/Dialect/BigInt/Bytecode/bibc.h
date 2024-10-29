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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zirgen::BigInt::Bytecode {

// Representation of BigIntType
struct Type {
  uint64_t coeffs;
  uint64_t maxPos;
  uint64_t maxNeg;
  uint64_t minBits;
};

// Input wire
struct Input {
  uint64_t label;
  uint32_t bitWidth;
  uint16_t minBits;
  bool isPublic;
};

// Code section is a sequence of ops
// Note that ModularInv and Reduce must be lowered beforehand

struct Op {
  enum Code {
    Eqz = 0x0,   // unary: value
    Def = 0x1,   // unary: input index
    Con = 0x2,   // unary: constant index
    Load = 0x3,  // unary: constant index
    Store = 0x4, // unary: constant index
    Add = 0x8,   // binary
    Sub = 0x9,   // binary
    Mul = 0xA,   // binary
    Rem = 0xB,   // binary
    Quo = 0xC,   // binary
    Inv = 0xE,   // binary
  };
  uint32_t code;
  size_t type;
  size_t operandA;
  size_t operandB;
};

// In-memory container
struct Program {
  std::vector<Input> inputs;
  std::vector<Type> types;
  std::vector<uint64_t> constants;
  std::vector<Op> ops;
  void clear();
};

bool operator<(const Type&, const Type&);

} // namespace zirgen::BigInt::Bytecode
