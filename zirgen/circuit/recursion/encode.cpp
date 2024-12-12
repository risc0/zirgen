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

#include "zirgen/circuit/recursion/encode.h"

#include "zirgen/compiler/zkp/baby_bear.h"
#include "zirgen/compiler/zkp/sha256.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen::recursion {

namespace {

struct MicroData {
  MicroOpcode opcode;
  uint64_t operands[3];
};

struct MacroData {
  MacroOpcode opcode;
  uint64_t operands[3];
};

struct CheckedBytesData {
  uint64_t evalPt;
  uint64_t keepCoeffs;
  uint64_t keepUpperState;
  uint64_t prepFull;
};

struct Poseidon2MemData {
  uint64_t doMont;
  uint64_t keepState;
  uint64_t keepUpperState;
  uint64_t prepFull;
  uint64_t group;
  uint64_t inputs[8];
};

struct Poseidon2FullData {
  uint64_t cycle;
};

struct InstData {
  OpType opType;
  uint64_t writeAddr;
  union {
    std::array<MicroData, 3> micro;
    MacroData macro;
    Poseidon2MemData poseidon2Mem;
    Poseidon2FullData poseidon2Full;
    CheckedBytesData checkedBytes;
  } data;
};

struct Instructions;

struct IRng {
  virtual ~IRng() = default;
  virtual uint64_t generateBits(Instructions& insts, size_t bits) = 0;
  virtual uint64_t generateFp(Instructions& insts) = 0;
  virtual void mix(Instructions& insts, uint64_t digest) = 0;
};

struct ShaRng : public IRng {
  // Generate the constants needed for ShaRng
  static void setConstants(Instructions& insts);

  // Create a new Rng
  ShaRng(Instructions& insts);

  // Low level helpers
  void step(Instructions& insts);
  uint64_t generate(Instructions& insts);

  // Implement the interface
  uint64_t generateBits(Instructions& insts, size_t bits) override;
  uint64_t generateFp(Instructions& insts) override;
  void mix(Instructions& insts, uint64_t digest) override;

  // Data
  uint64_t pool0;
  uint64_t pool1;
  size_t poolUsed;
};

struct Poseidon2Rng : public IRng {
  Poseidon2Rng(Instructions& insts);

  uint64_t generateBits(Instructions& insts, size_t bits) override;
  uint64_t generateFp(Instructions& insts) override;
  void mix(Instructions& insts, uint64_t digest) override;

  bool isInit;
  uint64_t curState;
  uint64_t stateUsed;
};

// Does everything like SHA, but needs to convert Poseidon2 input to SHA
struct MixedPoseidon2ShaRng : public ShaRng {
  MixedPoseidon2ShaRng(Instructions& insts) : ShaRng(insts) {}
  void mix(Instructions& insts, uint64_t digest) override;
};

struct Instructions {
  HashType hashType;
  uint64_t nextOut;
  size_t microUsed;
  std::vector<InstData> data;
  llvm::DenseMap<Value, uint64_t> toId;
  std::vector<uint64_t> shaInit;
  std::vector<uint64_t> shaK;
  size_t shaUsed = 0;
  size_t poseidon2Used = 0;
  llvm::DenseMap<Value, std::unique_ptr<IRng>> rngs;
  llvm::DenseMap<std::pair<uint32_t, uint32_t>, uint64_t> consts;
  // TODO: We should come up with a better way to do constants
  uint64_t fp4Rot1;
  uint64_t fp4Rot2;
  uint64_t fp4Rot3;
  uint64_t div2to16Const;
  uint64_t padShaEndConst;
  uint64_t padShaCountConst;
  uint64_t shaRngConsts;
  llvm::StringMap<uint64_t> tagConsts;

  Instructions(HashType hashType) : hashType(hashType), nextOut(1), microUsed(0) {
    addMacro(/*outs=*/0, MacroOpcode::WOM_INIT);
    // Make: [0, 1, 0, 0], [0, 0, 1, 0], and [0, 0, 0, 1]
    fp4Rot1 = addConst(0, 1);
    fp4Rot2 = addMicro(Value(), MicroOpcode::MUL, fp4Rot1, fp4Rot1);
    fp4Rot3 = addMicro(Value(), MicroOpcode::MUL, fp4Rot2, fp4Rot1);
    // 1 / 2^16 in Baby Bear, but also shift to second component
    div2to16Const = addConst(0, 2013235201);
    uint32_t shaInitVals[] = {0x6a09e667,
                              0xbb67ae85,
                              0x3c6ef372,
                              0xa54ff53a,
                              0x510e527f,
                              0x9b05688c,
                              0x1f83d9ab,
                              0x5be0cd19};
    for (size_t i = 0; i < 8; i++) {
      shaInit.push_back(addHalfsConst(shaInitVals[i]));
    }
    uint32_t shaKVals[] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                           0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                           0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                           0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                           0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                           0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                           0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                           0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                           0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                           0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                           0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    for (size_t i = 0; i < 64; i++) {
      shaK.push_back(addHalfsConst(shaKVals[i]));
    }
    ShaRng::setConstants(*this);
  }

  IRng* getRng(Value val) {
    if (!rngs.count(val)) {
      switch (hashType) {
      case HashType::SHA256:
        rngs[val] = std::make_unique<ShaRng>(*this);
        break;
      case HashType::POSEIDON2:
        rngs[val] = std::make_unique<Poseidon2Rng>(*this);
        break;
      case HashType::MIXED_POSEIDON2_SHA:
        rngs[val] = std::make_unique<MixedPoseidon2ShaRng>(*this);
        break;
      }
    }
    return rngs[val].get();
  }

  void finalize() { addMacro(/*outs=*/0, MacroOpcode::WOM_FINI); }

  uint64_t addConst(uint32_t a, uint32_t b = 0) {
    auto key = std::make_pair(a, b);
    auto it = consts.find(key);
    if (it == consts.end()) {
      uint64_t outId = addMicro(Value(), MicroOpcode::CONST, a, b);
      consts[key] = outId;
      return outId;
    }
    return it->second;
  }

  uint64_t addHalfsConst(uint32_t tot) { return addConst(tot & 0xffff, tot >> 16); }

  uint64_t
  addMicro(Value out, MicroOpcode opcode, uint64_t op0 = 0, uint64_t op1 = 0, uint64_t op2 = 0) {
    if (microUsed == 0) {
      data.emplace_back();
      data.back().opType = OpType::MICRO;
      data.back().writeAddr = nextOut;
    }
    size_t outId = nextOut++;
    if (out) {
      toId[out] = outId;
    }
    data.back().data.micro[microUsed].opcode = opcode;
    data.back().data.micro[microUsed].operands[0] = op0;
    data.back().data.micro[microUsed].operands[1] = op1;
    data.back().data.micro[microUsed].operands[2] = op2;
    microUsed++;
    if (microUsed == 3) {
      microUsed = 0;
    }
    return outId;
  }

  void finishMicros() {
    while (microUsed) {
      addMicro(Value(), MicroOpcode::CONST, 0, 0);
    }
  }

  uint64_t
  addMacro(size_t outs, MacroOpcode opcode, uint64_t op0 = 0, uint64_t op1 = 0, uint64_t op2 = 0) {
    finishMicros();
    data.emplace_back();
    data.back().opType = OpType::MACRO;
    data.back().data.macro.opcode = opcode;
    data.back().data.macro.operands[0] = op0;
    data.back().data.macro.operands[1] = op1;
    data.back().data.macro.operands[2] = op2;
    uint64_t writeAddr = nextOut;
    data.back().writeAddr = writeAddr;
    nextOut += outs;
    return writeAddr;
  }

  uint64_t addCheckedBytes(uint64_t evalPt,
                           uint64_t keepCoeffs,
                           uint64_t keepUpperState,
                           uint64_t prepFull) {
    finishMicros();
    poseidon2Used++;
    data.emplace_back();
    data.back().opType = OpType::CHECKED_BYTES;
    data.back().data.checkedBytes.evalPt = evalPt;
    data.back().data.checkedBytes.keepCoeffs = keepCoeffs;
    data.back().data.checkedBytes.keepUpperState = keepUpperState;
    data.back().data.checkedBytes.prepFull = prepFull;
    uint64_t writeAddr = nextOut;
    data.back().writeAddr = writeAddr;
    nextOut += 1; // Write exactly one thing (evaluated point)
    return writeAddr;
  }

  void addPoseidon2Load(uint64_t doMont,
                        uint64_t keepState,
                        uint64_t keepUpperState,
                        uint64_t prepFull,
                        uint64_t group,
                        llvm::ArrayRef<uint64_t> inputs) {
    assert(inputs.size() == 8);
    poseidon2Used++;
    finishMicros();
    data.emplace_back();
    data.back().opType = OpType::POSEIDON2_LOAD;
    data.back().data.poseidon2Mem.doMont = doMont;
    data.back().data.poseidon2Mem.keepState = keepState;
    data.back().data.poseidon2Mem.keepUpperState = keepUpperState;
    data.back().data.poseidon2Mem.prepFull = prepFull;
    data.back().data.poseidon2Mem.group = group;
    for (size_t i = 0; i < 8; i++) {
      data.back().data.poseidon2Mem.inputs[i] = inputs[i];
    }
    data.back().writeAddr = nextOut;
  }

  void addPoseidon2Full(uint64_t cycle) {
    poseidon2Used++;
    finishMicros();
    data.emplace_back();
    data.back().opType = OpType::POSEIDON2_FULL;
    data.back().data.poseidon2Full.cycle = cycle;
    data.back().writeAddr = nextOut;
  }

  void addPoseidon2Partial() {
    poseidon2Used++;
    finishMicros();
    data.emplace_back();
    data.back().opType = OpType::POSEIDON2_PARTIAL;
    data.back().writeAddr = nextOut;
  }

  uint64_t addPoseidon2Store(uint64_t doMont, uint64_t group) {
    poseidon2Used++;
    finishMicros();
    data.emplace_back();
    data.back().opType = OpType::POSEIDON2_STORE;
    data.back().data.poseidon2Mem.doMont = doMont;
    data.back().data.poseidon2Mem.group = group;
    uint64_t writeAddr = nextOut;
    data.back().writeAddr = writeAddr;
    nextOut += 8;
    return writeAddr;
  }

  void doShaInit() {
    for (size_t i = 0; i < 4; i++) {
      shaUsed++;
      addMacro(/*outs=*/0, MacroOpcode::SHA_INIT, shaInit[3 - i], shaInit[3 - i + 4]);
    }
  }

  void doShaLoad(llvm::ArrayRef<uint64_t> values, uint64_t subtype) {
    for (size_t i = 0; i < 16; i++) {
      shaUsed++;
      addMacro(/*outs=*/0, MacroOpcode::SHA_LOAD, values[i], shaK[i], subtype);
    }
  }

  void doShaMix() {
    for (size_t i = 0; i < 48; i++) {
      shaUsed++;
      addMacro(/*outs=*/0, MacroOpcode::SHA_MIX, 0, shaK[16 + i]);
    }
  }
  uint64_t doShaFini() {
    uint64_t out = nextOut;
    for (size_t i = 0; i < 4; i++) {
      shaUsed++;
      addMacro(/*outs=*/0, MacroOpcode::SHA_FINI, out + 3 - i, out + 7 - i);
    }
    nextOut += 8;
    return out;
  }

  uint64_t doSha(llvm::ArrayRef<uint64_t> values, uint64_t subtype) {
    doShaInit();
    uint64_t ret = 0;
    for (size_t i = 0; i < values.size() / 16; i++) {
      doShaLoad(values.slice(i * 16, 16), subtype);
      doShaMix();
      ret = doShaFini();
    }
    return ret;
  }

  uint64_t doShaFold(uint64_t lhs, uint64_t rhs) {
    std::vector<uint64_t> ids(16);
    for (size_t i = 0; i < 8; i++) {
      ids[i] = lhs + i;
      ids[i + 8] = rhs + i;
    }
    return doSha(ids, 1);
  }

  uint64_t doIntoDigestShaBytes(llvm::ArrayRef<uint64_t> bytes) {
    // We keep things in low / high form right until the end so that the final adds are
    // all contiguous since all the 'digest' stuff assumes digests are always contiguous.
    std::vector<uint64_t> low;
    std::vector<uint64_t> high;
    for (size_t i = 0; i < 8; i++) {
      uint64_t t1 = addMicro(Value(), MicroOpcode::MUL, bytes[i * 4 + 1], addConst(256));
      uint64_t t3 = addMicro(Value(), MicroOpcode::MUL, bytes[i * 4 + 3], addConst(256));
      low.push_back(addMicro(Value(), MicroOpcode::ADD, bytes[i * 4 + 0], t1));
      uint64_t high_sum = addMicro(Value(), MicroOpcode::ADD, bytes[i * 4 + 2], t3);
      high.push_back(addMicro(Value(), MicroOpcode::MUL, high_sum, fp4Rot1));
    }
    uint64_t ret = nextOut;
    for (size_t i = 0; i < 8; i++) {
      addMicro(Value(), MicroOpcode::ADD, low[i], high[i]);
    }
    return ret;
  }

  uint64_t doIntoDigestShaWords(llvm::ArrayRef<uint64_t> words) {
    // We keep things in low / high form right until the end so that the final adds are
    // all contiguous since all the 'digest' stuff assumes digests are always contiguous.
    std::vector<uint64_t> low;
    std::vector<uint64_t> high;
    for (size_t i = 0; i < 8; i++) {
      low.push_back(words[i * 2 + 0]);
      high.push_back(addMicro(Value(), MicroOpcode::MUL, words[i * 2 + 1], fp4Rot1));
    }
    uint64_t ret = nextOut;
    for (size_t i = 0; i < 8; i++) {
      addMicro(Value(), MicroOpcode::ADD, low[i], high[i]);
    }
    return ret;
  }

  uint64_t doShaTag(llvm::StringRef tag) {
    if (tagConsts.count(tag)) {
      return tagConsts.find(tag)->second;
    } else {
      uint64_t out = nextOut;
      tagConsts[tag] = nextOut;
      Digest tagDigest = shaHash(tag.str());
      for (size_t i = 0; i < 8; i++) {
        addMicro(
            Value(), MicroOpcode::CONST, tagDigest.words[i] & 0xffff, tagDigest.words[i] >> 16);
      }
      return out;
    }
  }

  uint64_t doTaggedStruct(uint64_t tag,
                          llvm::ArrayRef<uint64_t> digests,
                          llvm::ArrayRef<DigestKind> digestTypes,
                          llvm::ArrayRef<uint64_t> vals) {
    std::vector<uint64_t> words;

    for (size_t i = 0; i < 8; i++) {
      words.push_back(tag + i);
    }
    for (size_t i = 0; i < digests.size(); i++) {
      auto kind = digestTypes[i];
      if (kind == DigestKind::Default) {
        kind = (hashType == HashType::SHA256) ? DigestKind::Sha256 : DigestKind::Poseidon2;
      }
      if (kind == DigestKind::Sha256) {
        // SHA-256 Vals are already in the correct representation.
        for (size_t j = 0; j < 8; j++) {
          words.push_back(digests[i] + j);
        }
      } else if (kind == DigestKind::Poseidon2) {
        // Poseidon2 vals need to be broken up into the word representation.
        std::vector<uint64_t> digestVals;
        for (size_t j = 0; j < 8; j++) {
          digestVals.push_back(digests[i] + j);
        }
        taggedStructPushVals(words, digestVals);
      } else {
        throw std::runtime_error("Invalid kind for doTaggedStruct");
      }
    }

    taggedStructPushVals(words, vals);

    // Terminate with taggedStruct digests count, then SHA2 padding + bit count.
    size_t bitCount = words.size() * 32 + 16;
    uint32_t finalWord = 0x00800000 | digests.size();
    words.push_back(addHalfsConst(finalWord));
    if (words.size() % 16 == 15) {
      words.push_back(addConst(0));
    }
    while (words.size() % 16 != 15) {
      words.push_back(addConst(0));
    }
    bitCount = (bitCount & 0x0000FFFF) << 16 | (bitCount & 0xFFFF0000) >> 16;
    bitCount = (bitCount & 0x00FF00FF) << 8 | (bitCount & 0xFF00FF00) >> 8;
    words.push_back(addHalfsConst(bitCount));

    return doSha(words, 1);
  }

  // Break the given array of vals, turn them into the SHA256 word
  // representation, with 16 bits in each of the two low components of an
  // extension field element.
  void taggedStructPushVals(std::vector<uint64_t>& words, llvm::ArrayRef<uint64_t> vals) {
    // Get low 16 bits of each value (done in a loop for better packing)
    std::vector<uint64_t> lowVals;
    for (size_t i = 0; i < vals.size(); i++) {
      lowVals.push_back(addMacro(/*outs=*/1, MacroOpcode::BIT_AND_ELEM, vals[i], addConst(0xffff)));
    }
    // Now, transform to word format
    for (size_t i = 0; i < vals.size(); i++) {
      // Subtract low bits
      uint64_t highSub = addMicro(Value(), MicroOpcode::SUB, vals[i], lowVals[i]);
      // Divide high by 2^16 and move to high words
      uint64_t highAdjust = addMicro(Value(), MicroOpcode::MUL, highSub, div2to16Const);
      // Add back in low bits + push
      words.push_back(addMicro(Value(), MicroOpcode::ADD, lowVals[i], highAdjust));
    }
  }

  std::pair<uint64_t, std::vector<uint64_t>> doHashCheckedBytes(uint64_t evalPt, uint64_t count) {
    if (!count) {
      // Special case for 0 outputs
      return {doPoseidon2({}), {}};
    }

    // Load a series of bytes, check that they are truly bytes, create degree 16 polynomials with
    // each byte as one term's coefficient, evaluate these polynomials at `eval_pt`, and hash all
    // these byte coeffs with Poseidon2.
    std::vector<uint64_t> evals;
    uint32_t groupCount = 0; // How many in current group (0, 1, 2)
    for (size_t i = 0; i < count; i++) {
      bool lastInGroup = groupCount == 2 || i == count - 1;
      evals.push_back(addCheckedBytes(
          evalPt, /*keepCoeffs=*/(groupCount != 0), /*keepUpperState=*/(i != 0), lastInGroup));
      groupCount++;
      if (lastInGroup) {
        addPoseidon2Full(0);
        addPoseidon2Full(1);
        addPoseidon2Partial();
        addPoseidon2Full(2);
        addPoseidon2Full(3);
        groupCount = 0;
      }
    }
    auto poseidon = addPoseidon2Store(/*doMont=*/1, /*group=*/0);
    return {poseidon, evals};
  }

  std::tuple<uint64_t, uint64_t, std::vector<uint64_t>> doHashCheckedBytesPublic(uint64_t evalPt,
                                                                                 uint64_t count) {
    if (!count)
      throw std::runtime_error("Cannont publically hash empty checked bytes");

    std::vector<uint64_t> evals;
    std::vector<uint64_t> bytes;
    uint64_t zero = addConst(0, 0);
    std::vector<uint64_t> zeros(8, zero);
    for (size_t i = 0; i < count; i++) {
      evals.push_back(
          addCheckedBytes(evalPt, /*keepCoeffs=*/0, /*keepUpperState=*/(i != 0), /*prepFull=*/0));
      uint64_t b0 = addPoseidon2Store(/*doMont=*/0, /*group=*/0);
      addPoseidon2Store(/*doMont=*/0, /*group=*/1);
      for (size_t i = 0; i < 16; i++) {
        bytes.push_back(b0 + i);
      }
      addPoseidon2Load(
          /*doMont=*/0,
          /*keepState*/ 1,
          /*keepUpperState*/ 1,
          /*prepFull=*/1,
          /*group=*/0,
          zeros);
      addPoseidon2Full(0);
      addPoseidon2Full(1);
      addPoseidon2Partial();
      addPoseidon2Full(2);
      addPoseidon2Full(3);
    }
    auto poseidon = addPoseidon2Store(/*doMont=*/1, /*group=*/0);
    std::vector<uint64_t> mulBy;
    mulBy.push_back(addConst(1, 0));
    mulBy.push_back(addConst(256, 0));
    mulBy.push_back(addConst(0, 1));
    mulBy.push_back(addConst(0, 256));
    std::vector<uint64_t> coeffs;
    for (size_t i = 0; i < bytes.size() / 4; i++) {
      uint64_t tot = bytes[4 * i];
      for (size_t j = 1; j < 4; j++) {
        uint64_t prod = addMicro(Value(), MicroOpcode::MUL, bytes[4 * i + j], mulBy[j]);
        tot = addMicro(Value(), MicroOpcode::ADD, tot, prod);
      }
      coeffs.push_back(tot);
    }
    while (coeffs.size() % 16 != 0) {
      coeffs.push_back(zero);
    }
    auto sha = doSha(coeffs, 1);
    return {poseidon, sha, evals};
  }

  uint64_t doPoseidon2(llvm::ArrayRef<uint64_t> values) {
    if (values.empty()) {
      auto psuite = poseidon2HashSuite();
      auto hashVal = psuite->hash(nullptr, 0);
      std::vector<uint64_t> elems;
      for (auto elem : hashVal.words) {
        elems.push_back(addConst(elem));
      }
      return doIntoDigestPoseidon2(elems);
    }
    uint64_t keepUpperState = 0;
    for (size_t i = 0; i < values.size() / 16; i++) {
      addPoseidon2Load(
          /*doMont=*/0,
          /*keepState*/ 0,
          keepUpperState,
          /*prepFull=*/0,
          /*group=*/0,
          values.slice(i * 16, 8));
      keepUpperState = 1;
      addPoseidon2Load(
          /*doMont=*/0,
          /*keepState*/ 1,
          keepUpperState,
          /*prepFull=*/1,
          /*group=*/1,
          values.slice(i * 16 + 8, 8));
      addPoseidon2Full(0);
      addPoseidon2Full(1);
      addPoseidon2Partial();
      addPoseidon2Full(2);
      addPoseidon2Full(3);
    }
    return addPoseidon2Store(/*doMont=*/1, /*group=*/0);
  }

  uint64_t doPoseidon2Fold(uint64_t lhs, uint64_t rhs) {
    std::vector<uint64_t> ids(16);
    for (size_t i = 0; i < 8; i++) {
      ids[i] = lhs + i;
      ids[i + 8] = rhs + i;
    }
    addPoseidon2Load(
        /*doMont=*/1,
        /*keepState=*/0,
        /*keepUpperState=*/0,
        /*prepFull=*/0,
        /*group=*/0,
        llvm::ArrayRef<uint64_t>(ids).slice(0, 8));
    addPoseidon2Load(
        /*doMont=*/1,
        /*keepState=*/1,
        /*keepUpperState=*/0,
        /*prepFull=*/1,
        /*group=*/1,
        llvm::ArrayRef<uint64_t>(ids).slice(8, 8));
    addPoseidon2Full(0);
    addPoseidon2Full(1);
    addPoseidon2Partial();
    addPoseidon2Full(2);
    addPoseidon2Full(3);
    return addPoseidon2Store(/*doMont=*/1, /*group=*/0);
  }

  uint64_t doIntoDigestPoseidon2(llvm::ArrayRef<uint64_t> words) {
    // Do pointless adds to make all the words land in sequential spots
    uint64_t ret = nextOut;
    size_t pad = words.size() / 8 - 1;
    size_t offset = 0;
    for (size_t i = 0; i < 8; i++) {
      addMicro(Value(), MicroOpcode::ADD, words[offset++], 0);
      offset += pad;
    }
    return ret;
  }

  void addInst(Operation& op) {
    TypeSwitch<Operation*>(&op)
        .Case<Zll::ExternOp>([&](Zll::ExternOp op) {
          if (op.getName() == "write") {
            for (Value v : op.getIn()) {
              // Writes are encoded as an 'add' with the final param set to 1
              addMicro(Value(), MicroOpcode::ADD, toId[v], 0, 1);
            }
            return;
          }
          if (op.getName() != "log") {
            llvm::errs() << "Warning: discarding non-log extern " << op;
          }
        })
        .Case<ConstOp>([&](ConstOp op) {
          auto ref = op.getCoefficients();
          if (ref.size() == 1) {
            toId[op.getOut()] = addConst(ref[0]);
          } else {
            assert(ref.size() == kBabyBearExtSize);
            size_t low2 = addConst(ref[0], ref[1]);
            size_t high2 = addConst(ref[2], ref[3]);
            size_t mul = addMicro(Value(), MicroOpcode::MUL, high2, fp4Rot2);
            addMicro(op.getOut(), MicroOpcode::ADD, mul, low2);
          }
        })
        .Case<AddOp>([&](AddOp op) {
          addMicro(op.getOut(), MicroOpcode::ADD, toId[op.getLhs()], toId[op.getRhs()]);
        })
        .Case<SubOp>([&](SubOp op) {
          addMicro(op.getOut(), MicroOpcode::SUB, toId[op.getLhs()], toId[op.getRhs()]);
        })
        .Case<MulOp>([&](MulOp op) {
          addMicro(op.getOut(), MicroOpcode::MUL, toId[op.getLhs()], toId[op.getRhs()]);
        })
        .Case<BitAndOp>([&](BitAndOp op) {
          // Bitwise AND as if it's an element: [a, 0, 0, 0] & [b, 0, 0, 0] -> [a & b, 0, 0, 0]
          toId[op.getOut()] =
              addMacro(/*outs=*/1, MacroOpcode::BIT_AND_ELEM, toId[op.getLhs()], toId[op.getRhs()]);
        })
        .Case<IsZeroOp>(
            [&](IsZeroOp op) { addMicro(op.getOut(), MicroOpcode::INV, toId[op.getIn()], 0); })
        .Case<NegOp>(
            [&](NegOp op) { addMicro(op.getOut(), MicroOpcode::SUB, 0, toId[op.getIn()]); })
        .Case<InvOp>(
            [&](InvOp op) { addMicro(op.getOut(), MicroOpcode::INV, toId[op.getIn()], 1); })
        .Case<EqualZeroOp>(
            [&](EqualZeroOp op) { addMicro(Value(), MicroOpcode::EQ, toId[op.getIn()], 0); })
        .Case<HashOp>([&](HashOp op) {
          size_t k = 0;
          if (!op.getIn().empty())
            k = cast<ValType>(op.getIn()[0].getType()).getFieldK();
          size_t size = op.getIn().size();
          std::vector<uint64_t> ids(k * size);
          for (size_t i = 0; i < size; i++) {
            Value val = op.getIn()[i];
            std::vector<uint64_t> poly;
            uint64_t fullId = toId[val];
            if (k == 1) {
              poly.push_back(fullId);
            } else {
              assert(k == kBabyBearExtSize);
              for (size_t j = 0; j < kBabyBearExtSize; j++) {
                poly.push_back(addMicro(Value(), MicroOpcode::EXTRACT, fullId, j / 2, j % 2));
              }
            }
            for (size_t j = 0; j < k; j++) {
              if (op.getFlip()) {
                ids[i * k + j] = poly[j];
              } else {
                ids[j * size + i] = poly[j];
              }
            }
          }
          while (ids.size() % 16 != 0) {
            ids.push_back(0);
          }
          if (hashType == HashType::SHA256) {
            toId[op.getOut()] = doSha(ids, 0);
          } else if (hashType == HashType::POSEIDON2) {
            toId[op.getOut()] = doPoseidon2(ids);
          } else { // i.e. mixed
            toId[op.getOut()] = doPoseidon2(ids);
          }
        })
        .Case<IntoDigestOp>([&](IntoDigestOp op) {
          std::vector<uint64_t> inputs;
          for (size_t i = 0; i < op.getIn().size(); i++) {
            inputs.push_back(toId[op.getIn()[i]]);
          }
          auto kind = cast<DigestType>(op.getOut().getType()).getKind();
          if (kind == DigestKind::Default) {
            kind = (hashType == HashType::SHA256) ? DigestKind::Sha256 : DigestKind::Poseidon2;
          }
          if (kind == DigestKind::Sha256) {
            if (inputs.size() == 32) {
              toId[op.getOut()] = doIntoDigestShaBytes(inputs);
            } else if (inputs.size() == 16) {
              toId[op.getOut()] = doIntoDigestShaWords(inputs);
            } else {
              throw std::runtime_error("Invalid size for IntoDigestOp");
            }
          } else if (kind == DigestKind::Poseidon2) {
            toId[op.getOut()] = doIntoDigestPoseidon2(inputs);
          } else {
            throw std::runtime_error("Invalid kind for IntoDigestOp");
          }
        })
        .Case<FromDigestOp>([&](FromDigestOp op) {
          if (cast<DigestType>(op.getIn().getType()).getKind() != DigestKind::Sha256) {
            throw std::runtime_error("Unimplemented digest type in FromDigestOp encoding");
          }
          if (op.getOut().size() != 16) {
            throw std::runtime_error("Unimplemented digest size in FromDigestOp encoding");
          }
          uint64_t shaStart = toId[op.getIn()];
          for (size_t i = 0; i < 8; i++) {
            size_t id = shaStart + i;
            toId[op.getOut()[i * 2 + 0]] = addMicro(Value(), MicroOpcode::EXTRACT, id, 0, 0);
            toId[op.getOut()[i * 2 + 1]] = addMicro(Value(), MicroOpcode::EXTRACT, id, 0, 1);
          }
        })
        .Case<HashFoldOp>([&](HashFoldOp op) {
          if (hashType == HashType::SHA256) {
            toId[op.getOut()] = doShaFold(toId[op.getLhs()], toId[op.getRhs()]);
          } else if (hashType == HashType::POSEIDON2) {
            toId[op.getOut()] = doPoseidon2Fold(toId[op.getLhs()], toId[op.getRhs()]);
          } else { // i.e. mixed
            toId[op.getOut()] = doPoseidon2Fold(toId[op.getLhs()], toId[op.getRhs()]);
          }
        })
        .Case<TaggedStructOp>([&](TaggedStructOp op) {
          uint64_t tagDigest = doShaTag(op.getTag());
          std::vector<uint64_t> digestIds;
          std::vector<DigestKind> digestTypes;
          std::vector<uint64_t> valsIds;
          for (auto digest : op.getDigests()) {
            digestIds.push_back(toId[digest]);
            digestTypes.push_back(cast<DigestType>(digest.getType()).getKind());
          }
          for (auto val : op.getVals()) {
            valsIds.push_back(toId[val]);
          }
          toId[op.getOut()] = doTaggedStruct(tagDigest, digestIds, digestTypes, valsIds);
        })
        .Case<HashAssertEqOp>([&](HashAssertEqOp op) {
          uint64_t lhs = toId[op.getLhs()];
          uint64_t rhs = toId[op.getRhs()];
          for (size_t i = 0; i < 8; i++) {
            addMicro(Value(), MicroOpcode::EQ, lhs + i, rhs + i);
          }
        })
        .Case<Iop::ReadOp>([&](Iop::ReadOp op) {
          size_t k = 0;
          size_t rep = 1;
          size_t demont = false;
          if (auto type = mlir::dyn_cast<ValType>(op.getOuts()[0].getType())) {
            assert(type.getFieldK() == 1 || type.getFieldK() == kBabyBearExtSize);
            k = type.getFieldK();
          }
          if (auto type = mlir::dyn_cast<DigestType>(op.getOuts()[0].getType())) {
            k = (hashType == HashType::SHA256 ? 2 : 1);
            demont = hashType != HashType::SHA256;
            rep = 8;
          }
          assert(k);
          addMicro(Value(),
                   MicroOpcode::READ_IOP_HEADER,
                   op.getOuts().size() * rep,
                   k * 2 + op.getFlip());
          for (Value out : op.getOuts()) {
            addMicro(out, MicroOpcode::READ_IOP_BODY, (k == 1), (k != kBabyBearExtSize), demont);
            for (size_t i = 1; i < rep; i++) {
              addMicro(
                  Value(), MicroOpcode::READ_IOP_BODY, (k == 1), (k != kBabyBearExtSize), demont);
            }
          }
        })
        .Case<Iop::CommitOp>([&](Iop::CommitOp op) {
          DigestKind kind = op.getDigest().getType().getKind();
          if (kind != DigestKind::Default) {
            switch (hashType) {
            case HashType::SHA256:
              if (kind != DigestKind::Sha256)
                throw(std::runtime_error("Unexpected digest kind for SHA256 IOP"));
              break;
            case HashType::POSEIDON2:
              if (kind != DigestKind::Poseidon2)
                throw(std::runtime_error("Unexpected digest kind for Poseidon2 IOP"));
              break;
            default:
              throw(std::runtime_error("Unsupported IOP type"));
            }
          }
          getRng(op.getIop())->mix(*this, toId[op.getDigest()]);
        })
        .Case<Iop::RngValOp>([&](Iop::RngValOp op) {
          size_t k = mlir::cast<ValType>(op.getType()).getFieldK();
          uint64_t tot = 0;
          // We need to make an RNG for each field element
          for (size_t i = 0; i < k; i++) {
            // Get the part
            uint64_t part = getRng(op.getIop())->generateFp(*this);
            // We then accumulate parts into a whole field element
            if (i != 0) {
              part = addMicro(Value(), MicroOpcode::MUL, part, i);
              tot = addMicro(Value(), MicroOpcode::ADD, tot, part);
            } else {
              tot = part;
            }
          }
          // Set the final output
          toId[op.getOut()] = tot;
        })
        .Case<Iop::RngBitsOp>([&](Iop::RngBitsOp op) {
          toId[op.getOut()] = getRng(op.getIop())->generateBits(*this, op.getBits());
        })
        .Case<SetGlobalOp>([&](SetGlobalOp op) {
          // Pad with zeroes to return an item smaller than a digest.
          // This is primarily used for testing.
          auto inId = toId[op.getIn()];
          auto inType = op.getIn().getType();
          auto valType = mlir::dyn_cast<ValType>(inType);
          assert(valType);
          size_t k = valType.getFieldK();

          auto zeroId = addMicro(Value(), MicroOpcode::CONST, 0, 0);
          for (size_t i = 0; i < kDigestWords / 2; i++) {
            if (i >= k) {
              addMicro(Value(), MicroOpcode::CONST, 0, 0);
            } else {
              addMicro(Value(), MicroOpcode::ADD, zeroId, inId + i);
            }
          }
          // Our padded value is right after the zero constant we wrote.
          addMacro(/*outs=*/0, MacroOpcode::SET_GLOBAL, zeroId + 1, 0);
        })
        .Case<SetGlobalDigestOp>([&](SetGlobalDigestOp op) {
          auto inId = toId[op.getIn()];
          auto inType = op.getIn().getType();
          assert(mlir::dyn_cast<DigestType>(inType));
          // SET_GLOBAL takes two cycles since we don't have enough
          // registers to read the whole digest at once.
          addMacro(/*outs=*/0, MacroOpcode::SET_GLOBAL, inId, op.getOffset() * 2);
          addMacro(
              /*outs=*/0, MacroOpcode::SET_GLOBAL, inId + kDigestWords / 2, op.getOffset() * 2 + 1);
        })
        .Case<SelectOp>([&](SelectOp op) {
          size_t idx = toId[op.getIdx()];
          uint64_t start = toId[op.getElems()[0]];
          uint64_t step = (kBabyBearP + toId[op.getElems()[1]] - start) % kBabyBearP;
          addMicro(op.getOut(), MicroOpcode::SELECT, idx, start, step);
          if (isa<DigestType>(op.getElems()[0].getType())) {
            for (size_t i = 1; i < 8; i++) {
              addMicro(Value(), MicroOpcode::SELECT, idx, start + i, step);
            }
          }
        })
        .Case<HashCheckedBytesOp>([&](HashCheckedBytesOp op) {
          Value evalPt = op.getEvalPt();
          uint64_t count = op.getEvalsCount();
          auto [poseidon, evals] = doHashCheckedBytes(toId[evalPt], count);
          for (size_t i = 0; i < evals.size(); i++) {
            toId[op.getEvaluations()[i]] = evals[i];
          }
          toId[op.getDigest()] = poseidon;
        })
        .Case<HashCheckedBytesPublicOp>([&](HashCheckedBytesPublicOp op) {
          Value evalPt = op.getEvalPt();
          uint64_t count = op.getEvalsCount();
          auto [poseidon, sha, evals] = doHashCheckedBytesPublic(toId[evalPt], count);
          for (size_t i = 0; i < evals.size(); i++) {
            toId[op.getEvaluations()[i]] = evals[i];
          }
          toId[op.getDigest()] = poseidon;
          toId[op.getPubDigest()] = sha;
        })
        .Default([&](Operation* op) {
          llvm::errs() << "Invalid operation: " << *op << "\n";
          throw std::runtime_error("Invalid operation");
        });
  }
};

void ShaRng::setConstants(Instructions& insts) {
  auto pool0Digest = shaHash("Hello");
  auto pool1Digest = shaHash("World");
  insts.shaRngConsts = insts.nextOut;
  for (size_t i = 0; i < 8; i++) {
    insts.addHalfsConst(pool0Digest.words[i]);
  }
  for (size_t i = 0; i < 8; i++) {
    insts.addHalfsConst(pool1Digest.words[i]);
  }
}

ShaRng::ShaRng(Instructions& insts) {
  pool0 = insts.shaRngConsts;
  pool1 = insts.shaRngConsts + 8;
  poolUsed = 0;
}

void ShaRng::step(Instructions& insts) {
  pool0 = insts.doShaFold(pool0, pool1);
  pool1 = insts.doShaFold(pool0, pool1);
  poolUsed = 0;
}

uint64_t ShaRng::generate(Instructions& insts) {
  if (poolUsed == 8) {
    step(insts);
  }
  return pool0 + (poolUsed++);
}

// Implement the interface
uint64_t ShaRng::generateBits(Instructions& insts, size_t bits) {
  uint32_t andMask = (1 << bits) - 1;
  auto maskId = insts.addHalfsConst(andMask);
  auto genId = generate(insts);
  // AND and combine [a, b, 0, 0] & [c, d, 0, 0] -> [(a + (b<<16)) & (c + (d<<16)), 0, 0, 0]
  return insts.addMacro(
      /*outs=*/1, MacroOpcode::BIT_OP_SHORTS, maskId, genId, /*and_and_combine=*/1);
}

uint64_t ShaRng::generateFp(Instructions& insts) {
  // Combine 6 U32 values to produce an FP
  std::vector<uint64_t> part_ids;
  // First we make the 6 parts
  for (size_t i = 0; i < 6; i++) {
    part_ids.push_back(generate(insts));
  }
  // We now these in 2 at a time to create a single uniformly distributed Fp
  insts.addMicro(Value(), MicroOpcode::MIX_RNG, part_ids[0], part_ids[1], 0);
  insts.addMicro(Value(), MicroOpcode::MIX_RNG, part_ids[2], part_ids[3], 1);
  return insts.addMicro(Value(), MicroOpcode::MIX_RNG, part_ids[4], part_ids[5], 1);
}

void ShaRng::mix(Instructions& insts, uint64_t digest) {
  uint64_t xorOut = insts.nextOut;
  for (size_t i = 0; i < 8; i++) {
    // Xors and returns 2 shorts: [a, b, 0, 0] ^ [c, d, 0, 0] -> [a ^ c, b ^ d, 0, 0]
    insts.addMacro(
        /*outs=*/1, MacroOpcode::BIT_OP_SHORTS, digest + i, pool0 + i, /*and_and_combine=*/0);
  }
  pool0 = xorOut;
  step(insts);
}

Poseidon2Rng::Poseidon2Rng(Instructions& insts) : isInit(true), curState(0), stateUsed(0) {}

uint64_t Poseidon2Rng::generateBits(Instructions& insts, size_t bits) {
  assert(!isInit);
  assert(bits <= 27);
  uint64_t val = generateFp(insts);
  for (size_t i = 0; i < 3; i++) {
    uint64_t newVal = generateFp(insts);
    uint64_t oldValIsZero = insts.addMicro(Value(), MicroOpcode::INV, val, 0);
    uint64_t step = (kBabyBearP + newVal - val) % kBabyBearP;
    val = insts.addMicro(Value(), MicroOpcode::SELECT, oldValIsZero, val, step);
  }
  uint32_t andMask = (1 << bits) - 1;
  auto maskId = insts.addMicro(Value(), MicroOpcode::CONST, andMask, 0);
  // AND with mask
  return insts.addMacro(/*outs=*/1, MacroOpcode::BIT_AND_ELEM, val, maskId);
}

uint64_t Poseidon2Rng::generateFp(Instructions& insts) {
  assert(!isInit);
  if (stateUsed == 16) {
    stateUsed = 0;
    mix(insts, 0);
  }
  return curState + (stateUsed++);
}

void Poseidon2Rng::mix(Instructions& insts, uint64_t digest) {
  if (stateUsed != 0) {
    stateUsed = 0;
    mix(insts, 0);
  }
  std::vector<uint64_t> digest_ids;
  uint64_t keepState = 0;
  if (digest != 0) {
    for (size_t i = 0; i < 8; i++) {
      digest_ids.push_back(digest + i);
    }
    insts.addPoseidon2Load(/*doMont=*/1,
                           keepState,
                           /*keepUpperState=*/0,
                           /*prepFull=*/isInit,
                           /*group=*/0,
                           digest_ids);
    keepState = 1;
  }
  if (!isInit) {
    for (size_t i = 0; i < 3; i++) {
      std::vector<uint64_t> prev_ids;
      for (size_t j = 0; j < 8; j++) {
        prev_ids.push_back(curState + i * 8 + j);
      }
      insts.addPoseidon2Load(
          /*doMont=*/0,
          keepState,
          /*keepUpperState=*/0,
          /*addprepFullConsts=*/i == 2,
          /*group=*/i,
          prev_ids);
      keepState = 1;
    }
  }
  isInit = false;
  insts.addPoseidon2Full(0);
  insts.addPoseidon2Full(1);
  insts.addPoseidon2Partial();
  insts.addPoseidon2Full(2);
  insts.addPoseidon2Full(3);
  curState = insts.addPoseidon2Store(/*doMont=*/0, /*group=*/0);
  insts.addPoseidon2Store(/*doMont=*/0, /*group=*/1);
  insts.addPoseidon2Store(/*doMont=*/0, /*group=*/2);
  stateUsed = 0;
}

void MixedPoseidon2ShaRng::mix(Instructions& insts, uint64_t digest) {
  // For each element of the Poseidon2 hash, we convert it to a form usable by SHA.
  // This is done in stages so that macro ops are grouped together for efficiency
  // First, we 'and' things by 0xffff
  std::vector<uint64_t> lowHalves;
  for (size_t i = 0; i < 8; i++) {
    lowHalves.push_back(
        insts.addMacro(/*outs=*/1, MacroOpcode::BIT_AND_ELEM, digest + i, insts.addConst(0xffff)));
  }
  std::vector<uint64_t> splitVals;
  // Next we compute the split version
  for (size_t i = 0; i < 8; i++) {
    // Subtract low 16 bits
    uint64_t upperHalf = insts.addMicro(Value(), MicroOpcode::SUB, digest + i, lowHalves[i]);
    // Divide (mul by inverse) to shift down + and also move to second component
    uint64_t top = insts.addMicro(Value(), MicroOpcode::MUL, upperHalf, insts.div2to16Const);
    // Add two parts
    splitVals.push_back(insts.addMicro(Value(), MicroOpcode::ADD, lowHalves[i], top));
  }
  // Now do xor + mix as above
  uint64_t xorOut = insts.nextOut;
  for (size_t i = 0; i < 8; i++) {
    // Xors and returns 2 shorts: [a, b, 0, 0] ^ [c, d, 0, 0] -> [a ^ c, b ^ d, 0, 0]
    insts.addMacro(
        /*outs=*/1, MacroOpcode::BIT_OP_SHORTS, splitVals[i], pool0 + i, /*and_and_combine=*/0);
  }
  pool0 = xorOut;
  step(insts);
}

} // namespace

std::vector<uint32_t> encode(HashType hashType,
                             mlir::Block* block,
                             llvm::DenseMap<Value, uint64_t>* toIdReturn,
                             EncodeStats* stats) {
  Instructions insts(hashType);
  for (Operation& op : block->without_terminator()) {
    insts.addInst(op);
  }

  insts.finalize();
  llvm::errs() << "Actual cycles = " << insts.data.size() << "\n";
  llvm::errs() << "SHA cycles = " << insts.shaUsed << "\n";
  llvm::errs() << "Poseidon2 cycles = " << insts.poseidon2Used << "\n";
  if (stats) {
    stats->totCycles = insts.data.size();
    stats->shaCycles = insts.shaUsed;
    stats->poseidon2Cycles = insts.poseidon2Used;
  }
  std::vector<uint32_t> code;
  code.reserve(nearestPo2(insts.data.size() * kCodeSize));
  size_t cycle = 0;
  auto set = [&](size_t pos, uint64_t val) {
    size_t codePos = cycle * kCodeSize + pos;
    assert(codePos < code.size());
    assert(val <= std::numeric_limits<uint32_t>::max());
    assert(val < kFieldPrimeDefault);
    code[codePos] = val;
  };
  for (cycle = 0; cycle < insts.data.size(); cycle++) {
    code.resize((cycle + 1) * kCodeSize);
    size_t pos = 0;
    set(pos++, insts.data[cycle].writeAddr);
    set(pos + size_t(insts.data[cycle].opType), 1);
    pos += OP_TYPE_COUNT;
    switch (insts.data[cycle].opType) {
    case OpType::MICRO:
      for (size_t i = 0; i < 3; i++) {
        set(pos++, uint64_t(insts.data[cycle].data.micro[i].opcode));
        set(pos++, insts.data[cycle].data.micro[i].operands[0]);
        set(pos++, insts.data[cycle].data.micro[i].operands[1]);
        set(pos++, insts.data[cycle].data.micro[i].operands[2]);
      }
      break;
    case OpType::MACRO:
      set(pos + size_t(insts.data[cycle].data.macro.opcode), 1);
      pos += MACRO_OPCODE_COUNT;
      set(pos++, insts.data[cycle].data.macro.operands[0]);
      set(pos++, insts.data[cycle].data.macro.operands[1]);
      set(pos++, insts.data[cycle].data.macro.operands[2]);
      break;
    case OpType::POSEIDON2_LOAD:
    case OpType::POSEIDON2_STORE:
      set(pos++, insts.data[cycle].data.poseidon2Mem.doMont);
      set(pos++, insts.data[cycle].data.poseidon2Mem.keepState);
      set(pos++, insts.data[cycle].data.poseidon2Mem.keepUpperState);
      set(pos++, insts.data[cycle].data.poseidon2Mem.prepFull);
      set(pos + size_t(insts.data[cycle].data.poseidon2Mem.group), 1);
      pos += 3;
      for (size_t i = 0; i < 8; i++) {
        set(pos++, insts.data[cycle].data.poseidon2Mem.inputs[i]);
      }
      break;
    case OpType::POSEIDON2_FULL:
      set(pos + insts.data[cycle].data.poseidon2Full.cycle, 1);
      break;
    case OpType::POSEIDON2_PARTIAL:
      break;
    case OpType::CHECKED_BYTES:
      set(pos++, insts.data[cycle].data.checkedBytes.evalPt);
      set(pos++, insts.data[cycle].data.checkedBytes.keepCoeffs);
      set(pos++, insts.data[cycle].data.checkedBytes.keepUpperState);
      set(pos++, insts.data[cycle].data.checkedBytes.prepFull);
      break;
    default:
      assert(false);
    }
  }
  if (toIdReturn) {
    *toIdReturn = std::move(insts.toId);
  }
  return code;
}

std::vector<uint32_t> encode(HashType hashType, mlir::Block* block, EncodeStats* stats) {
  return encode(hashType, block, /*toIdReturn=*/nullptr, stats);
}

} // namespace zirgen::recursion
