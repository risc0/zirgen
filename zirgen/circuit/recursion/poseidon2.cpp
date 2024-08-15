// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/recursion/poseidon2.h"
#include "zirgen/compiler/zkp/baby_bear.h"

using namespace zirgen::recursion::poseidon2;

namespace zirgen::recursion {

std::array<Val, 4> mul4x4Circulant(std::array<Val, 4> in) {
  // See appendix B of Poseidon2 paper for additional details.
  Val t0 = in[0] + in[1];
  Val t1 = in[2] + in[3];
  Val t2 = 2 * in[1] + t1;
  Val t3 = 2 * in[3] + t0;
  Val t4 = 4 * t1 + t3;
  Val t5 = 4 * t0 + t2;
  Val t6 = t3 + t5;
  Val t7 = t2 + t4;
  return {t6, t5, t7, t4};
}

CellVals mulMExt(CellVals in) {
  // Optimized method for multiplication by M_EXT.
  // See appendix B of Poseidon2 paper for additional details.
  CellVals out;
  std::array<Val, 4> tmp_sums{0, 0, 0, 0};
  for (size_t i = 0; i < CELLS / 4; i++) {
    std::array<Val, 4> chunk =
        mul4x4Circulant({in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]});
    for (size_t j = 0; j < 4; j++) {
      Val to_add = chunk[j];
      tmp_sums[j] = tmp_sums[j] + to_add;
      out[4 * i + j] = to_add;
    }
  }
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = out[i] + tmp_sums[i % 4];
  }
  return out;
}

Poseidon2LoadImpl::Poseidon2LoadImpl(Code code, WomHeader header)
    : body(Label("wom_body"), header, 9, 4) {
  for (size_t i = 0; i < 8; i++) {
    ios.emplace_back();
  }
}

void Poseidon2LoadImpl::set(Code code, Val writeAddr) {
  auto inst = code->inst->at<size_t(OpType::POSEIDON2_LOAD)>();
  std::vector<Val> readVals;
  Val mul = inst->doMont * zirgen::kBabyBearFromMontgomery + (1 - inst->doMont) * 1;
  for (size_t i = 0; i < 8; i++) {
    readVals.push_back(ios[i]->doRead(inst->inputs[i])[0] * mul);
  }
  std::vector<Val> stateVals;
  CellVals in;
  for (size_t i = 0; i < CELLS; i++) {
    // We keep the state if _either_ of the following hold:
    //  1. KeepState == 1
    //  2. KeepUpperState == 1 _and_ i >= 16
    // We check this as follows:
    //   To accomplish an `inclusive or` on A, B in {0, 1}, do A + B - A * B.
    auto keep_upper_case = inst->keepUpperState * (i / 8 == 2);
    auto keep_state = inst->keepState + keep_upper_case - inst->keepState * keep_upper_case;
    in[i] = keep_state * UNCHECKED_BACK(1, output[i]->get()) + // Maybe add old state
            inst->group->at(i / 8) * readVals[i % 8];          // Maybe add loaded value
  }

  CellVals mult = mulMExt(in);
  for (size_t i = 0; i < CELLS; i++) {
    Val out;
    out = inst->prepFull * mult[i] + (1 - inst->prepFull) * in[i]; // Maybe mulMExt
    out = out + inst->prepFull * ROUND_CONSTANTS[i];               // Maybe add constant
    output[i]->set(out);
  }
  XLOG("POSEIDON2_LOAD: keepState(%u), keepUpperState(%u) prepFull(%u), group(%u), doMont(%u)",
       inst->keepState,
       inst->keepUpperState,
       inst->prepFull,
       inst->group,
       inst->doMont);
  // XLOG("  readVals: %u %u %u %u %u %u %u %u",
  //      readVals[0],
  //      readVals[1],
  //      readVals[2],
  //      readVals[3],
  //      readVals[4],
  //      readVals[5],
  //      readVals[6],
  //      readVals[7]);
  // XLOG("  in: %u %u %u %u %u %u %u %u", in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]);
  // XLOG("      %u %u %u %u %u %u %u %u",
  //      in[8],
  //      in[9],
  //      in[10],
  //      in[11],
  //      in[12],
  //      in[13],
  //      in[14],
  //      in[15]);
  // XLOG("      %u %u %u %u %u %u %u %u",
  //      in[16],
  //      in[17],
  //      in[18],
  //      in[19],
  //      in[20],
  //      in[21],
  //      in[22],
  //      in[23]);
  // XLOG("  output: %u %u %u %u %u %u %u %u",
  //      output[0],
  //      output[1],
  //      output[2],
  //      output[3],
  //      output[4],
  //      output[5],
  //      output[6],
  //      output[7]);
  // XLOG("          %u %u %u %u %u %u %u %u",
  //      output[8],
  //      output[9],
  //      output[10],
  //      output[11],
  //      output[12],
  //      output[13],
  //      output[14],
  //      output[15]);
  // XLOG("          %u %u %u %u %u %u %u %u",
  //      output[16],
  //      output[17],
  //      output[18],
  //      output[19],
  //      output[20],
  //      output[21],
  //      output[22],
  //      output[23]);
}

Poseidon2FullImpl::Poseidon2FullImpl(Code code, WomHeader header) : pass(header) {}

CellVals mulMInt(CellVals in) {
  // Exploits the fact that off-diagonal entries of M_INT are all 1.
  Val sum = 0;
  CellVals out;
  for (size_t i = 0; i < CELLS; i++) {
    sum = sum + in[i];
  }
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = sum + M_INT_DIAG_HZN[i] * in[i];
  }
  return out;
}

Val doSbox2(Reg tmp, Val in) {
  Val in2 = in * in;
  Val in4 = in2 * in2;
  tmp->set(in4);
  Val in4b = tmp->get();
  Val in7 = in4b * in2 * in;
  return in7;
}

CellVals getConstsPartial(size_t round) {
  CellVals out;
  out[0] = ROUND_CONSTANTS[round * CELLS];
  for (size_t i = 1; i < CELLS; i++) {
    out[i] = 0;
  }
  return out;
}

CellVals getConstsFull(OneHot<4> cycle, size_t add) {
  CellVals out;
  for (size_t i = 0; i < CELLS; i++) {
    Val tot = 0;
    for (size_t j = 0; j < 4; j++) {
      size_t idx = 2 * j + add + 1;
      if (idx == 4 || idx == 8) {
        continue;
      }
      if (j >= 2) {
        idx += ROUNDS_PARTIAL;
      }
      tot = tot + cycle->at(j) * ROUND_CONSTANTS[idx * CELLS + i];
    }
    out[i] = tot;
  }
  return out;
}

void Poseidon2FullImpl::set(Code code, Val writeAddr) {
  auto inst = code->inst->at<size_t(OpType::POSEIDON2_FULL)>();
  XLOG("POSEIDON2_FULL: %u", inst->cycle);
  CellVals sboxOut;
  // The initial mulMExt for cycle 0, happens at the end of the Load, not here
  for (size_t i = 0; i < CELLS; i++) {
    sboxOut[i] = BACK(1, output[i]->get());
  }
  for (size_t i = 0; i < CELLS; i++) {
    sboxOut[i] = doSbox2(pre1[i], sboxOut[i]);
  }
  auto step1MExt = mulMExt(sboxOut);
  // XLOG("  1st mExt cycle %u: %u %u %u %u %u %u %u %u",
  //      inst->cycle,
  //      sboxOut[0],
  //      sboxOut[1],
  //      sboxOut[2],
  //      sboxOut[3],
  //      sboxOut[4],
  //      sboxOut[5],
  //      sboxOut[6],
  //      sboxOut[7]);
  // XLOG("                     %u %u %u %u %u %u %u %u",
  //      sboxOut[8],
  //      sboxOut[9],
  //      sboxOut[10],
  //      sboxOut[11],
  //      sboxOut[12],
  //      sboxOut[13],
  //      sboxOut[14],
  //      sboxOut[15]);
  // XLOG("                     %u %u %u %u %u %u %u %u",
  //      sboxOut[16],
  //      sboxOut[17],
  //      sboxOut[18],
  //      sboxOut[19],
  //      sboxOut[20],
  //      sboxOut[21],
  //      sboxOut[22],
  //      sboxOut[23]);

  auto step1Consts = getConstsFull(inst->cycle, 0);
  for (size_t i = 0; i < CELLS; i++) {
    post1[i]->set(step1MExt[i] + step1Consts[i]);
    sboxOut[i] = doSbox2(pre2[i], BACK(0, post1[i]->get()));
  }
  auto step2MExt = mulMExt(sboxOut);
  // XLOG("  2nd mExt cycle %u: %u %u %u %u %u %u %u %u",
  //      inst->cycle,
  //      sboxOut[0],
  //      sboxOut[1],
  //      sboxOut[2],
  //      sboxOut[3],
  //      sboxOut[4],
  //      sboxOut[5],
  //      sboxOut[6],
  //      sboxOut[7]);
  // XLOG("                    %u %u %u %u %u %u %u %u",
  //      sboxOut[8],
  //      sboxOut[9],
  //      sboxOut[10],
  //      sboxOut[11],
  //      sboxOut[12],
  //      sboxOut[13],
  //      sboxOut[14],
  //      sboxOut[15]);
  // XLOG("                    %u %u %u %u %u %u %u %u",
  //      sboxOut[16],
  //      sboxOut[17],
  //      sboxOut[18],
  //      sboxOut[19],
  //      sboxOut[20],
  //      sboxOut[21],
  //      sboxOut[22],
  //      sboxOut[23]);

  auto step2Consts = getConstsFull(inst->cycle, 1);
  for (size_t i = 0; i < CELLS; i++) {
    output[i]->set(step2MExt[i] + step2Consts[i]);
  }
  /*
  for (size_t i = 0; i < 24; i++) {
    XLOG("  %u", output[i]);
  }
  */
}

Poseidon2PartialImpl::Poseidon2PartialImpl(Code code, WomHeader header) : pass(header) {}

void Poseidon2PartialImpl::set(Code code, Val writeAddr) {
  XLOG("POSEIDON2_PARTIAL");
  auto inst = code->inst->at<size_t(OpType::POSEIDON2_PARTIAL)>();
  CellVals in;
  for (size_t i = 0; i < CELLS; i++) {
    in[i] = BACK(1, output[i]->get());
  }
  sboxIn[0]->set(in[0]);
  for (size_t j = 0; j < ROUNDS_PARTIAL; j++) {
    Val rd_in = BACK(0, sboxIn[j]->get());
    assert(ROUNDS_PARTIAL <=
           CELLS); // We can fit everything in tmp if this assertion is true (which it is)
    rd_in = rd_in + ROUND_CONSTANTS[(j + 4) * CELLS];
    Val sbox = doSbox2(tmp[j], rd_in);
    in[0] = sbox;
    // m_int needs to be applied to whole cell list, but then only forward the first element
    in = mulMInt(in);
    sboxIn[j + 1]->set(in[0]); // the first element of the cell list post-m_int
  }
  // XLOG("  Partial rounds result: %u %u %u %u %u %u %u %u",
  //      in[0],
  //      in[1],
  //      in[2],
  //      in[3],
  //      in[4],
  //      in[5],
  //      in[6],
  //      in[7]);
  // XLOG("                         %u %u %u %u %u %u %u %u",
  //      in[8],
  //      in[9],
  //      in[10],
  //      in[11],
  //      in[12],
  //      in[13],
  //      in[14],
  //      in[15]);
  // XLOG("                         %u %u %u %u %u %u %u %u",
  //      in[16],
  //      in[17],
  //      in[18],
  //      in[19],
  //      in[20],
  //      in[21],
  //      in[22],
  //      in[23]);

  for (size_t i = 0; i < CELLS; i++) {
    // We add the round constants for the first full round of the 2nd half here
    in[i] = in[i] + ROUND_CONSTANTS[CELLS * (ROUNDS_PARTIAL + ROUNDS_HALF_FULL) + i];
    output[i]->set(in[i]);
  }
  /*
  for (size_t i = 0; i < 24; i++) {
    XLOG("  %u", output[i]);
  }
  */
}

Poseidon2StoreImpl::Poseidon2StoreImpl(Code code, WomHeader header)
    : body(Label("wom_body"), header, 9, 4) {
  for (size_t i = 0; i < 8; i++) {
    ios.emplace_back();
  }
}

void Poseidon2StoreImpl::set(Code code, Val writeAddr) {
  auto inst = code->inst->at<size_t(OpType::POSEIDON2_STORE)>();
  std::vector<Val> toWrite(8, Val(0));
  for (size_t i = 0; i < CELLS; i++) {
    toWrite[i % 8] = toWrite[i % 8] + inst->group->at(i / 8) * BACK(1, output[i]->get());
  }
  Val mul = inst->doMont * zirgen::kBabyBearToMontgomery + (1 - inst->doMont) * 1;
  for (size_t i = 0; i < 8; i++) {
    ios[i]->doWrite(writeAddr + i, {toWrite[i] * mul, 0, 0, 0});
  }
  for (size_t i = 0; i < 24; i++) {
    output[i]->set(BACK(1, output[i]->get()));
  }
  XLOG("POSEIDON2_OUTPUT: group(%u), doMont(%u)", inst->group, inst->doMont);
  // {
  //   // Interacting with ios[0]->data() directly causes problems, so recalculate
  //   XLOG("  wrote: %u %u %u %u %u %u %u %u",
  //        toWrite[0] * mul,
  //        toWrite[1] * mul,
  //        toWrite[2] * mul,
  //        toWrite[3] * mul,
  //        toWrite[4] * mul,
  //        toWrite[5] * mul,
  //        toWrite[6] * mul,
  //        toWrite[7] * mul);
  // }
  /*
  for (size_t i = 0; i < 24; i++) {
    XLOG("  %x", output[i]);
  }
  */
}

} // namespace zirgen::recursion
