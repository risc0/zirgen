// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/circuit/predicates/predicates.h"

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"

namespace zirgen::predicates {

using Zll::DigestKind;

Val readVal(llvm::ArrayRef<Val>& stream) {
  assert(stream.size() >= 1);
  Val out = stream[0];
  stream = stream.drop_front();
  return out;
}

static std::array<Val, 4> readExtVal(llvm::ArrayRef<Val>& stream) {
  assert(stream.size() >= 4);
  std::array<Val, 4> out;
  for (size_t i = 0; i < 4; i++) {
    out[i] = stream[i];
  }
  stream = stream.drop_front(4);
  return out;
}

// Read a default digest (i.e. Poseidon2) from the stream.
static DigestVal readDigest(llvm::ArrayRef<Val>& stream, bool longDigest = false) {
  size_t digestSize = (longDigest ? 32 : 16);
  assert(stream.size() >= digestSize);
  DigestVal out = intoDigest(stream.take_front(digestSize), DigestKind::Default);
  stream = stream.drop_front(digestSize);
  return out;
}

DigestVal readSha(llvm::ArrayRef<Val>& stream, bool longDigest) {
  size_t digestSize = (longDigest ? 32 : 16);
  assert(stream.size() >= digestSize);
  DigestVal out = intoDigest(stream.take_front(digestSize), DigestKind::Sha256);
  stream = stream.drop_front(digestSize);
  return out;
}

void writeSha(DigestVal val, std::vector<Val>& stream) {
  auto vals = fromDigest(val, 16);
  for (size_t i = 0; i < 16; i++) {
    stream.push_back(vals[i]);
  }
}

PCVal::PCVal(llvm::ArrayRef<Val>& stream) {
  for (size_t i = 0; i < PCVal::size; i++) {
    val[i] = readVal(stream);
  }
}

PCVal PCVal::zero() {
  return PCVal({Val(0), Val(0), Val(0), Val(0)});
}

Val PCVal::flat() {
  Val tot = 0;
  Val mul = 1;
  for (size_t i = 0; i < PCVal::size; i++) {
    tot = tot + mul * val[i];
    mul = mul * 256;
  }
  return tot;
}

void PCVal::write(std::vector<Val>& stream) {
  for (size_t i = 0; i < PCVal::size; i++) {
    stream.push_back(val[i]);
  }
}

U64Val::U64Val(uint64_t x) {
  for (size_t i = 0; i < U64Val::size; i++) {
    size_t shift = 16 * i;
    uint64_t shortx = (x & (uint64_t(0xffff) << shift)) >> shift;
    shorts[i] = Val(shortx);
  }
}

U64Val U64Val::zero() {
  return U64Val({Val(0), Val(0), Val(0), Val(0)});
}

U64Val U64Val::one() {
  return U64Val({Val(1), Val(0), Val(0), Val(0)});
}

U64Val::U64Val(llvm::ArrayRef<Val>& stream) {
  for (size_t i = 0; i < U64Val::size; i++) {
    Val val = readVal(stream);
    // Ensure that the read value is at most 16 bits.
    zirgen::eq(val, val & 0xffff);
    shorts[i] = val;
  }
}

U64Val U64Val::add(const U64Val& x) {
  std::array<Val, U64Val::size> out;
  Val carry = 0;
  for (size_t i = 0; i < U64Val::size; i++) {
    Val val = shorts.at(i) + x.shorts.at(i) + carry;
    out.at(i) = val & 0xffff;
    carry = (val - out.at(i)) / 0x10000;
  }
  // Disallow overflows.
  eqz(carry);
  return U64Val(out);
}

void U64Val::eq(U64Val& a, U64Val& b) {
  for (size_t i = 0; i < U64Val::size; i++) {
    zirgen::eq(a.shorts.at(i), b.shorts.at(i));
  }
}

void U64Val::write(std::vector<Val>& stream) {
  for (size_t i = 0; i < U64Val::size; i++) {
    stream.push_back(shorts[i]);
  }
}

U256Val U256Val::zero() {
  return U256Val({Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0)});
}

U256Val U256Val::one() {
  return U256Val({Val(1),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0),
                  Val(0)});
}

U256Val::U256Val(llvm::ArrayRef<Val>& stream) {
  for (size_t i = 0; i < U256Val::size; i++) {
    Val val = readVal(stream);
    // Ensure that the read value is at most 16 bits.
    zirgen::eq(val, val & 0xffff);
    shorts[i] = val;
  }
}

U256Val U256Val::add(const U256Val& x) {
  std::array<Val, U256Val::size> out;
  Val carry = 0;
  for (size_t i = 0; i < U256Val::size; i++) {
    Val val = shorts.at(i) + x.shorts.at(i) + carry;
    out.at(i) = val & 0xffff;
    carry = (val - out.at(i)) / 0x10000;
  }
  // Disallow overflows.
  eqz(carry);
  return U256Val(out);
}

void U256Val::eq(const U256Val& a, const U256Val& b) {
  for (size_t i = 0; i < U256Val::size; i++) {
    zirgen::eq(a.shorts.at(i), b.shorts.at(i));
  }
}

void U256Val::write(std::vector<Val>& stream) {
  for (size_t i = 0; i < U256Val::size; i++) {
    stream.push_back(shorts[i]);
  }
}

SystemState::SystemState(llvm::ArrayRef<Val>& stream, bool longDigest)
    : pc(stream), memory(readSha(stream, longDigest)) {}

void SystemState::write(std::vector<Val>& stream) {
  pc.write(stream);
  writeSha(memory, stream);
}

DigestVal SystemState::digest() {
  return taggedStruct("risc0.SystemState", {memory}, {pc.flat()});
}

ReceiptClaim::ReceiptClaim(llvm::ArrayRef<Val>& stream, bool longDigest)
    : input(readSha(stream, longDigest))
    , pre(stream, longDigest)
    , post(stream, longDigest)
    , sysExit(readVal(stream))
    , userExit(readVal(stream))
    , output(readSha(stream, longDigest)) {}

void ReceiptClaim::write(std::vector<Val>& stream) {
  writeSha(input, stream);
  pre.write(stream);
  post.write(stream);
  stream.push_back(sysExit);
  stream.push_back(userExit);
  writeSha(output, stream);
}

DigestVal ReceiptClaim::digest() {
  return taggedStruct(
      "risc0.ReceiptClaim", {input, pre.digest(), post.digest(), output}, {sysExit, userExit});
}

DigestVal Output::digest() {
  return taggedStruct("risc0.Output", {journal, assumptions}, {});
}

Assumption::Assumption(llvm::ArrayRef<Val>& stream, bool longDigest)
    : claim(readSha(stream, longDigest)), controlRoot(readDigest(stream, longDigest)) {}

DigestVal Assumption::digest() {
  return taggedStruct("risc0.Assumption", {claim, controlRoot}, {});
}

void UnionClaim::write(std::vector<Val>& stream) {
  writeSha(left, stream);
  writeSha(right, stream);
}

DigestVal UnionClaim::digest() {
  return taggedStruct("risc0.UnionClaim", {left, right}, {});
}

Work::Work(llvm::ArrayRef<Val>& stream) : nonceMin(stream), nonceMax(stream), value(stream) {}

DigestVal Work::digest() {
  std::vector<Val> vals;
  write(vals);
  return taggedStruct("risc0.Work", {}, vals);
}

void Work::write(std::vector<Val>& stream) {
  nonceMin.write(stream);
  nonceMax.write(stream);
  value.write(stream);
}

ReceiptClaim join(ReceiptClaim claim1, ReceiptClaim claim2) {
  // Make an empty output
  ReceiptClaim claimOut;

  // Constraints on pre/post state
  assert_eq(claim1.post.memory, claim2.pre.memory);
  eq(claim1.post.pc.flat(), claim2.pre.pc.flat());
  claimOut.pre = claim1.pre;
  claimOut.post = claim2.post;

  // Verify first receipt is a system split
  eq(claim1.sysExit, 2);
  eq(claim1.userExit, 0);
  std::vector zeroVec(16, Val(0));
  auto zeroHash = intoDigest(zeroVec, DigestKind::Sha256);
  assert_eq(claim1.output, zeroHash);

  // Final output comes from second receipt
  claimOut.output = claim2.output;
  claimOut.sysExit = claim2.sysExit;
  claimOut.userExit = claim2.userExit;

  // Input must be the same for all
  assert_eq(claim1.input, claim2.input);
  claimOut.input = claim1.input;

  // Return output
  return claimOut;
}

// Given a conditional receipt (cond) on some assumption, and a receipt
// attesting to the validity of an assumption (assum) which proves the
// assumption holds produce a receipt which is unconditionally valid. The tail
// of the assumptions list and the journal digest on the conditional receipt
// must be provided such that the head can be checked and removed.
ReceiptClaim resolve(ReceiptClaim cond, Assumption assum, DigestVal tail, DigestVal journal) {
  // Construct the expected output digest of the conditional receipt.
  // NOTE: If this is the last assumption to be resolved, tail will be all zeroes.
  auto assumptionsDigest = taggedListCons("risc0.Assumptions", assum.digest(), tail);
  Output expectedOutput;
  expectedOutput.journal = journal;
  expectedOutput.assumptions = assumptionsDigest;
  assert_eq(expectedOutput.digest(), cond.output);

  // Make an empty output receipt claim.
  ReceiptClaim claimOut;

  // All fields except output are copied over from the conditional receipt.
  claimOut.input = cond.input;
  claimOut.pre = cond.pre;
  claimOut.post = cond.post;
  claimOut.sysExit = cond.sysExit;
  claimOut.userExit = cond.userExit;

  // Calculate and set the new output digest with the head assumption removed.
  Output resolvedOutput;
  resolvedOutput.journal = journal;
  resolvedOutput.assumptions = tail;
  claimOut.output = resolvedOutput.digest();

  // Return output
  return claimOut;
}

UnionClaim unionFunc(Assumption left, Assumption right) {
  UnionClaim claim;
  claim.left = left.digest();
  claim.right = right.digest();
  return claim;
}

WorkClaim<ReceiptClaim> wrap_povw(size_t po2, U256Val nonce, ReceiptClaim claim) {
  Work work(nonce, nonce, U64Val(1 << uint64_t(po2)));
  return WorkClaim(claim, work);
}

WorkClaim<ReceiptClaim> join_povw(WorkClaim<ReceiptClaim> a, WorkClaim<ReceiptClaim> b) {
  U256Val::eq(a.work.nonceMax.add(U256Val::one()), b.work.nonceMin);

  Work work(a.work.nonceMin, b.work.nonceMax, a.work.value.add(b.work.value));
  auto claim = join(a.claim, b.claim);

  return WorkClaim(claim, work);
}

WorkClaim<ReceiptClaim>
resolve_povw(WorkClaim<ReceiptClaim> cond, Assumption assum, DigestVal tail, DigestVal journal) {
  auto claim = resolve(cond.claim, assum, tail, journal);
  return WorkClaim(claim, cond.work);
}

ReceiptClaim unwrap_povw(WorkClaim<ReceiptClaim> claim) {
  return claim.claim;
}

std::pair<ReceiptClaim, U256Val> readReceiptClaimAndPovwNonce(llvm::ArrayRef<Val>& stream,
                                                              size_t po2) {
  // NOTE: Ordering of these read operations must match the layout of the circuit globals.
  // This ordering can be found in the generated rv32im.cpp.inc file as _globalLayout
  DigestVal input = readSha(stream);
  Val isTerminate = readVal(stream);
  DigestVal output = readSha(stream);
  // NOTE: povwNonce is not part of the rv32im claim, and its value does not matter for the validity
  // of the ReceiptClaim. This nonce is used only for the PoVW accounting logic.
  U256Val povwNonce(stream);
  // NOTE: rng is not part of the claim, and is fully constrained by the circuit. It is included in
  // the globals because putting it there made the circuit construction easier.
  /*Val rng =*/readExtVal(stream);
  // NOTE: shutdownCycle allows checking that splits occurs within a constrained range of cycles,
  // near the end of the trace. This feature is not yet fully implemented. As a result, this value
  // is unchecked.
  /*Val shutdownCycle =*/readVal(stream);
  DigestVal stateIn = readSha(stream);
  DigestVal stateOut = readSha(stream);
  Val termA0High = readVal(stream);
  Val termA0Low = readVal(stream);
  // Val termA1High = readVal(stream);
  // Val termA1Low = readVal(stream);

  // TODO(flaub): implement this once shutdownCycle is finished in the rv32im-v2 circuit
  // size_t segmentThreshold = (1 << po2) - kMaxInsnCycles;
  // eq(shutdownCycle, segmentThreshold);

  ReceiptClaim claim;
  claim.input = input;
  claim.output = output;

  claim.pre.pc = PCVal::zero();
  claim.pre.memory = stateIn;

  claim.post.pc = PCVal::zero();
  eqz(isTerminate * (1 - isTerminate));
  std::vector zeroVec(16, Val(0));
  DigestVal zeroHash = intoDigest(zeroVec, DigestKind::Sha256);
  claim.post.memory = select(isTerminate, {stateOut, zeroHash});

  // Constrain termA0Low to be either 0 or 1.
  //
  // Note that when isTerminate = 1, the system exit code (used to indicate whether the system is
  // e.g. halted, paused, or system split) is set to termA0Low. Without this constraint, it is
  // possible for the RISC-V code to set sysExit to e.g. 2 when isTerminate is true, which is
  // semantically inconsistent in the v1 ReceiptClaim. This would require non-standard RISC-V
  // guest runtime, and so is mitigated by any program that uses the RISC Zero provided runtime.
  eqz(termA0Low * (1 - termA0Low));

  // isTerminate:
  // 0 -> 2
  // 1 -> termA0Low (0, 1)
  claim.sysExit = (2 - 2 * isTerminate) + (isTerminate * termA0Low);
  claim.userExit = isTerminate * termA0High;

  return std::make_pair(claim, povwNonce);
}

ReceiptClaim ReceiptClaim::fromRv32imV2(llvm::ArrayRef<Val>& stream, size_t po2) {
  return readReceiptClaimAndPovwNonce(stream, po2).first;
}

} // namespace zirgen::predicates
