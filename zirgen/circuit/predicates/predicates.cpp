// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/predicates/predicates.h"

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"

namespace zirgen::predicates {

using Zll::DigestKind;

static Val readVal(llvm::ArrayRef<Val>& stream) {
  assert(stream.size() >= 1);
  Val out = stream[0];
  stream = stream.drop_front();
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

static DigestVal readSha(llvm::ArrayRef<Val>& stream, bool longDigest = false) {
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

U32Reg::U32Reg(llvm::ArrayRef<Val>& stream) {
  for (size_t i = 0; i < 4; i++) {
    val[i] = readVal(stream);
  }
}

Val U32Reg::flat() {
  Val tot = 0;
  Val mul = 1;
  for (size_t i = 0; i < 4; i++) {
    tot = tot + mul * val[i];
    mul = mul * 256;
  }
  return tot;
}

void U32Reg::write(std::vector<Val>& stream) {
  for (size_t i = 0; i < 4; i++) {
    stream.push_back(val[i]);
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

ReceiptClaim identity(ReceiptClaim in) {
  // Make an empty output
  return in;
}

} // namespace zirgen::predicates
