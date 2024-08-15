// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// AB tests; runs the functions in both the emulator, and in the
// recursion circuit in the emulator, and compares the results to make
// sure everything works exactly the same.

#include "zirgen/compiler/passes/test/AB.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"

#include <gtest/gtest.h>

namespace zirgen::recursion {

// Make a random set of iop values
static std::vector<uint32_t> random_iop(size_t size) {
  std::vector<uint32_t> iopVals;
  for (size_t i = 0; i < size; i++) {
    iopVals.push_back(i * i + 17);
  }
  return iopVals;
}

TEST(AB, ArithAdd) {
  doInlineFpExtAB(random_iop(kBabyBearExtSize * 2), [&](ReadIopVal iop) {
    auto a = iop.readExtVals(1)[0];
    auto b = iop.readExtVals(1)[0];
    auto c = a + b;
    iop.commit(hash(c));
  });
}

TEST(AB, ArithMulBroadcast) {
  doInlineFpExtAB(random_iop(kBabyBearExtSize + 1), [&](ReadIopVal iop) {
    auto a = iop.readExtVals(1)[0];
    auto b = iop.readBaseVals(1)[0];
    auto c = a * b;
    iop.commit(hash(c));
  });
}

TEST(AB, ArithMul) {
  doInlineFpExtAB(random_iop(kBabyBearExtSize * 2), [&](ReadIopVal iop) {
    auto a = iop.readExtVals(1)[0];
    auto b = iop.readExtVals(1)[0];
    auto c = a * b;
    iop.commit(hash(c));
  });
}

TEST(AB, ArithInv) {
  doInlineFpExtAB(random_iop(4), [&](ReadIopVal iop) {
    auto a = iop.readExtVals(1)[0];
    auto c = inv(a);
    iop.commit(hash(c));
  });
}

TEST(AB, MultiOp) {
  doInlineFpExtAB(random_iop(kBabyBearExtSize * 2), [&](ReadIopVal iop) {
    std::vector<Val> in = iop.readExtVals(2);
    Val b = llvm::ArrayRef<uint64_t>({1, 2, 3, 4});
    for (size_t i = 0; i < 2; i++) {
      in[i] = in[i] * b + 77;
    }
    in[0] = in[0] & in[1];
    DigestVal d = hash(in, true);
    DigestVal e = hash(in, false);
    DigestVal x = fold(d, e);
    iop.commit(x);
    Val y = iop.rngExtVal() * iop.rngBaseVal();
    iop.commit(hash(y));
  });
}

TEST(AB, Verify) {
  // Read the seal if it exists
  FILE* file = fopen("/tmp/seal.r0", "rb");
  if (!file) {
    // TODO: Make this test always run, right now we skip if file doesn't exist
    std::cerr << "Didn't find file: /tmp/seal.r0, to generate run:\n";
    std::cerr << "cargo run -r --bin gen_receipt -- --receipt /tmp/seal.r0 --only-seal\n";
    return;
  }
  fseek(file, 0, SEEK_END);
  size_t size = ftell(file) / 4;
  fseek(file, 0, SEEK_SET);
  std::vector<uint32_t> proof(size);
  size_t nread = fread(proof.data(), 4, size, file);
  ASSERT_EQ(nread, size);
  fclose(file);

  // Extract the PO2 from its known position
  auto rv32im = verify::getInterfaceRV32IM();
  size_t po2 = proof[rv32im->out_size()];
  llvm::errs() << "po2 = " << po2 << "\n";

  // Compile the verifier
  doInlineFpExtAB(proof, [&](ReadIopVal iop) { zirgen::verify::verify(iop, po2, *rv32im); });
}

} // namespace zirgen::recursion
