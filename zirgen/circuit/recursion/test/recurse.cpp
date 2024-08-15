// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/circuit/recursion/test/AB.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"

#include <gtest/gtest.h>

namespace zirgen::recursion {

TEST(RECURSION, Recurse) {
  // Read the seal if it exists
  FILE* file = fopen("/tmp/seal.r0", "rb");
  if (!file) {
    // TODO: Make this test always run, right now we skip if file doesn't exist
    std::cerr << "Didn't find file: /tmp/seal.r0, to generate run:\n";
    std::cerr << "cargo run -r --bin gen_receipt -- --receipt /tmp/seal.r0 --loop-count 10000 "
                 "--only-seal\n";
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
  size_t po2 = proof[rv32im_v1::kInOutSize];
  llvm::errs() << "po2 = " << po2 << "\n";

  // Now do an AB
  doAB(HashType::POSEIDON2, {proof}, [&](Buffer out, ReadIopVal iop) {
    auto rv32im = verify::getInterfaceRV32IM();
    // auto rv32im = verify::getInterfaceRecursion();
    verify::VerifyInfo info = verify::verify(iop, po2, *rv32im);
    out.setDigest(0, info.codeRoot, "codeRoot");
    out.setDigest(1, info.outDigest, "outDigest");
  });
}

TEST(RECURSION, MetricsRec) {
  Module module;
  module.addFunc<6>("verify_recursion",
                    {gbuf(recursion::kOutSize), ioparg(), ioparg(), ioparg(), ioparg(), ioparg()},
                    [&](Buffer out,
                        ReadIopVal allowedRoot,
                        ReadIopVal seal1,
                        ReadIopVal allowed1,
                        ReadIopVal seal2,
                        ReadIopVal allowed2) {
                      auto circuit = verify::getInterfaceRecursion();

                      verify::VerifyInfo info = zirgen::verify::verifyRecursion(
                          allowedRoot, {seal1, seal2}, {allowed1, allowed2}, *circuit);
                      out.setDigest(0, info.codeRoot, "codeRoot");
                      out.setDigest(1, info.outDigest, "outDigest");
                    });
  module.optimize();
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("verify_recursion");
  llvm::DenseMap<mlir::Value, uint64_t> toId;
  std::vector<uint32_t> code = encode(HashType::POSEIDON2, &func.front(), &toId);
  llvm::errs() << "CYCLES = " << (code.size() / kCodeSize) << "\n";
}

TEST(RECURSION, MetricsRV32IM) {
  Module module;
  module.addFunc<1>("verify", {ioparg()}, [&](ReadIopVal iop) {
    auto circ = verify::getInterfaceRV32IM();
    zirgen::verify::verify(iop, 17, *circ);
  });
  module.optimize();
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("verify");
  llvm::DenseMap<mlir::Value, uint64_t> toId;
  std::vector<uint32_t> code = encode(HashType::MIXED_POSEIDON2_SHA, &func.front(), &toId);
  llvm::errs() << "CYCLES = " << (code.size() / kCodeSize) << "\n";
}

TEST(RECURSION, RecurseRecurse) {
  // Read the seal if it exists.  This is generated from the test_recursion tests in risc0-recursion
  // and written out from the new_verify_recursion method if the fs::write line is uncommented.
  FILE* file = fopen("/tmp/recursion-seal.r0", "rb");
  if (!file) {
    // TODO: Make this test always run, right now we skip if file doesn't exist
    std::cerr << "Skipping RecurseRecurse for now; generate /tmp/recursion-seal.r0 by\n"
              << "uncommenting the associated fs::write in recursion/src/prove/mod.rs\n"
              << "and running cargo test -p risc0-recursion\n";
    return;
  }

  fseek(file, 0, SEEK_END);
  size_t size = ftell(file) / 4;
  fseek(file, 0, SEEK_SET);
  std::vector<uint32_t> proof(size);
  size_t nread = fread(proof.data(), 4, size, file);
  ASSERT_EQ(nread, size);
  fclose(file);

  const uint32_t* proof_ptr = proof.data();
  std::vector<std::vector<uint32_t>> proofs;
  auto add_proof = [&](size_t len) {
    std::cerr << "Extracting proof starting at " << (proof_ptr - proof.data()) << ", len=" << len
              << "\n";
    for (size_t i = 0; i < len && i < 20; ++i) {
      std::cerr << " " << proof_ptr[i];
    }
    std::cerr << "\n";
    proofs.emplace_back(proof_ptr, proof_ptr + len);
    proof_ptr += len;
  };
  // TODO: These should not be hardcoded; perhaps we need more
  // structure definition here.
  add_proof(kDigestWords);                               // allowedRoot
  add_proof(67733);                                      // seal1
  add_proof(1 + kDigestWords * kAllowedCodeMerkleDepth); // allowed1
  add_proof(67733);                                      // seal2
  add_proof(1 + kDigestWords * kAllowedCodeMerkleDepth); // allowed2
  size_t used = proof_ptr - proof.data();

  // Extract the PO2 from its known position
  ASSERT_EQ(proofs[1][kOutSize], kRecursionPo2);
  ASSERT_EQ(proofs[3][kOutSize], kRecursionPo2);

  ASSERT_EQ(used, proof.size());

  // Now do an AB
  doAB<6>(HashType::SHA256,
          proofs,
          [&](Buffer out,
              ReadIopVal allowedRoot,
              ReadIopVal seal1,
              ReadIopVal allowed1,
              ReadIopVal seal2,
              ReadIopVal allowed2) {
            auto circuit = verify::getInterfaceRecursion();

            verify::VerifyInfo info = zirgen::verify::verifyRecursion(
                allowedRoot, {seal1, seal2}, {allowed1, allowed2}, *circuit);
            out.setDigest(0, info.codeRoot, "codeRoot");
            out.setDigest(1, info.outDigest, "outDigest");
          });
}

} // namespace zirgen::recursion
