// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// Disable these tests to prevent dependencies on old RISC0 repo
#if 0

#include <gtest/gtest.h>

#include "risc0/core/log.h"
#include "risc0/core/rng.h"
#include "risc0/zkp/core/poly.h"
#include "risc0/zkp/core/rou.h"
#include "risc0/zkp/prove/fri.h"
#include "zirgen/circuit/verify/fri.h"
#include "zirgen/circuit/verify/poly.h"

using namespace zirgen;
using namespace zirgen::verify;

TEST(verify, fri) {
  size_t deg = 1024 * 1024;
  size_t domain = deg * kInvRate;

  // Make a random polynomial Fp4 polynomial
  risc0::PsuedoRng rng(2);
  std::vector<risc0::Fp4> poly(deg);
  for (size_t i = 0; i < deg; i++) {
    poly[i] = risc0::Fp4::random(rng);
    // LOG(0, "POLY[" << i << "] = " << poly[i]);
  }

  // Make convert it to accel version
  auto polyAccel = risc0::AccelSlice<risc0::Fp>::allocate(deg * 4);
  {
    risc0::AccelWriteOnlyLock lock(polyAccel);
    for (size_t i = 0; i < lock.size(); i++) {
      lock[i] = poly[i % deg].elems[i / deg];
    }
  }
  batchBitReverse(polyAccel, 4);

  // Generate the proof
  risc0::WriteIOP wiop;
  friProve(wiop, polyAccel, [&](risc0::WriteIOP& iop, size_t idx) {
    risc0::Fp x = pow(risc0::kRouFwd[risc0::log2Ceil(domain)], idx);
    risc0::Fp4 fx = risc0::polyEval(poly.data(), poly.size(), risc0::Fp4(x));
    LOG(0, "IDX=" << idx << ", x=" << x << " fx=" << fx);
    iop.write(&fx, 1);
  });
  auto proof = wiop.getProof();

  // Now do the FRI verify
  Module module;
  module.addFunc<1>("fri_verify", {ioparg()}, [&](ReadIopVal iop) {
    friVerify(iop, deg, [&](ReadIopVal& iop, Val idx) { return iop.readExtVals(1)[0]; });
  });
  module.optimize();
  // module.dump();
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("fri_verify");
  std::map<std::string, size_t> opCounts;
  size_t totOps = 0;
  size_t hashCycles = 0;
  size_t cyclesPerBlock = 12;
  for (mlir::Operation& op : func.front().without_terminator()) {
    if (auto hashOp = llvm::dyn_cast<HashOp>(op)) {
      size_t count = hashOp.in().size();
      size_t k = llvm::cast<ValType>(hashOp.in()[0].getType()).getFieldK();
      hashCycles += ((count * k) + 15) / 16 * cyclesPerBlock;
    }
    if (auto hashOp = llvm::dyn_cast<HashFoldOp>(op)) {
      hashCycles += cyclesPerBlock;
    }
    opCounts[op.getName().getStringRef().str()]++;
    totOps++;
  }
  std::cout << "Tot ops = " << totOps << "\n";
  std::cout << "Hash cycles = " << hashCycles << "\n";
  for (const auto& kvp : opCounts) {
    std::cout << kvp.first << ": " << kvp.second << "\n";
  }
  Interpreter interp;
  risc0::ReadIopVal riop(proof.data(), proof.size());
  interp.setIOP(func.getArgument(0), &riop);
  // interp.setDebug(true);
  if (failed(interp.runBlock(func.front())))
    FAIL() << "failed to evaluate block in interpreter";
}

#endif
