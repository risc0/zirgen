// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/u32.h"

#include "zirgen/components/test/test_with_bytes.h"
#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

TEST(TopBit, basic) {
  TestWithBytes test(
      /*userIn=*/1,
      /*userOut=*/2,
      /*bytesRequired=*/1,
      /*regsRequired=*/2,
      []() {
        Reg val("code");
        Reg highOut("out");
        Reg lowOut("out");

        TopBit tb;
        tb->set(U32Val{0, 0, 0, val});
        highOut->set(tb->getHighBit());
        lowOut->set(tb->getLow31().bytes[3]);
      });

  ASSERT_EQ(test.run({0}), Interpreter::Buffer({{0}, {0}}));
  ASSERT_EQ(test.run({127}), Interpreter::Buffer({{0}, {127}}));
  ASSERT_EQ(test.run({128}), Interpreter::Buffer({{1}, {0}}));
  ASSERT_EQ(test.run({255}), Interpreter::Buffer({{1}, {127}}));
}

TEST(U32Normalize, basic) {
  TestWithBytes test(
      /*userIn=*/8,
      /*userOut=*/4,
      /*bytesRequired=*/4,
      /*regsRequired=*/8,
      []() {
        std::vector<Reg> inputs;
        std::vector<Reg> outputs;
        for (size_t i = 0; i < 8; i++) {
          inputs.emplace_back("code");
        }
        for (size_t i = 0; i < 4; i++) {
          outputs.emplace_back("out");
        }

        U32Val inA = {inputs[0], inputs[1], inputs[2], inputs[3]};
        U32Val inB = {inputs[4], inputs[5], inputs[6], inputs[7]};

        TwitPrepare<2> twitPrep;
        U32Normalize norm;
        norm->set(inA + U32Val::underflowProtect() - inB);

        U32Val out = norm->getNormed();

        for (size_t i = 0; i < 4; i++) {
          outputs[i]->set(out.bytes[i]);
        }
      });
  // test.runner.module.dumpStage(0);

  auto runSub = [&](uint32_t a, uint32_t b) {
    std::vector<uint64_t> input(8);
    input[0] = (a >> 0) & 0xff;
    input[1] = (a >> 8) & 0xff;
    input[2] = (a >> 16) & 0xff;
    input[3] = (a >> 24) & 0xff;
    input[4] = (b >> 0) & 0xff;
    input[5] = (b >> 8) & 0xff;
    input[6] = (b >> 16) & 0xff;
    input[7] = (b >> 24) & 0xff;
    auto output = test.run(input);
    uint32_t out =
        (output[0][0] | (output[1][0] << 8) | (output[2][0] << 16) | (output[3][0] << 24));
    return out;
  };

  srand(2);
  auto rand32 = []() { return (rand() << 17) ^ (rand() << 2) ^ rand(); };

  ASSERT_EQ(runSub(5, 2), 3);
  ASSERT_EQ(runSub(2, 5), -3);
  for (size_t i = 0; i < 20; i++) {
    uint32_t a = rand32();
    uint32_t b = rand32();
    ASSERT_EQ(runSub(a, b), a - b);
  }
}

TEST(U32Mul, basic) {
  TestWithBytes test(
      /*userIn=*/10,
      /*userOut=*/8,
      /*bytesRequired=*/13,
      /*regsRequired=*/8,
      []() {
        std::vector<Reg> inputs;
        std::vector<Reg> outputs;
        for (size_t i = 0; i < 10; i++) {
          inputs.emplace_back("code");
        }
        for (size_t i = 0; i < 8; i++) {
          outputs.emplace_back("out");
        }

        U32Val inA = {inputs[0], inputs[1], inputs[2], inputs[3]};
        U32Val inB = {inputs[4], inputs[5], inputs[6], inputs[7]};
        Val signedA = inputs[8];
        Val signedB = inputs[9];

        TwitPrepare<4> twitPrep;
        U32Mul mul;
        mul->set(inA, inB, signedA, signedB);
        U32Val outLow = mul->getLow();
        U32Val outHigh = mul->getHigh();

        for (size_t i = 0; i < 4; i++) {
          outputs[i]->set(outLow.bytes[i]);
          outputs[4 + i]->set(outHigh.bytes[i]);
        }
      });
  // test.runner.module.dumpStage(0);

  auto runMul = [&](uint32_t a, uint32_t b, bool signedA, bool signedB) {
    std::vector<uint64_t> input(10);
    input[0] = (a >> 0) & 0xff;
    input[1] = (a >> 8) & 0xff;
    input[2] = (a >> 16) & 0xff;
    input[3] = (a >> 24) & 0xff;
    input[4] = (b >> 0) & 0xff;
    input[5] = (b >> 8) & 0xff;
    input[6] = (b >> 16) & 0xff;
    input[7] = (b >> 24) & 0xff;
    input[8] = signedA;
    input[9] = signedB;
    auto output = test.run(input);
    uint32_t outLow =
        (output[0][0] | (output[1][0] << 8) | (output[2][0] << 16) | (output[3][0] << 24));
    uint32_t outHigh =
        (output[4][0] | (output[5][0] << 8) | (output[6][0] << 16) | (output[7][0] << 24));
    uint64_t out = (uint64_t(outHigh) << 32) | outLow;
    return out;
  };

  srand(2);
  auto rand32 = []() { return (rand() << 17) ^ (rand() << 2) ^ rand(); };

  ASSERT_EQ(runMul(3, 3, false, false), 9);
  ASSERT_EQ(runMul(-3, 3, true, false), -9);
  for (size_t i = 0; i < 20; i++) {
    uint32_t a = rand32();
    uint32_t b = rand32();
    ASSERT_EQ(runMul(a, b, false, false), uint64_t(a) * uint64_t(b));
  }
  for (size_t i = 0; i < 20; i++) {
    int32_t a = rand32();
    uint32_t b = rand32();
    ASSERT_EQ(runMul(a, b, true, false), int64_t(a) * uint64_t(b));
  }
  for (size_t i = 0; i < 20; i++) {
    uint32_t a = rand32();
    int32_t b = rand32();
    ASSERT_EQ(runMul(a, b, false, true), uint64_t(a) * int64_t(b));
  }
  for (size_t i = 0; i < 20; i++) {
    int32_t a = rand32();
    int32_t b = rand32();
    ASSERT_EQ(runMul(a, b, true, true), int64_t(a) * int64_t(b));
  }
}

} // namespace zirgen
