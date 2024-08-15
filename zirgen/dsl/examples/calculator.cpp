// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// Does simple tests of the "calculator" circuit

#include <array>
#include <deque>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

namespace {

#include "Fp.h"

struct ExecContext {
  std::vector<uint32_t>& data;
  std::vector<uint32_t>& global;
};
using ValidityRegsContext = ExecContext;
using ValidityTapsContext = ExecContext;

std::deque<int> fromUser;
std::deque<int> toUser;

struct CalcExterns {
  // Provide a mechanism to supply data from the circuit; the generated
  // code calls externGetValFromUser to retrieve a value.
  Val getValFromUser() {
    assert(!fromUser.empty());
    int val = fromUser.front();
    fromUser.pop_front();
    return val;
  }

  // Provide a mechanism to retrieve data from the circuit; code calls
  // externOutputToUser to output a value;
  void outputToUser(Val val) { toUser.push_back(val); }

  void log(std::string message, std::initializer_list<Val> x) {
    return log_impl(message, x.begin());
  }
};
CalcExterns externs;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include "zirgen/dsl/examples/calculator.cpp.inc"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace

enum Operation { ADD = 0, SUB = 1 };

// Performs a simple test of 123 + 456 using the generated code for the calculator circuit.
TEST(calculator, adder) {
  std::vector<uint32_t> dataBuffer;
  dataBuffer.resize(kRegCountData, 0);
  std::vector<uint32_t> globalBuffer;
  globalBuffer.resize(kRegCountGlobal, 0);
  ExecContext ctx = {.data = dataBuffer, .global = globalBuffer};

  // Suppy "123" and "456" to the adder sub-circuit, which retrieves them via externGetValFromUser.
  fromUser.clear();
  fromUser.push_back(ADD);
  fromUser.push_back(123);
  fromUser.push_back(456);

  // Collect output from the circuit supplied via externOutputToUser.
  toUser.clear();

  // Allocate a new layout.
  const auto& layout = kLayout_Top;

  // Run the top of the circuit using our layout as output registers.
  TopStruct top = exec_Top(ctx, BoundLayout(&layout, dataBuffer));

  // Make sure our output registers were filled in correctly.
  EXPECT_EQ(ctx.data.at(layout.left._super.index), 123);
  EXPECT_EQ(ctx.data.at(layout.right._super.index), 456);
  // TODO: hide this _construct7 name.
  Val result = ctx.data.at(layout.result._super._super.index);
  EXPECT_EQ(result, 123 + 456);

  // Make sure the values returned from the "Top" component were filled in correctly.
  EXPECT_EQ(top.left._super, 123);
  EXPECT_EQ(top.right._super, 456);
  result = top.result._super._super;
  EXPECT_EQ(result, 123 + 456);

  // Make sure the circuit consumed the two values that we supplied it via externGetValFromUser.
  EXPECT_THAT(fromUser, IsEmpty());

  // Make sure the circuit outputted the sum via externOutputToUser.
  EXPECT_THAT(toUser, ElementsAre(123 + 456));
}

// Performs a simple test of 456 - 123  using the generated code for the calculator circuit.
TEST(calculator, subtractor) {
  std::vector<uint32_t> dataBuffer;
  dataBuffer.resize(kRegCountData, 0);
  std::vector<uint32_t> globalBuffer;
  globalBuffer.resize(kRegCountGlobal, 0);
  ExecContext ctx = {.data = dataBuffer, .global = globalBuffer};

  // Suppy "123" and "456" to the subtractor sub-circuit, which retrieves them via
  // externGetValFromUser.
  fromUser.clear();
  fromUser.push_back(SUB);
  fromUser.push_back(456);
  fromUser.push_back(123);

  // Collect output from the circuit supplied via externOutputToUser.
  toUser.clear();

  // Allocate a new layout.
  const TopLayout& layout = kLayout_Top;

  // Run the top of the circuit using our layout as output registers.
  TopStruct top = exec_Top(ctx, BoundLayout(&layout, ctx.data));

  // Make sure our output registers were filled in correctly.
  EXPECT_EQ(ctx.data.at(layout.left._super.index), 456);
  EXPECT_EQ(ctx.data.at(layout.right._super.index), 123);
  Val result = ctx.data.at(layout.result._super._super.index);
  EXPECT_EQ(result, 456 - 123);

  // Make sure the values returned from the "Top" component were filled in correctly.
  EXPECT_EQ(top.left._super, 456);
  EXPECT_EQ(top.right._super, 123);
  result = top.result._super._super;
  EXPECT_EQ(result, 456 - 123);

  // Make sure the circuit consumed the two values that we supplied it via externGetValFromUser.
  EXPECT_THAT(fromUser, IsEmpty());

  // Make sure the circuit outputted the sum via externOutputToUser.
  EXPECT_THAT(toUser, ElementsAre(456 - 123));
}
