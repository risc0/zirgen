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

#include "zirgen/compiler/r1cs/lower.h"
#include "zirgen/Dialect/R1CS/IR/R1CS.h"

#include "mlir/IR/Builders.h"

using mlir::MLIRContext;

namespace zirgen::R1CS {

namespace {

struct Impl {
  Impl(mlir::OpBuilder& builder, std::vector<DefOp>& wires, r1csfile::BigintLE& prime)
      : builder(builder), wires(wires), prime(gen(prime)) {}
  mlir::IntegerAttr gen(r1csfile::BigintLE&);
  mlir::Value gen(r1csfile::Factor&);
  mlir::Value gen(r1csfile::Combination&);
  void gen(r1csfile::Constraint&);

private:
  mlir::Location loc() { return builder.getUnknownLoc(); }
  mlir::OpBuilder& builder;
  std::vector<DefOp>& wires;
  mlir::IntegerAttr prime;
};

mlir::IntegerAttr Impl::gen(r1csfile::BigintLE& in) {
  // R1CS bigint size must be a multiple of 8 bytes
  assert(0 == (in.size() & 0x07));
  // This conveniently allows us to copy the bits into an array of
  // uint64_t, from which we can construct an APInt.
  unsigned words = in.size() >> 3;
  unsigned bits = in.size() << 3;
  std::vector<uint64_t> bigVal(words);
  for (unsigned i = 0; i < words; ++i) {
    // compute the position from the input array
    size_t offset = i << 3;
    // shift & merge bytes into 64-bit value
    bigVal[i] = (static_cast<uint64_t>(in[0 + offset]) << 0x00) |
                (static_cast<uint64_t>(in[1 + offset]) << 0x08) |
                (static_cast<uint64_t>(in[2 + offset]) << 0x10) |
                (static_cast<uint64_t>(in[3 + offset]) << 0x18) |
                (static_cast<uint64_t>(in[4 + offset]) << 0x20) |
                (static_cast<uint64_t>(in[5 + offset]) << 0x28) |
                (static_cast<uint64_t>(in[6 + offset]) << 0x30) |
                (static_cast<uint64_t>(in[7 + offset]) << 0x38);
  }
  mlir::Type type = builder.getIntegerType(bits);
  mlir::APInt value(bits, words, bigVal.data());
  return builder.getIntegerAttr(type, value);
}

mlir::Value Impl::gen(r1csfile::Factor& in) {
  assert(in.index < wires.size());
  mlir::Value wire = wires[in.index];
  return builder.create<MulOp>(loc(), wire, gen(in.value), prime);
}

mlir::Value Impl::gen(r1csfile::Combination& in) {
  mlir::Value v;
  for (size_t i = 0; i < in.size(); ++i) {
    mlir::Value next = gen(in[i]);
    v = v ? builder.create<SumOp>(loc(), v, next) : next;
  }
  return v;
}

void Impl::gen(r1csfile::Constraint& in) {
  mlir::Value a = gen(in.A);
  mlir::Value b = gen(in.B);
  mlir::Value c = gen(in.C);
  builder.create<ConstrainOp>(loc(), a, b, c);
}

} // namespace

std::optional<mlir::ModuleOp> lower(MLIRContext& ctx, r1csfile::System& src) {

  mlir::OpBuilder builder(&ctx);

  auto loc = builder.getUnknownLoc();
  auto out = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(&out.getBodyRegion().front());

  // Generate wires. If there is a map section, use its labels.
  std::vector<DefOp> wires;
  size_t nTotalPublic = 1 + src.header.nPubOut + src.header.nPubIn;
  if (!src.map.empty()) {
    for (uint32_t i = 0; i < src.map.size(); ++i) {
      uint32_t index = src.map[i];
      bool isPublic = index < nTotalPublic;
      wires.push_back(builder.create<DefOp>(loc, index, isPublic));
    }
  } else {
    // include mandatory 1-signal, at index zero, along with public wires
    for (uint32_t i = 0; i < src.header.nWires; ++i) {
      bool isPublic = i < nTotalPublic;
      wires.push_back(builder.create<DefOp>(loc, i, isPublic));
    }
  }
  assert(wires.size() == src.header.nWires);

  Impl impl(builder, wires, src.header.prime);

  // Generate constraints.
  for (auto& cons : src.constraints) {
    impl.gen(cons);
  }

  return out;
}

} // namespace zirgen::R1CS
