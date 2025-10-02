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

#include "zirgen/circuit/hello_v3/CircuitInterface.h"
#include "zirgen/circuit/verify/verify.h"
#include <memory>

namespace zirgen::hello_v3 {

using namespace verify;

namespace {

class HelloCircuitInterface : public CircuitInterfaceV3 {
public:
  HelloCircuitInterface() {
    tapSet.groups.resize(3);
    tapSet.groups[0].regs.push_back({0, 0, 0, {0}});
    tapSet.groups[1].regs.push_back({0, 0, 1, {0}});
    tapSet.groups[2].regs.push_back({0, 0, 2, {0}});
    tapSet.combos.push_back({0, {0}});
    tapSet.tapCount = 3;

    groupInfo.push_back({0, 0});
    groupInfo.push_back({0, 0});
    groupInfo.push_back({0, 0});
  }

  const Zll::TapSet& getTaps() const override { return tapSet; }
  const llvm::ArrayRef<GroupInfoV3> getGroupInfo() const override { return groupInfo; }

  virtual Val computePolyExt(llvm::ArrayRef<Val> u,
                             llvm::ArrayRef<Val> out,
                             llvm::ArrayRef<Val> accumMix,
                             Val polyMix,
                             Val z) const override {
    Val tot({0, 0, 0, 0});
    Val mul({1, 0, 0, 0});
    auto eqz = [&](Val inner) {
      tot = tot + mul * inner;
      mul = mul * polyMix;
    };
    eqz(u[0]);
    eqz(u[1]);
    eqz(u[2] * (u[2] - Val(1)));
    return tot;
  }

private:
  Zll::TapSet tapSet;
  llvm::SmallVector<GroupInfoV3> groupInfo;
};

} // namespace

std::unique_ptr<zirgen::verify::CircuitInterfaceV3> getCircuitInterface() {
  return std::make_unique<HelloCircuitInterface>();
}

} // namespace zirgen::hello_v3
