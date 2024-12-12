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

#include "zirgen/circuit/verify/circom/test/run_circom.h"

#include <cstdlib>
#include <fstream>

#include "mlir/Pass/PassManager.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/circuit/verify/circom/circom.h"
#include "zirgen/compiler/zkp/poseidon_254.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen::snark {

std::vector<uint64_t>
run_circom(func::FuncOp func, const std::vector<uint32_t>& iop, const std::string& tmp_path) {
  // Write the circom file
  std::ofstream circom_writer;
  circom_writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  circom_writer.open(tmp_path + "/test.circom");
  CircomGenerator gen(circom_writer);
  gen.emit(func, false);
  circom_writer.close();

  // Write the JSON file
  std::ofstream json_writer;
  json_writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  json_writer.open(tmp_path + "/input.json");
  json_writer << "{\n";
  json_writer << "  \"iop\": [\n";
  size_t idx = 0;
  func.front().walk([&](Iop::ReadOp op) {
    if (auto type = dyn_cast<ValType>(op.getOuts()[0].getType())) {
      for (size_t i = 0; i < op.getOuts().size(); i++) {
        uint32_t demont = uint64_t(iop[idx++]) * kBabyBearFromMontgomery % kBabyBearP;
        json_writer << "  \"" << demont << '"' << (idx == iop.size() ? ' ' : ',');
      }
    } else {
      for (size_t i = 0; i < op.getOuts().size(); i++) {
        Digest d = *reinterpret_cast<const Digest*>(iop.data() + idx);
        idx += 8;
        P254 out(d);
        json_writer << "  \"" << out.toString() << '"' << (idx == iop.size() ? ' ' : ',');
      }
    }
  });
  json_writer << "  ]\n";
  json_writer << "}\n";
  json_writer.close();

  // Compile the circom file
  int ret = system(("circom -l zirgen/circuit/verify/circom/include/ --r1cs --wasm --sym -o " +
                    tmp_path + " " + tmp_path + "/test.circom")
                       .c_str());
  if (ret != 0) {
    throw std::runtime_error("Unable to run circom");
  }
  // Run snarkjs
  ret = system(("snarkjs wc " + tmp_path + "/test_js/test.wasm " + tmp_path + "/input.json " +
                tmp_path + "/output.wtns")
                   .c_str());
  if (ret != 0) {
    throw std::runtime_error("Unable to run snarkjs");
  }
  // Convert witness to js
  ret = system(("snarkjs wej " + tmp_path + "/output.wtns " + tmp_path + "/output.json").c_str());
  if (ret != 0) {
    throw std::runtime_error("Unable to run snarkjs");
  }

  // Read the outputs
  std::vector<uint64_t> out;
  size_t outSize = cast<BufferType>(func.front().getArgument(0).getType()).getSize();
  std::ifstream json_reader;
  json_reader.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
  json_reader.open(tmp_path + "/output.json");
  std::string line;
  // Read the '[' line
  getline(json_reader, line);
  // Read the 1 line
  getline(json_reader, line);
  for (size_t i = 0; i < outSize; i++) {
    // Read and push back the numbers
    std::getline(json_reader, line);
    out.push_back(atol(line.substr(2, std::string::npos).c_str()));
  }

  return out;
}

} // namespace zirgen::snark
