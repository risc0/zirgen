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

#include "zirgen/Main/Target.h"

llvm::StringLiteral licenseHeader = R"(// Copyright 2025 RISC Zero, Inc.
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
)";

namespace zirgen {

llvm::StringRef CppCodegenTarget::getDeclExtension() const {
  return "h";
}
llvm::StringRef CppCodegenTarget::getImplExtension() const {
  return "cpp";
}

llvm::StringRef RustCodegenTarget::getDeclExtension() const {
  return "rs";
}
llvm::StringRef RustCodegenTarget::getImplExtension() const {
  return "rs";
}

llvm::StringRef CudaCodegenTarget::getDeclExtension() const {
  return "cuh";
}
llvm::StringRef CudaCodegenTarget::getImplExtension() const {
  return "cu";
}

Template CppCodegenTarget::getStepTemplate() const {
  return Template{.header = (licenseHeader + R"(
#include "steps.h"
#include "witgen.h"

namespace )" + circuitName.getCppNamespace() +
                             R"(::cpu {
)")
                                .str(),
                  .footer = R"(
} // namespace )" + circuitName.getCppNamespace() +
                            R"(::cpu
)"};
}

Template CppCodegenTarget::getStepDeclTemplate() const {
  return Template{.header = (licenseHeader + R"(
#pragma once

#include "buffers.h"
#include "fp.h"
#include "fpext.h"
#include "witgen.h"

namespace )" + circuitName.getCppNamespace() +
                             R"(::cpu {

)")
                                .str(),
                  .footer = R"(
} // namespace )" + circuitName.getCppNamespace() +
                            R"(::cpu
)"};
}

Template CudaCodegenTarget::getStepTemplate() const {
  return Template{.header = (licenseHeader + R"(
#include "steps.cuh"
#include "witgen.h"

namespace )" + circuitName.getCppNamespace() +
                             R"(::cuda {

)")
                                .str(),
                  .footer = R"(
} // namespace )" + circuitName.getCppNamespace() +
                            R"(::cuda
)"};
}

Template CudaCodegenTarget::getStepDeclTemplate() const {
  return Template{.header = (licenseHeader + R"(
#pragma once

#include "witgen.h"

namespace )" + circuitName.getCppNamespace() +
                             R"(::cuda {
)")
                                .str(),
                  .footer = R"(
} // namespace )" + circuitName.getCppNamespace() +
                            R"(::cuda
)"};
}

} // namespace zirgen
