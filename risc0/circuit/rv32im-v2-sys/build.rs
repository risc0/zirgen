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

use std::env;

use risc0_build_kernel::{KernelBuild, KernelType};

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    build_cpu_kernels();
}

fn build_cpu_kernels() {
    KernelBuild::new(KernelType::Cpp)
        .files(glob::glob("kernels/cxx/*.cpp").unwrap().map(|x| x.unwrap()))
        .deps(glob::glob("kernels/cxx/*.h").unwrap().map(|x| x.unwrap()))
        .deps(
            glob::glob("kernels/cxx/*.cpp.inc")
                .unwrap()
                .map(|x| x.unwrap()),
        )
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_rv32im_v2_cpu");
}

fn build_cuda_kernels() {
    KernelBuild::new(KernelType::Cuda)
        .files([
            "kernels/cuda/eval_check_0.cu",
            "kernels/cuda/eval_check_1.cu",
            "kernels/cuda/eval_check_2.cu",
            "kernels/cuda/eval_check_3.cu",
            "kernels/cuda/eval_check_4.cu",
            "kernels/cuda/eval_check.cu",
            "kernels/cuda/ffi.cu",
            "kernels/cuda/ffi_supra.cu",
        ])
        .deps([
            // "kernels/cuda/context.h",
            "kernels/cuda/defs.cu.inc",
            // "kernels/cuda/extern.h",
            "kernels/cuda/kernels.h",
            "kernels/cuda/layout.cu.inc",
            "kernels/cuda/steps.cu.inc",
            "kernels/cuda/types.cu.inc",
        ])
        .include(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .compile("risc0_rv32im_v2_cuda");
}
