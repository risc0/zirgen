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

use std::{env, path::Path};

use risc0_build_kernel::{KernelBuild, KernelType};

fn main() {
    build_cpu_kernels();

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    if env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos" || os == "ios") {
        build_metal_kernels();
    }
}

fn build_cpu_kernels() {
    KernelBuild::new(KernelType::Cpp)
        .files([
            "cxx/ffi.cpp",
            "cxx/rust_poly_fp_0.cpp",
            "cxx/rust_poly_fp_1.cpp",
            "cxx/rust_poly_fp_2.cpp",
            "cxx/rust_poly_fp_3.cpp",
            "cxx/rust_poly_fp_4.cpp",
        ])
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("circuit");
}

fn build_cuda_kernels() {
    KernelBuild::new(KernelType::Cuda)
        .files([
            "kernels/cuda/eval_check_0.cu",
            "kernels/cuda/eval_check_1.cu",
            "kernels/cuda/eval_check_2.cu",
            "kernels/cuda/eval_check_3.cu",
            "kernels/cuda/eval_check_4.cu",
            "kernels/cuda/ffi.cu",
            "kernels/cuda/ffi_supra.cu",
        ])
        .deps(["kernels/cuda/kernels.h"])
        .include(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .compile("risc0_keccak_cuda");
}

fn build_metal_kernels() {
    const SRCS: &[&str] = &[
        "eval_check.metal",
    ];

    let dir = Path::new("kernels").join("metal");
    let src_paths = SRCS.iter().map(|x| dir.join(x));

    KernelBuild::new(KernelType::Metal)
        .files(src_paths)
        .compile("metal_kernel");
}
