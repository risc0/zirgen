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

use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "zkvm" {
        let srcs: Vec<PathBuf> = glob::glob("cxx/*.cpp")
            .unwrap()
            .map(|x| x.unwrap())
            .collect();
        cc::Build::new()
            .cpp(true)
            .files(&srcs)
            .flag_if_supported("/std:c++17")
            .flag_if_supported("-std=c++17")
            .compile("circuit");
        for src in srcs {
            println!("cargo:rerun-if-changed={}", src.display());
        }
    }
}
