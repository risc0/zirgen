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

use std::{
    io,
    path::{Path, PathBuf},
    process::{exit, Command},
};

use clap::{Parser, ValueEnum};

mod edsl {
    pub const RUST_OUTPUTS: &[&str] = &["poly_ext.rs", "taps.rs", "info.rs", "layout.rs.inc"];

    pub const CPP_OUTPUTS: &[&str] = &[
        "poly_fp.cpp",
        "step_exec.cpp",
        "step_verify_bytes.cpp",
        "step_verify_mem.cpp",
        "step_compute_accum.cpp",
        "step_verify_accum.cpp",
    ];

    pub const CUDA_OUTPUTS: &[&str] = &[
        "step_exec.cu",
        "step_verify_bytes.cu",
        "step_verify_mem.cu",
        "step_compute_accum.cu",
        "step_verify_accum.cu",
        "eval_check.cu",
    ];

    pub const METAL_OUTPUTS: &[&str] = &[
        // "step_exec.metal",
        // "step_verify_bytes.metal",
        // "step_verify_mem.metal",
        "step_compute_accum.metal",
        "step_verify_accum.metal",
        "eval_check.metal",
    ];
}

mod zirgen {
    pub const RUST_OUTPUTS: &[&str] = &[
        "taps.rs",
        "info.rs",
        "poly_ext.rs",
        "defs.rs.inc",
        "types.rs.inc",
        "layout.rs.inc",
        "steps.rs.inc",
    ];

    pub const SYS_OUTPUTS: &[&str] = &[
        "defs.cpp.inc",
        "types.h.inc",
        "layout.cpp.inc",
        "steps.cpp.inc",
    ];

    pub const CPP_OUTPUTS: &[&str] = &[
        "defs.cpp.inc",
        "types.h.inc",
        "layout.cpp.inc",
        "steps.cpp.inc",
        "rust_poly_fp_0.cpp",
        "rust_poly_fp_1.cpp",
        "rust_poly_fp_2.cpp",
        "rust_poly_fp_3.cpp",
        "rust_poly_fp_4.cpp",
    ];

    pub const CUDA_OUTPUTS: &[&str] = &[
        "defs.cu.inc",
        "eval_check_0.cu",
        "eval_check_1.cu",
        "eval_check_2.cu",
        "eval_check_3.cu",
        "eval_check_4.cu",
        "layout.cu.inc",
        "steps.cu.inc",
        "types.cuh.inc",
    ];

    pub const METAL_OUTPUTS: &[&str] = &[];
}

const RECURSION_ZKR_ZIP: &str = "recursion_zkr.zip";

const CALCULATOR_RUST_OUTPUTS: &[&str] = &[
    "taps.rs",
    "info.rs",
    "poly_ext.rs",
    "defs.rs.inc",
    "types.rs.inc",
    "layout.rs.inc",
    "steps.rs.inc",
    "validity.rs.inc",
];

const KECCAK_ZKR_ZIP: &str = "keccak_zkr.zip";

#[derive(Clone, Debug, ValueEnum)]
enum Circuit {
    Bigint2,
    Calculator,
    Fib,
    Keccak,
    Predicates,
    Recursion,
    Rv32im,
    Rv32imV2,
    Verify,
}

#[derive(Parser)]
#[clap(about, version, author)]
struct Args {
    /// Which circuit to bootstrap.
    #[clap(value_enum)]
    circuit: Circuit,

    /// Output path for the generated circuit files.
    ///
    /// When bootstapping the risc0 monorepo, this should be the path to the repo root.
    #[clap(long)]
    output: Option<PathBuf>,
}

fn get_bazel_bin() -> PathBuf {
    let bazel_bin = Command::new("bazelisk")
        .arg("info")
        .arg("--config")
        .arg(bazel_config())
        .arg("bazel-bin")
        .output()
        .unwrap();
    let bazel_bin = String::from_utf8(bazel_bin.stdout).unwrap();
    Path::new(bazel_bin.trim()).to_path_buf()
}

fn copy_group(
    circuit: &str,
    src_path: &Path,
    out_dir: &Option<PathBuf>,
    outputs: &[&str],
    child_dir: &str,
    prefix: &str,
) {
    let bazel_bin = get_bazel_bin();
    for filename in outputs {
        let src_filename = format!("{prefix}{filename}");
        let src_path = bazel_bin.join(src_path).join(src_filename);
        let tgt_dir = if let Some(ref out_dir) = out_dir {
            PathBuf::from(out_dir)
        } else {
            PathBuf::from("zirgen/circuit").join(circuit)
        }
        .join(child_dir);

        copy(&src_path, &tgt_dir.join(filename));
    }
}

fn copy_file(src_dir: &Path, tgt_dir: &Path, filename: &str) {
    copy(&src_dir.join(filename), &tgt_dir.join(filename))
}

fn copy(src_path: &Path, tgt_path: &Path) {
    println!("{} -> {}", src_path.display(), tgt_path.display());

    let tgt_dir = tgt_path.parent().unwrap();
    std::fs::create_dir_all(&tgt_dir).unwrap();

    // Avoid using `std::fs::copy` because bazel creates files with r/o permissions.
    // We create a new file with r/w permissions and copy the contents instead.
    let mut src = std::fs::File::open(src_path)
        .expect(&format!("Could not open source: {}", src_path.display()));
    let mut tgt = std::fs::File::create(&tgt_path).unwrap();
    io::copy(&mut src, &mut tgt).unwrap();

    let filename = tgt_path.file_name().unwrap().to_str().unwrap();
    if filename.ends_with(".rs.inc") {
        // Cargo will not format these automatically, so do so by hand
        Command::new("rustfmt")
            .args(tgt_path.to_str())
            .status()
            .expect(&format!("Unable to format {}", tgt_path.display()));
    }

    if filename.ends_with(".cpp.inc") || filename.ends_with(".cu.inc") {
        // Cargo will not format these automatically, so do so by hand
        if let Err(err) = Command::new("clang-format")
            .args(["-i", tgt_path.to_str().unwrap()])
            .status()
        {
            eprintln!(
                "Warning: unable to format {} with clang-format: {err}",
                tgt_path.display()
            );
        }
    }
}

fn cargo_fmt_circuit(circuit: &str, tgt_dir: &Option<PathBuf>, manifest: &Option<PathBuf>) {
    let tgt_dir = if let Some(ref tgt_dir) = tgt_dir {
        PathBuf::from(tgt_dir)
    } else {
        PathBuf::from("zirgen/circuit").join(circuit)
    };
    let path = if let Some(ref manifest) = manifest {
        PathBuf::from(manifest).join("Cargo.toml")
    } else {
        tgt_dir.join("Cargo.toml")
    };
    if path.exists() {
        cargo_fmt(&path);
    }
}

fn cargo_fmt(manifest: &Path) {
    let status = Command::new("cargo")
        .arg("fmt")
        .arg("--manifest-path")
        .arg(manifest)
        .status()
        .unwrap();
    if !status.success() {
        exit(status.code().unwrap());
    }
}

impl Args {
    fn run(&self) {
        match self.circuit {
            Circuit::Bigint2 => self.bigint2(),
            Circuit::Calculator => self.calculator(),
            Circuit::Fib => self.fib(),
            Circuit::Keccak => self.keccak(),
            Circuit::Predicates => self.predicates(),
            Circuit::Recursion => self.recursion(),
            Circuit::Rv32im => self.rv32im(),
            Circuit::Rv32imV2 => self.rv32im_v2(),
            Circuit::Verify => self.stark_verify(),
        }
    }

    fn fib(&self) {
        let circuit = "fib";
        let src_path = Path::new("zirgen/circuit/fib");
        let out = &self.output;
        copy_group(circuit, &src_path, out, edsl::CPP_OUTPUTS, "cxx", "rust_");
        copy_group(circuit, &src_path, out, edsl::RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src_path, out, edsl::CUDA_OUTPUTS, "kernels", "");
        copy_group(circuit, &src_path, out, edsl::METAL_OUTPUTS, "kernels", "");
        cargo_fmt_circuit(circuit, &self.output, &None);
    }

    fn predicates(&self) {
        let bazel_bin = get_bazel_bin();
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let rust_path = risc0_root.join("circuit/recursion");
        let zkr_src_path = bazel_bin.join("zirgen/circuit/predicates");
        let zkr_tgt_path = rust_path.join("src");
        copy_file(&zkr_src_path, &zkr_tgt_path, RECURSION_ZKR_ZIP);
    }

    fn recursion(&self) {
        self.copy_edsl_style("recursion", "zirgen/circuit/recursion")
    }

    fn rv32im(&self) {
        self.copy_edsl_style("rv32im", "zirgen/circuit/rv32im/v1/edsl")
    }

    fn rv32im_v2(&self) {
        self.copy_zirgen_style("rv32im-v2", "zirgen/circuit/rv32im/v2/dsl")
    }

    fn keccak(&self) {
        let bazel_bin = get_bazel_bin();
        let circuit = "keccak";
        let src_path = Path::new("zirgen/circuit/keccak");
        let sys_root = Path::new("zirgen/circuit/keccak-sys").to_path_buf();
        let hal_root = Some(sys_root.join("kernels"));
        let zkr_src_path = bazel_bin.join("zirgen/circuit/predicates");
        let zkr_tgt_path = self
            .output
            .clone()
            .unwrap_or(Path::new("zirgen/circuit/keccak/src/prove").to_path_buf());

        copy_group(
            circuit,
            &src_path,
            &self.output,
            zirgen::RUST_OUTPUTS,
            "src",
            "",
        );
        copy_group(
            circuit,
            &src_path,
            &Some(sys_root),
            zirgen::SYS_OUTPUTS,
            "cxx",
            "",
        );
        copy_group(
            circuit,
            &src_path,
            &hal_root,
            zirgen::CUDA_OUTPUTS,
            "cuda",
            "",
        );
        copy_file(&zkr_src_path, &zkr_tgt_path, KECCAK_ZKR_ZIP);
        cargo_fmt_circuit(circuit, &self.output, &None);
    }

    fn calculator(&self) {
        let out = self.output.clone().or(Some(
            Path::new("zirgen/dsl/examples/calculator").to_path_buf(),
        ));
        let circuit = "calculator";
        let src_path = Path::new("zirgen/dsl/examples/calculator/");

        copy_group(circuit, &src_path, &out, CALCULATOR_RUST_OUTPUTS, "", "");
        copy_group(circuit, &src_path, &out, zirgen::SYS_OUTPUTS, "", "");
        cargo_fmt_circuit(circuit, &self.output, &None);
    }

    fn copy_zirgen_style(&self, circuit: &str, src_dir: &str) {
        let root = std::env::current_dir().unwrap().join("risc0");
        let src = Path::new(src_dir);
        let rust = Some(root.join("circuit").join(circuit));
        let kernels = root
            .join("circuit")
            .join(String::from(circuit) + "-sys")
            .join("kernels");
        let kernels = Some(kernels);

        copy_group(circuit, &src, &rust, zirgen::RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src, &kernels, zirgen::CPP_OUTPUTS, "cxx", "");
        copy_group(circuit, &src, &kernels, zirgen::CUDA_OUTPUTS, "cuda", "");
        copy_group(circuit, &src, &kernels, zirgen::METAL_OUTPUTS, "metal", "");

        cargo_fmt_circuit(circuit, &rust, &None);
    }

    fn copy_edsl_style(&self, circuit: &str, src_dir: &str) {
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let src = Path::new(src_dir);
        let rust = Some(risc0_root.join("circuit").join(circuit));
        let sys = risc0_root
            .join("circuit")
            .join(String::from(circuit) + "-sys");
        let kernels = Some(sys.join("kernels"));
        let sys = Some(sys);

        copy_group(circuit, &src, &rust, edsl::RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src, &sys, edsl::CPP_OUTPUTS, "cxx", "rust_");
        copy_group(circuit, &src, &kernels, edsl::CUDA_OUTPUTS, "cuda", "");
        copy_group(circuit, &src, &kernels, edsl::METAL_OUTPUTS, "metal", "");

        copy_group(circuit, &src, &sys, &["layout.cpp.inc"], "cxx", "");
        copy_group(circuit, &src, &kernels, &["layout.cu.inc"], "cuda", "");
        cargo_fmt_circuit(circuit, &rust, &None);
    }

    fn stark_verify(&self) {
        let bazel_bin = get_bazel_bin();
        let risc0_root = self.output.as_ref().expect("--output is required");
        let inc_path = Path::new("zirgen/circuit/verify/circom/include");
        let src_path = bazel_bin.join("zirgen/circuit/verify/circom");
        let tgt_path = risc0_root.join("groth16_proof/groth16");
        let rust_path = risc0_root.join("risc0/groth16/src");

        copy_file(&inc_path, &tgt_path, "risc0.circom");
        copy_file(&src_path, &tgt_path, "stark_verify.circom");
        copy_file(&src_path, &rust_path, "seal_format.rs");
        cargo_fmt(&risc0_root.join("risc0/groth16/Cargo.toml"));
    }

    fn bigint2(&self) {
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let bazel_bin = get_bazel_bin();
        let src_path = bazel_bin.join("zirgen/circuit/bigint");
        let rsa_path = risc0_root.join("bigint2/src/rsa");
        let ec_path = risc0_root.join("bigint2/src/ec");

        copy_file(&src_path, &rsa_path, "modpow_65537.blob");
        copy(
            &src_path.join("ec_double.blob"),
            &ec_path.join("double.blob"),
        );
        copy(&src_path.join("ec_add.blob"), &ec_path.join("add.blob"));
    }
}

fn bazel_config() -> &'static str {
    // Use a hermetic C++ toolchain to get consistent builds across host platforms.
    // See: https://github.com/uber/hermetic_cc_toolchain
    // Also ensure that these configs exist in `.bazelrc`.
    if cfg!(target_os = "macos") {
        "bootstrap_macos_arm64"
    } else {
        "bootstrap_linux_amd64"
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    let bazel_args = ["build", "--config", bazel_config(), "//zirgen/circuit"];

    // Build the circuits using bazel(isk).
    let status = Command::new("bazelisk").args(bazel_args).status();
    let status = match status {
        Ok(stat) => stat,
        Err(err) => match err.kind() {
            std::io::ErrorKind::NotFound => {
                Command::new("bazel").args(bazel_args).status().unwrap()
            }
            _ => panic!("{}", err.to_string()),
        },
    };
    if !status.success() {
        exit(status.code().unwrap());
    }

    args.run();
}
