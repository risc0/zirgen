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
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    process::{exit, Command},
};

use clap::{Parser, ValueEnum};

const MAIN_CPP_OUTPUTS: &[&str] = &[
    "poly_fp.cpp",
    "step_exec.cpp",
    "step_verify_bytes.cpp",
    "step_verify_mem.cpp",
    "step_compute_accum.cpp",
    "step_verify_accum.cpp",
];

const MAIN_RUST_OUTPUTS: &[&str] = &["poly_ext.rs", "taps.rs", "info.rs", "layout.rs.inc"];

const CUDA_OUTPUTS: &[&str] = &[
    "step_exec.cu",
    "step_verify_bytes.cu",
    "step_verify_mem.cu",
    "step_compute_accum.cu",
    "step_verify_accum.cu",
    "eval_check.cu",
];

const METAL_OUTPUTS: &[&str] = &[
    // "step_exec.metal",
    // "step_verify_bytes.metal",
    // "step_verify_mem.metal",
    "step_compute_accum.metal",
    "step_verify_accum.metal",
    "eval_check.metal",
];

const RECURSION_ZKR_ZIP: &str = "recursion_zkr.zip";

const BIGINT_OUTPUTS: &[&str] = &["bigint.rs.inc"];
const BIGINT_ZKR_ZIP: &str = "bigint_zkr.zip";

#[derive(Clone, Debug, ValueEnum)]
enum Circuit {
    Fib,
    Predicates,
    Recursion,
    Rv32im,
    Verify,
    #[clap(name("bigint"))]
    BigInt,
}

#[derive(Parser)]
#[clap(about, version, author)]
struct Args {
    /// Which circuit to bootstrap.
    #[clap(value_enum)]
    circuit: Circuit,

    /// Output path for the generated circuit files.
    #[clap(long)]
    output: Option<PathBuf>,
}

fn get_bazel_bin() -> PathBuf {
    let bazel_bin = Command::new("bazelisk")
        .arg("info")
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
            Circuit::Fib => self.fib(),
            Circuit::Predicates => self.predicates(),
            Circuit::Recursion => self.recursion(),
            Circuit::Rv32im => self.rv32im(),
            Circuit::Verify => self.stark_verify(),
            Circuit::BigInt => self.bigint(),
        }
    }

    fn fib(&self) {
        let circuit = "fib";
        let src_path = Path::new("zirgen/circuit/fib");
        let out = &self.output;
        copy_group(circuit, &src_path, out, MAIN_CPP_OUTPUTS, "cxx", "rust_");
        copy_group(circuit, &src_path, out, MAIN_RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src_path, out, CUDA_OUTPUTS, "kernels", "");
        copy_group(circuit, &src_path, out, METAL_OUTPUTS, "kernels", "");
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
        let circuit = "recursion";
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let src_path = Path::new("zirgen/circuit/recursion");
        let rust_path = risc0_root.join("circuit/recursion");
        let rust_path = Some(rust_path);
        let sys_path = risc0_root.join("circuit/recursion-sys");
        let hal_root = Some(sys_path.join("kernels"));
        let sys_path = Some(sys_path);

        copy_group(
            circuit,
            &src_path,
            &sys_path,
            MAIN_CPP_OUTPUTS,
            "cxx",
            "rust_",
        );
        copy_group(circuit, &src_path, &rust_path, MAIN_RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src_path, &hal_root, CUDA_OUTPUTS, "cuda", "");
        copy_group(circuit, &src_path, &hal_root, METAL_OUTPUTS, "metal", "");
        // copy_group(circuit, &src_path, &rust_path, LAYOUT_OUTPUTS, "src", "");
        copy_group(
            circuit,
            &src_path,
            &sys_path,
            &["layout.cpp.inc"],
            "cxx",
            "",
        );
        copy_group(
            circuit,
            &src_path,
            &hal_root,
            &["layout.cu.inc"],
            "cuda",
            "",
        );
        cargo_fmt_circuit(circuit, &rust_path, &None);
    }

    fn rv32im(&self) {
        let circuit = "rv32im";
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let src_path = Path::new("zirgen/circuit/rv32im/v1/edsl");
        let rust_path = Some(risc0_root.join("circuit/rv32im"));
        let sys_path = risc0_root.join("circuit/rv32im-sys");
        let hal_root = Some(sys_path.join("kernels"));
        let sys_path = Some(sys_path);

        copy_group(
            circuit,
            &src_path,
            &sys_path,
            MAIN_CPP_OUTPUTS,
            "cxx",
            "rust_",
        );
        copy_group(circuit, &src_path, &rust_path, MAIN_RUST_OUTPUTS, "src", "");
        copy_group(circuit, &src_path, &hal_root, CUDA_OUTPUTS, "cuda", "");
        copy_group(circuit, &src_path, &hal_root, METAL_OUTPUTS, "metal", "");
        copy_group(
            circuit,
            &src_path,
            &sys_path,
            &["layout.cpp.inc"],
            "cxx",
            "",
        );
        copy_group(
            circuit,
            &src_path,
            &hal_root,
            &["layout.cu.inc"],
            "cuda",
            "",
        );
        cargo_fmt_circuit(circuit, &rust_path, &None);
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

    fn bigint(&self) {
        let circuit = "bigint";
        let risc0_root = self.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let bigint_crate_root = risc0_root.join("circuit/bigint");
        let out = bigint_crate_root.join("src");

        let bazel_bin = get_bazel_bin();
        let src_path = bazel_bin.join("zirgen/circuit/bigint");

        copy_file(&src_path, &out, BIGINT_ZKR_ZIP);
        copy_group(
            circuit,
            &src_path,
            &Some(out.clone()),
            BIGINT_OUTPUTS,
            "",
            "",
        );

        // Generate control IDs

        // Remove magic rust environment variables for the
        // risczero-wip repositroy so we can use the risc0 repository
        // settings.
        let filtered_env: BTreeMap<String, String> = std::env::vars()
            .filter(|&(ref k, _)| !k.starts_with("CARGO") && !k.starts_with("RUSTUP"))
            .collect();

        let output = Command::new("cargo")
            .current_dir(&risc0_root)
            .env_clear()
            .envs(filtered_env)
            .arg("run")
            .arg("-p")
            .arg("risc0-circuit-bigint")
            .arg("-F")
            .arg("make_control_ids")
            .arg("--bin")
            .arg("make_control_ids")
            .output()
            .unwrap();

        if !output.status.success() {
            panic!(
                "Failed to generate bigint control_ids:\n{}",
                String::from_utf8(output.stderr).unwrap()
            );
        }

        std::fs::write(out.join("control_id.rs"), output.stdout).unwrap();

        cargo_fmt_circuit(circuit, &Some(bigint_crate_root), &None);
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    // Use a hermetic C++ toolchain to get consistent builds across host platforms.
    // See: https://github.com/uber/hermetic_cc_toolchain
    // Also ensure that these configs exist in `.bazelrc`.
    let config = if cfg!(target_os = "macos") {
        "bootstrap_macos_arm64"
    } else {
        "bootstrap_linux_amd64"
    };
    let bazel_args = ["build", "--config", config, "//zirgen/circuit/..."];

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
