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

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use glob::Pattern;
use regex::Regex;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    io,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{exit, Command, Stdio},
};

#[derive(Debug)]
struct Rule<'a> {
    pattern: &'a str,
    dest_dir: &'a str,
    remove_prefix: &'a str,
    base_suffix: &'a str,
}

impl<'a> Rule<'a> {
    const fn copy(pattern: &'a str, dest_dir: &'a str) -> Self {
        Self {
            pattern,
            dest_dir,
            remove_prefix: "",
            base_suffix: "",
        }
    }

    const fn remove_prefix(self, remove_prefix: &'a str) -> Self {
        Self {
            remove_prefix,
            ..self
        }
    }

    const fn base_suffix(self, base_suffix: &'a str) -> Self {
        Self {
            base_suffix,
            ..self
        }
    }
}

type Rules<'a> = &'a [Rule<'a>];

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
    Keccak,
    Calculator,
    Verify,
    #[clap(name("bigint"))]
    BigInt,
    #[clap(name("bigint2"))]
    BigInt2,
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

    /// Show what would have been done instead of actually doing it.
    #[clap(long)]
    dry_run: bool,

    /// Instead of installing, check to make sure installed files are
    /// up to date.  Exits with an error if they're not.
    #[clap(long)]
    check: bool,
}

struct Bootstrap {
    args: Args,

    /// True if we've encountered an error.
    error: RefCell<bool>,
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

fn strip_locations_and_whitespace(text: &str) -> Result<String> {
    // replace everything that might be a line number with '*'
    let re = Regex::new(":[0-9]+")?;
    let stripped = re.replace_all(text, "*");
    // Remove all whitespace to avoid noticing reformatting changes
    Ok(stripped
        .as_ref()
        .chars()
        .filter(|&c| c != ' ' && c == '\t' && c != '\n')
        .collect())
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

impl Bootstrap {
    fn new(args: Args) -> Self {
        Self {
            args,
            error: RefCell::new(false),
        }
    }

    fn apply_rule(
        &self,
        rule: &Rule,
        src_file: impl AsRef<Path>,
        dest_base: impl AsRef<Path>,
    ) -> bool {
        let src_file = src_file.as_ref().to_path_buf();
        let mut dest_base = dest_base.as_ref().to_path_buf();

        let src_fn = src_file
            .file_name()
            .expect("Empty filename?")
            .to_str()
            .unwrap();

        if !Pattern::new(rule.pattern).unwrap().matches(src_fn) {
            return false;
        }
        if !rule.base_suffix.is_empty() {
            dest_base.set_file_name(
                [
                    dest_base.file_name().unwrap().to_str().unwrap(),
                    rule.base_suffix,
                ]
                .join(""),
            )
        }
        let dest_fn;
        if src_fn.starts_with(rule.remove_prefix) {
            dest_fn = &src_fn[rule.remove_prefix.len()..]
        } else {
            dest_fn = &src_fn
        }
        self.copy(
            &src_file,
            &dest_base.join(rule.dest_dir).join(Path::new(dest_fn)),
        );
        true
    }
    fn copy_group(
        &self,
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

            self.copy(&src_path, &tgt_dir.join(filename));
        }
    }

    fn copy_file(&self, src_dir: &Path, tgt_dir: &Path, filename: &str) {
        self.copy(&src_dir.join(filename), &tgt_dir.join(filename))
    }

    fn check(&self, src_path: &Path, tgt_path: &Path) -> Result<()> {
        let src = std::fs::read(src_path)?;
        let dst = std::fs::read(tgt_path)?;

        let src_fn = src_path.file_name().unwrap().to_str().unwrap();

        let textual: bool = src_fn.ends_with(".cu")
            || src_fn.ends_with(".h")
            || src_fn.ends_with(".inc")
            || src_fn.ends_with(".cuh")
            || src_fn.ends_with(".cpp")
            || src_fn.ends_with(".rs");

        if textual {
            // Things might change like line numbers or whitespace that don't have a functional impact.
            let src = strip_locations_and_whitespace(&String::from_utf8(src)?)?;
            let dst = strip_locations_and_whitespace(&String::from_utf8(dst)?)?;
            if src != dst {
                bail!(
                    "Mismatch: {} != {} (text)",
                    src_path.display(),
                    tgt_path.display()
                )
            }
        } else {
            if src != dst {
                bail!(
                    "Mismatch: {} != {} (binary)",
                    src_path.display(),
                    tgt_path.display()
                )
            }
        }

        Ok(())
    }

    fn copy(&self, src_path: &Path, tgt_path: &Path) {
        if self.args.dry_run {
            println!(
                "{} -> {} (dry run, skipping)",
                src_path.display(),
                tgt_path.display()
            );
            return;
        } else if self.args.check {
            if let Err(err) = self.check(src_path, tgt_path) {
                eprintln!("Error: {}", err);
                *self.error.borrow_mut() = true;
            };
        } else {
            println!("{} -> {}", src_path.display(), tgt_path.display());
            let tgt_dir = tgt_path.parent().unwrap();
            std::fs::create_dir_all(&tgt_dir).unwrap();

            // Avoid using `std::fs::copy` because bazel creates files with r/o permissions.
            // We create a new file with r/w permissions and .copy the contents instead.
            let mut src = std::fs::File::open(src_path)
                .expect(&format!("Could not open source: {}", src_path.display()));
            let mut tgt = std::fs::File::create(&tgt_path).unwrap();
            io::copy(&mut src, &mut tgt).unwrap();

            let filename = tgt_path.file_name().unwrap().to_str().unwrap();
            if filename.ends_with(".rs.inc") || filename.ends_with(".rs") {
                Command::new("rustfmt")
                    .args(tgt_path.to_str())
                    .status()
                    .expect(&format!("Unable to format {}", tgt_path.display()));
            }

            if filename.ends_with(".cpp.inc")
                || filename.ends_with(".cu.inc")
                || filename.ends_with(".cuh")
            {
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
    }

    fn output_or(&self, alternative: &str) -> PathBuf {
        self.args
            .output
            .clone()
            .unwrap_or(PathBuf::from(alternative))
    }

    fn build_all_circuits(&self) {
        // Build the circuits using bazel(isk).
        // TODO: Migrate to install_from_bazel which builds just what's necessary.

        let bazel_args = ["build", "--config", bazel_config(), "//zirgen/circuit"];

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
    }

    fn install_from_bazel(
        &self,
        build_target: &str,
        dest_base: impl AsRef<Path>,
        install_rules: Rules,
    ) {
        let dest_base = dest_base.as_ref();

        let bazel_args = ["build", "--config", bazel_config(), build_target];

        let mut child = Command::new("bazelisk")
            .args(bazel_args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Unable to run bazelisk");
        let child_out = BufReader::new(child.stderr.take().unwrap());

        let mut rules_used: Vec<bool> = Vec::new();
        rules_used.resize(install_rules.len(), false);

        for line in child_out.lines() {
            let line = line.unwrap();
            eprintln!("{line}");
            if !line.starts_with("  bazel-bin/") {
                continue;
            }

            let path = PathBuf::from(&line[2..]);

            let mut found_rule = false;
            for (idx, rule) in install_rules.iter().enumerate() {
                if self.apply_rule(rule, &path, dest_base) {
                    found_rule = true;
                    rules_used[idx] = true;
                }
            }
            if !found_rule {
                eprintln!("WARNING: Could not find rule to apply to {path:?}");
            }
        }

        let status = child.wait().expect("Unable to wait for bazel");

        if !status.success() {
            eprintln!("Bazel did not return success.");
            exit(status.code().unwrap());
        }

        for (used, rule) in rules_used.iter().zip(install_rules) {
            if !used {
                eprintln!("WARNING: Rule matched no files: {rule:?}");
            }
        }
    }

    fn run(&self) {
        match self.args.circuit {
            Circuit::Fib => self.fib(),
            Circuit::Predicates => self.predicates(),
            Circuit::Recursion => self.recursion(),
            Circuit::Rv32im => self.rv32im(),
            Circuit::Keccak => self.keccak(),
            Circuit::Calculator => self.calculator(),
            Circuit::Verify => self.stark_verify(),
            Circuit::BigInt => self.bigint(),
            Circuit::BigInt2 => self.bigint2(),
        }

        if self.args.check {
            if *self.error.borrow() {
                eprintln!("--check: Mismatches encountered");
                // Ado a different error message if we have errors during other operations
                exit(1);
            } else {
                eprintln!("--check: All installed files match");
            }
        } else {
            if *self.error.borrow() {
                panic!("Please add an appropriate error message here if we have defererred errors during a non-check operation");
            }
        }
    }

    fn fib(&self) {
        self.install_from_bazel(
            "//zirgen/circuit/fib",
            self.output_or("zirgen/circuit/fib"),
            &[
                Rule::copy("*.cpp", "cxx").remove_prefix("rust_"),
                Rule::copy("*.rs", "src"),
                Rule::copy("*.cu", "kernels"),
                Rule::copy("*.metal", "kernels"),
            ],
        );
        cargo_fmt_circuit("fib", &self.args.output, &None);
    }

    fn predicates(&self) {
        self.build_all_circuits();

        let bazel_bin = get_bazel_bin();
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let rust_path = risc0_root.join("circuit/recursion");
        let zkr_src_path = bazel_bin.join("zirgen/circuit/predicates");
        let zkr_tgt_path = rust_path.join("src");
        self.copy_file(&zkr_src_path, &zkr_tgt_path, RECURSION_ZKR_ZIP);
    }

    fn recursion(&self) {
        self.build_all_circuits();

        self.copy_edsl_style("recursion", "zirgen/circuit/recursion")
    }

    fn rv32im(&self) {
        self.build_all_circuits();

        self.copy_edsl_style("rv32im", "zirgen/circuit/rv32im/v1/edsl")
    }

    fn keccak(&self) {
        self.install_from_bazel(
            "//zirgen/circuit/keccak:circuit_and_zkr",
            self.output_or("zirgen/circuit/keccak"),
            &[
                Rule::copy("*.cpp", "cxx").base_suffix("-sys"),
                Rule::copy("*.cpp.inc", "cxx").base_suffix("-sys"),
                Rule::copy("*.h.inc", "cxx").base_suffix("-sys"),
                Rule::copy("*.cu", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cuh", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.rs", "src"),
                Rule::copy("*.rs.inc", "src"),
                Rule::copy("keccak_zkr.zip", "src/prove"),
            ],
        );
    }

    fn calculator(&self) {
        self.install_from_bazel(
            "//zirgen/dsl/examples/calculator",
            self.output_or("zirgen/dsl/examples/calculator"),
            &[Rule::copy("*.inc", ""), Rule::copy("*.rs", "")],
        );
        cargo_fmt_circuit("calculator", &self.args.output, &None);
    }

    fn copy_edsl_style(&self, circuit: &str, src_dir: &str) {
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let src_path = Path::new(src_dir);
        let rust_path = risc0_root.join("circuit").join(circuit);
        let rust_path = Some(rust_path);
        let sys_path = risc0_root
            .join("circuit")
            .join(String::from(circuit) + "-sys");
        let hal_root = Some(sys_path.join("kernels"));
        let sys_path = Some(sys_path);

        self.copy_group(
            circuit,
            &src_path,
            &sys_path,
            MAIN_CPP_OUTPUTS,
            "cxx",
            "rust_",
        );
        self.copy_group(circuit, &src_path, &rust_path, MAIN_RUST_OUTPUTS, "src", "");
        self.copy_group(circuit, &src_path, &hal_root, CUDA_OUTPUTS, "cuda", "");
        self.copy_group(circuit, &src_path, &hal_root, METAL_OUTPUTS, "metal", "");

        self.copy_group(
            circuit,
            &src_path,
            &sys_path,
            &["layout.cpp.inc"],
            "cxx",
            "",
        );
        self.copy_group(
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
        self.build_all_circuits();

        let bazel_bin = get_bazel_bin();
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let inc_path = Path::new("zirgen/circuit/verify/circom/include");
        let src_path = bazel_bin.join("zirgen/circuit/verify/circom");
        let tgt_path = risc0_root.join("groth16_proof/groth16");
        let rust_path = risc0_root.join("risc0/groth16/src");

        self.copy_file(&inc_path, &tgt_path, "risc0.circom");
        self.copy_file(&src_path, &tgt_path, "stark_verify.circom");
        self.copy_file(&src_path, &rust_path, "seal_format.rs");
        cargo_fmt(&risc0_root.join("risc0/groth16/Cargo.toml"));
    }

    fn bigint(&self) {
        self.build_all_circuits();

        let circuit = "bigint";
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let bigint_crate_root = risc0_root.join("circuit/bigint");
        let out = bigint_crate_root.join("src");

        let bazel_bin = get_bazel_bin();
        let src_path = bazel_bin.join("zirgen/circuit/bigint");

        self.copy_file(&src_path, &out, BIGINT_ZKR_ZIP);
        self.copy_group(
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

    fn bigint2(&self) {
        self.build_all_circuits();

        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let bazel_bin = get_bazel_bin();
        let src_path = bazel_bin.join("zirgen/circuit/bigint");
        let rsa_path = risc0_root.join("bigint2/src/rsa");
        let ec_path = risc0_root.join("bigint2/src/ec");

        self.copy_file(&src_path, &rsa_path, "modpow_65537.blob");
        self.copy(
            &src_path.join("ec_double.blob"),
            &ec_path.join("double.blob"),
        );
        self.copy(&src_path.join("ec_add.blob"), &ec_path.join("add.blob"));
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
    let bootstrap = Bootstrap::new(Args::parse());
    bootstrap.run();
}
