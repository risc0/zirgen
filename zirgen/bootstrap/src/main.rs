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

use std::{
    io::{self, stdout, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{exit, Command, Stdio},
};

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use glob::Pattern;
use regex::Regex;
use std::sync::Mutex;
use threadpool::ThreadPool;
use xz2::read::XzEncoder;

#[derive(Debug)]
struct Rule<'a> {
    pattern: &'a str,
    dest_dir: &'a str,
    remove_prefix: &'a str,
    base_suffix: &'a str,
    compressed: bool,
}

impl<'a> Rule<'a> {
    const fn copy(pattern: &'a str, dest_dir: &'a str) -> Self {
        Self {
            pattern,
            dest_dir,
            remove_prefix: "",
            base_suffix: "",
            compressed: false,
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

    const fn compressed(self) -> Self {
        Self {
            compressed: true,
            ..self
        }
    }
}

type Rules<'a> = &'a [Rule<'a>];

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
    error: Mutex<bool>,

    pool: ThreadPool,
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
    // replace everything that might be a line number or a copyright date with '*'
    let re = Regex::new("(:[0-9]+)|(Copyright [0-9]+ RISC)")?;
    let stripped = re.replace_all(text, "*");

    // Remove anything that might be changed by reformatting
    Ok(stripped
        .as_ref()
        .chars()
        .filter(|&c| {
            // Reformatting may change whitespace
            c != ' ' && c != '\t' && c != '\n'
            // Reformatting may add commas at the end of lists
            && c != ','
            // Reformatting may break apart long strings into separate lines
            && c != '\\' && c != '"'
        })
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
            error: Mutex::new(false),
            pool: threadpool::Builder::new().build(),
        }
    }

    fn apply_rule(
        &'static self,
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
        let dest_fn = src_fn.strip_prefix(rule.remove_prefix).unwrap_or(src_fn);
        let mut dst_file = dest_base.join(rule.dest_dir).join(Path::new(dest_fn));

        if rule.compressed {
            let mut dst_file_name = dst_file.file_name().unwrap().to_os_string();
            dst_file_name.push(std::ffi::OsStr::new(".xz"));
            dst_file.set_file_name(dst_file_name);
        }

        let src_data = std::fs::File::open(&src_file)
            .unwrap_or_else(|_| panic!("Could not open source: {}", src_file.display()));
        if rule.compressed {
            let compressed_data = XzEncoder::new(src_data, 6);
            self.copy(&src_file, compressed_data, &dst_file);
        } else {
            self.copy(&src_file, src_data, &dst_file);
        }
        true
    }

    fn copy_group(
        &'static self,
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

            let src_data = std::fs::File::open(&src_path)
                .unwrap_or_else(|_| panic!("Could not open source: {}", src_path.display()));
            self.copy(&src_path, src_data, &tgt_dir.join(filename));
        }
    }

    fn copy_file(&'static self, src_dir: &Path, tgt_dir: &Path, filename: &str) {
        let src_file = src_dir.join(filename);
        let src_data = std::fs::File::open(&src_file)
            .unwrap_or_else(|_| panic!("Could not open source: {}", src_file.display()));
        self.copy(&src_file, src_data, &tgt_dir.join(filename))
    }

    fn check(&self, src_path: &Path, src_data: impl Read, tgt_path: &Path) -> Result<()> {
        let src: Vec<u8> = src_data.bytes().collect::<std::io::Result<Vec<u8>>>()?;
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
                bail!("Text does not match")
            }
        } else if src != dst {
            bail!("Binary blobs do not match")
        }

        Ok(())
    }

    fn copy(
        &'static self,
        src_path: &Path,
        mut src_data: impl Read + Send + 'static,
        tgt_path: &Path,
    ) {
        // Allocate copies for thread pool work item
        let src_path = src_path.to_path_buf();
        let tgt_path = tgt_path.to_path_buf();

        if self.args.dry_run {
            println!(
                "{} -> {} (dry run, skipping)",
                src_path.display(),
                tgt_path.display()
            );
        } else if self.args.check {
            self.pool.execute(move || {
                if let Err(err) = self.check(&src_path, src_data, &tgt_path) {
                    write!(
                        stdout().lock(),
                        "ERROR: {} != {}: {}\n",
                        src_path.display(),
                        tgt_path.display(),
                        err
                    )
                    .unwrap();
                    *self.error.lock().unwrap() = true;
                } else {
                    write!(
                        stdout().lock(),
                        "{} == {}\n",
                        src_path.display(),
                        tgt_path.display()
                    )
                    .unwrap();
                }
            });
        } else {
            let tgt_dir = tgt_path.parent().unwrap();
            std::fs::create_dir_all(tgt_dir).unwrap();

            self.pool.execute(move || {
                // Avoid using `std::fs::copy` because bazel creates files with r/o permissions.
                // We create a new file with r/w permissions and .copy the contents instead.
                let mut tgt = std::fs::File::create(&tgt_path).unwrap();
                io::copy(&mut src_data, &mut tgt).unwrap();

                let filename = tgt_path.file_name().unwrap().to_str().unwrap();
                if filename.ends_with(".rs.inc") || filename.ends_with(".rs") {
                    Command::new("rustfmt")
                        .args(tgt_path.to_str())
                        .status()
                        .unwrap_or_else(|_| panic!("Unable to format {}", tgt_path.display()));
                }

                if filename.ends_with(".cpp.inc")
                    || filename.ends_with(".cu.inc")
                    || filename.ends_with(".cuh")
                    || filename.ends_with(".cu")
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

                write!(
                    stdout().lock(),
                    "{} -> {}\n",
                    src_path.display(),
                    tgt_path.display()
                )
                .unwrap();
            });
        }
    }

    fn output_or(&self, alternative: &str) -> PathBuf {
        self.args
            .output
            .clone()
            .unwrap_or(PathBuf::from(alternative))
    }

    fn output_and(&self, relative: &str) -> PathBuf {
        self.args
            .output
            .clone()
            .expect("Output directory must be specified for external circuit")
            .join(relative)
            .to_path_buf()
    }

    fn build_all_circuits(&self) {
        // Build the circuits using bazel(isk).
        // TODO: Migrate to install_from_bazel which builds just what's necessary.

        let bazel_args = ["build", "--config", bazel_config(), "//zirgen/circuit"];

        let status = Command::new("bazelisk").args(bazel_args).status();
        let status = match status {
            Ok(stat) => stat,
            Err(err) => match err.kind() {
                io::ErrorKind::NotFound => Command::new("bazel").args(bazel_args).status().unwrap(),
                _ => panic!("{}", err.to_string()),
            },
        };
        if !status.success() {
            exit(status.code().unwrap());
        }
    }

    fn install_from_bazel(
        &'static self,
        build_target: &str,
        dest_base: impl AsRef<Path>,
        install_rules: Rules,
    ) {
        let dest_base = dest_base.as_ref();

        let bazel_args = ["build", "--config", bazel_config(), build_target];

        let mut command = Command::new("bazelisk");
        command
            .args(bazel_args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped());

        if std::env::var("CC").is_ok_and(|cc| cc.contains(" ")) {
            // HACK: Rust in CI gives us a CC of "sccache clang", which bazel can't handle.
            // TODO: find a better way of handling this.
            command.env_remove("CC");
        }

        let mut child = command.spawn().expect("Unable to run bazelisk");
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

    fn run(&'static self) {
        match self.args.circuit {
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

        self.pool.join();

        if self.args.check {
            if *self.error.lock().unwrap() {
                eprintln!("--check: Mismatches encountered");
                // Ado a different error message if we have errors during other operations
                exit(1);
            } else {
                eprintln!("--check: All installed files match");
            }
        } else if *self.error.lock().unwrap() {
            panic!("Please add an appropriate error message here if we have defererred errors during a non-check operation");
        }
    }

    fn fib(&'static self) {
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

    fn predicates(&'static self) {
        self.install_from_bazel(
            "//zirgen/circuit/predicates:recursion_zkr",
            self.output_and("risc0/circuit/recursion"),
            &[Rule::copy("*.zip", "src")],
        );
    }

    fn recursion(&'static self) {
        self.build_all_circuits();
        self.copy_edsl_style("recursion", "zirgen/circuit/recursion")
    }

    fn rv32im(&'static self) {
        self.build_all_circuits();
        self.copy_edsl_style("rv32im", "zirgen/circuit/rv32im/v1/edsl")
    }

    fn rv32im_v2(&'static self) {
        self.install_from_bazel(
            "//zirgen/circuit/rv32im/v2/dsl:codegen",
            self.output_and("risc0/circuit/rv32im-v2"),
            &[
                Rule::copy("*.cpp", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.cpp.inc", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.h", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.h.inc", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.cu", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cu.inc", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cuh", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cuh.inc", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.rs", "src/zirgen"),
                Rule::copy("*.rs.inc", "src/zirgen"),
            ],
        );
    }

    fn keccak(&'static self) {
        self.install_from_bazel(
            "//zirgen/circuit/keccak2:bootstrap",
            self.output_and("risc0/circuit/keccak"),
            &[
                Rule::copy("*.cpp", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.cpp.inc", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.h.inc", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.h", "kernels/cxx").base_suffix("-sys"),
                Rule::copy("*.cu", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cu.inc", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cuh", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.cuh.inc", "kernels/cuda").base_suffix("-sys"),
                Rule::copy("*.rs", "src/zirgen"),
                Rule::copy("*.rs.inc", "src/zirgen"),
                Rule::copy("*.zkr", "src/prove").compressed(),
            ],
        );
    }

    fn calculator(&'static self) {
        self.install_from_bazel(
            "//zirgen/dsl/examples/calculator",
            self.output_or("zirgen/dsl/examples/calculator"),
            &[Rule::copy("*.inc", ""), Rule::copy("*.rs", "")],
        );
    }

    fn copy_edsl_style(&'static self, circuit: &str, src_dir: &str) {
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let src = Path::new(src_dir);
        let rust = Some(risc0_root.join("circuit").join(circuit));
        let sys = risc0_root
            .join("circuit")
            .join(String::from(circuit) + "-sys");
        let kernels = Some(sys.join("kernels"));
        let sys = Some(sys);

        self.copy_group(circuit, src, &rust, edsl::RUST_OUTPUTS, "src", "");
        self.copy_group(circuit, src, &sys, edsl::CPP_OUTPUTS, "cxx", "rust_");
        self.copy_group(circuit, src, &kernels, edsl::CUDA_OUTPUTS, "cuda", "");
        self.copy_group(circuit, src, &kernels, edsl::METAL_OUTPUTS, "metal", "");

        self.copy_group(circuit, src, &sys, &["layout.cpp.inc"], "cxx", "");
        self.copy_group(circuit, src, &kernels, &["layout.cu.inc"], "cuda", "");
        cargo_fmt_circuit(circuit, &rust, &None);
    }

    fn stark_verify(&'static self) {
        self.build_all_circuits();

        let bazel_bin = get_bazel_bin();
        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let inc_path = Path::new("zirgen/circuit/verify/circom/include");
        let src_path = &bazel_bin.join("zirgen/circuit/verify/circom");
        let tgt_path = risc0_root.join("groth16_proof/groth16");
        let rust_path = risc0_root.join("risc0/groth16/src");

        self.copy_file(inc_path, &tgt_path, "risc0.circom");
        self.copy_file(src_path, &tgt_path, "stark_verify.circom");
        self.copy_file(src_path, &rust_path, "seal_format.rs");
        cargo_fmt(&risc0_root.join("risc0/groth16/Cargo.toml"));
    }

    fn bigint2(&'static self) {
        self.build_all_circuits();

        let risc0_root = self.args.output.as_ref().expect("--output is required");
        let risc0_root = risc0_root.join("risc0");
        let bazel_bin = get_bazel_bin();
        let src_path = bazel_bin.join("zirgen/circuit/bigint");
        let ec_path = risc0_root.join("bigint2/src/ec");
        let field_path = risc0_root.join("bigint2/src/field");
        let rsa_path = risc0_root.join("bigint2/src/rsa");

        self.copy_file(&src_path, &field_path, "extfield_deg2_add_256.blob");
        self.copy_file(&src_path, &field_path, "extfield_deg2_mul_256.blob");
        self.copy_file(&src_path, &field_path, "extfield_deg4_mul_256.blob");
        self.copy_file(&src_path, &field_path, "extfield_deg2_sub_256.blob");
        self.copy_file(&src_path, &field_path, "extfield_xxone_mul_256.blob");
        self.copy_file(&src_path, &field_path, "modadd_256.blob");
        self.copy_file(&src_path, &field_path, "modinv_256.blob");
        self.copy_file(&src_path, &field_path, "modmul_256.blob");
        self.copy_file(&src_path, &field_path, "modsub_256.blob");
        self.copy_file(&src_path, &rsa_path, "modpow65537_4096.blob");
        self.copy_file(&src_path, &ec_path, "ec_add_256.blob");
        self.copy_file(&src_path, &ec_path, "ec_double_256.blob");
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

    // Allocate `bootstrap` object with static lifetime so we can reference `self` in threadpool callbacks.
    let bootstrap: &'static mut _ = Box::leak(Box::new(Bootstrap::new(Args::parse())));
    bootstrap.run();
}
