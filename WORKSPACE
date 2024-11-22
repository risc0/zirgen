workspace(name = "zirgen")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("//bazel/rules/zirgen:deps.bzl", "zirgen_dependencies")
zirgen_dependencies()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

LLVM_COMMIT = "fea7b65f23632b42ff8f7e2595ac0641e2c1d214"

LLVM_SHA256 = "7f8da8de897f20824e7d11204768ccb29a47419385bb6c9b3f5eccfd738d7510"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project")

load("//bazel/toolchain/risc0:repo.bzl", "risc0_toolchain")

risc0_toolchain(name = "risc0_toolchain")

load("//bazel/toolchain/rv32im-linux:repo.bzl", "rv32im_linux_toolchain")

rv32im_linux_toolchain(name = "rv32im_linux_toolchain")

http_archive(
    name = "riscv_tests",
    build_file = "//bazel/third_party:riscv_tests.BUILD",
    sha256 = "831a91955287865f9c09bbb7102f7ba2987d86823fd7b67c8a62eff46e21a83c",
    strip_prefix = "riscv-tests-a6ab6ae6008ffc2ea907ea9f6d2b8379583e7d56",
    url = "https://github.com/riscv/riscv-tests/archive/a6ab6ae6008ffc2ea907ea9f6d2b8379583e7d56.zip",
)

http_archive(
    name = "rules_conda",
    patch_args = ["-p1"],
    patches = ["//bazel/third_party/rules_conda:prefix.patch"],
    sha256 = "9793f86162ec5cfb32a1f1f13f5bf776e2c06b243c4f1ee314b9ec870144220d",
    url = "https://github.com/spietras/rules_conda/releases/download/0.1.0/rules_conda-0.1.0.zip",
)

load("@rules_conda//:defs.bzl", "conda_create", "load_conda", "register_toolchain")

load_conda(
    install_mamba = True,
    installer = "miniforge",
    quiet = False,
)

conda_create(
    name = "py3_env",
    environment = "@//:environment.yml",
    quiet = False,
    use_mamba = True,
)

register_toolchain(py3_env = "py3_env")

register_toolchains(
    "//bazel/rules/clang_format:toolchain",
)

http_archive(
    name = "com_google_googletest",
    sha256 = "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363",
    strip_prefix = "googletest-1.13.0",
    url = "https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz",
)

http_archive(
    name = "rules_pkg",
    sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
# tip: use `bazel run @hedron_compile_commands//:refresh_all`
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()

HERMETIC_CC_TOOLCHAIN_VERSION = "v3.1.0"

http_archive(
    name = "hermetic_cc_toolchain",
    sha256 = "df091afc25d73b0948ed371d3d61beef29447f690508e02bc24e7001ccc12d38",
    urls = [
        "https://mirror.bazel.build/github.com/uber/hermetic_cc_toolchain/releases/download/{0}/hermetic_cc_toolchain-{0}.tar.gz".format(HERMETIC_CC_TOOLCHAIN_VERSION),
        "https://github.com/uber/hermetic_cc_toolchain/releases/download/{0}/hermetic_cc_toolchain-{0}.tar.gz".format(HERMETIC_CC_TOOLCHAIN_VERSION),
    ],
)

load("@hermetic_cc_toolchain//toolchain:defs.bzl", zig_toolchains = "toolchains")

# Plain zig_toolchains() will pick reasonable defaults. See
# toolchain/defs.bzl:toolchains on how to change the Zig SDK version and
# download URL.
zig_toolchains()
