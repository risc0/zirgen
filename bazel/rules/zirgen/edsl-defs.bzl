DEFAULT_OUTS = [
    "eval_check.cu",
    "eval_check.cuh",
    "eval_check.metal",
    "eval_check.h",
    "impl.h",
    "info.rs",
    "poly_edsl.cpp",
    "poly_ext.rs",
    "rust_poly_fp.cpp",
    "rust_step_exec.cpp",
    "rust_step_verify_mem.cpp",
    "rust_step_verify_bytes.cpp",
    "rust_step_compute_accum.cpp",
    "rust_step_verify_accum.cpp",
    "step_exec.cu",
    "step_verify_mem.cu",
    "step_verify_bytes.cu",
    "step_compute_accum.cu",
    "step_verify_accum.cu",
    # "step_exec.metal",
    # "step_verify_mem.metal",
    # "step_verify_bytes.metal",
    "step_compute_accum.metal",
    "step_verify_accum.metal",
    "taps.cpp",
    "taps.rs",
    "layout.rs.inc",
    "layout.cpp.inc",
    "layout.cu.inc",
]

ZIRGEN_OUTS = [
    "defs.cpp.inc",
    "defs.cu.inc",
    "defs.rs.inc",
    "info.rs",
    "layout.cpp.inc",
    "layout.cu.inc",
    "layout.rs.inc",
    "poly_ext.rs",
    "taps.rs",
    "types.h.inc",
    "types.cuh.inc",
    "types.rs.inc",
    "validity.ir",
]

def _impl(ctx):
    outs = []
    out_dirs = dict()
    for out in ctx.attr.outs:
        declared = ctx.actions.declare_file(out.name)
        outs.append(declared)
        dirname = declared.dirname
        out_dirs[dirname] = dirname

    if len(out_dirs) != 1:
        fail("Must have exactly one output directory")

    ctx.actions.run(
        mnemonic = "CodegenCircuits",
        executable = ctx.executable.binary,
        arguments = [ctx.expand_location(arg, targets = ctx.attr.data) for arg in ctx.attr.extra_args] + ["--output-dir", dirname],
        inputs = ctx.files.data,
        outputs = outs,
        tools = [ctx.executable.binary],
    )

    runfiles = ctx.runfiles(files = outs)
    return [DefaultInfo(files = depset(outs), runfiles = runfiles)]

_build_circuit_rule = rule(
    implementation = _impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            cfg = "exec",
            doc = "The cpp program to run to generate results",
        ),
        "data": attr.label_list(
            allow_files = True,
        ),
        "outs": attr.output_list(mandatory = True),
        "extra_args": attr.string_list(),
    },
)

def build_circuit(name, srcs = [], bin = None, deps = [], outs = None, data = [], extra_args = []):
    if outs == None:
        outs = DEFAULT_OUTS

    if not bin:
        bin = name + "_gen"
        native.cc_binary(
            name = bin,
            srcs = srcs,
            deps = deps + [
                "@zirgen//zirgen/compiler/edsl",
                "@zirgen//zirgen/compiler/codegen",
            ],
        )

    _build_circuit_rule(
        name = name,
        binary = bin,
        data = ["@zirgen//zirgen/compiler/codegen:data"] + data,
        outs = outs,
        extra_args = extra_args,
    )
