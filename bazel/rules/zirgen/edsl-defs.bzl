DEFAULT_OUTS = [
    "eval_check.cu",
    "eval_check.metal",
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

def _impl(ctx):
    outs = []
    for out in ctx.attr.outs:
        outs.append(ctx.actions.declare_file(out.name))

    ctx.actions.run(
        mnemonic = "CodegenCircuits",
        executable = ctx.executable.binary,
        arguments = [x.path for x in outs],
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
    },
)

def build_circuit(name, srcs, deps = [], outs = None):
    bin = name + "_gen"

    if outs == None:
        outs = DEFAULT_OUTS

    native.cc_binary(
        name = bin,
        srcs = srcs,
        deps = deps + [
            "//zirgen/compiler/edsl",
            "//zirgen/compiler/codegen",
        ],
    )

    _build_circuit_rule(
        name = name,
        binary = bin,
        data = ["//zirgen/compiler/codegen:data"],
        outs = outs,
    )
