def zirgen_genfiles(name, zir_file, zirgen_outs, data = [], zirgen = None, **kwargs):
    """Define multiple ZirGen generated files using the same tool and input.

    Args:
      name: The name of the generated filegroup rule for use in dependencies.
      zir_file: The primary zirgen language file.
      tbl_outs: A list of tuples ([opts], out), where each 'opts' is a list of
        options passed to zirgen, each option being a string, and 'out' is the
        corresponding output file produced.
      **kwargs: Extra keyword arguments to pass to all generated rules.

    Since these files are potentially designed for use outside of bazel,
    we require the generated files to be present in the source repository.

    This macro generates a rule ("Generate" + name) usable with "bazel run"
    to generate the files and copy them into a repository.

    It also generates a rule ("TestFresh" + name) usable with "bazel test"
    to make sure that the files in the repository are up to date with
    the ones generated through bazel.

    """
    zir_files = []
    if zir_file:
        zir_files = [zir_file]
    for (opts, out) in zirgen_outs:
        rule_name = "%s_%s_genrule" % (name, str(hash(" ".join(opts))))
        _zirgen_rule(
            name = rule_name,
            zirgen = zirgen or Label("//zirgen/dsl:zirgen"),
            zir_files = zir_files,
            data = data,
            opts = opts,
            out = "zirgen-generated/" + out,
        )
    generated = ["zirgen-generated/" + out for (opts, out) in zirgen_outs]
    repo_files = [out for (opts, out) in zirgen_outs]
    native.filegroup(name = name, srcs = repo_files, **kwargs)
    zirgen_generate(name = "Generate" + name, generated = generated)
    zirgen_fresh_test(name = "TestFresh" + name, generated = generated, repo_targets = [name])

def zirgen_build(name, out, zir_file, opts, data = [], **kwarg):
    _zirgen_rule(
        name = name,
        out = out,
        zirgen = Label("//zirgen/dsl:zirgen"),
        zir_files = [zir_file],
        data = data,
        opts = opts,
    )

def dirname(f):
    return f.dirname

def _zirgen_rule_impl(ctx):
    args = ctx.actions.args()
    args.add_all(ctx.attr.opts)
    args.add_all(ctx.files.zir_files)

    # Add includes for each directory containing data files
    args.add_all(ctx.files.data, before_each = "-I", map_each = dirname, uniquify = True)

    command = ctx.executable.zirgen.path + " \"$@\" >" + ctx.outputs.out.path
    inputs = ctx.files.zir_files + [ctx.executable.zirgen] + ctx.files.data
    if ctx.outputs.out.path.endswith(".cpp.inc"):
        # Attempt to run clang-format if available.  This makes things
        # nice if a human needs to read the generated file, but is
        # otherwise unnecessary so errors are ignored.
        command += " && ( clang-format -i " + ctx.outputs.out.path + " || true)"

    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        inputs = inputs,
        command = command,
        arguments = [args],
        # Make sure action_env settings are honored so the env is the same as
        # when the tool was built. Important for locating shared libraries with
        # a custom LD_LIBRARY_PATH.
        use_default_shell_env = True,
        mnemonic = "ZirGenerate",
    )

    return [DefaultInfo()]

_zirgen_rule = rule(
    _zirgen_rule_impl,
    doc = "Invokes the zirgen circuit compiler.",
    # Match genrule behavior
    output_to_genfiles = True,
    attrs = {
        "zirgen": attr.label(
            doc = "The ZirGen executable with which to generate `out`.",
            executable = True,
            cfg = "exec",
        ),
        "zir_files": attr.label_list(
            doc = "The zirgen source file(s) to run through `zirgen`.",
            allow_files = True,
        ),
        "data": attr.label_list(
            doc = "Additional source files needed by zirgen",
            allow_files = True,
        ),
        "out": attr.output(
            doc = "The output file for the zirgen invocation.",
            mandatory = True,
        ),
        "opts": attr.string_list(
            doc = "Additional command line options to add to the zirgen" +
                  " invocation.",
        ),
    },
)

def zirgen_fresh_test(name, generated, repo_targets):
    generated_paths = ["$(rootpath " + out + ")" for out in generated]
    repo_paths = ["$(rootpath " + out + ")" for out in repo_targets]
    native.py_test(
        name = name,
        srcs = [Label("//bazel/rules/zirgen:test_fresh.py")],
        data = generated + repo_targets,
        args = generated_paths,
        main = "test_fresh.py",
        local = True,
    )

def zirgen_generate(name, generated):
    generated_paths = ["$(rootpath " + out + ")" for out in generated]
    native.py_binary(
        name = name,
        srcs = [Label("//bazel/rules/zirgen:test_fresh.py")],
        data = generated,
        args = ["--install"] + generated_paths,
        main = "test_fresh.py",
    )
