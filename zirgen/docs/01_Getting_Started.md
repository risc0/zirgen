# Getting Started

We don't currently release Zirgen in any form, so it's only available through
this repository. Assuming you've cloned and built things from this repo before,
building Zirgen with Bazel is simple with the following command. Note, though,
that this isn't strictly necessary, and that Bazel will automatically (re)build
Zirgen if you use it to invoke the tests as well.

```
bazel build //zirgen/dsl:zirgen
```

## Hello world!

Following in the footsteps of our forebears, let's take a look at a classic
"Hello world" program. This program is already available as a test and example,
so we can run it with the following command from the root of this repository:

```
$ bazel run //zirgen/dsl:zirgen -- $(pwd)/zirgen/dsl/test/hello_world.zir --test
...
Running 0
[0] Log: Hello world!
Lookups resolved
```

This command passes two arguments to the Zirgen executable. The first,
`$(pwd)/zirgen/dsl/test/hello_world.zir`, specifies the path of the Zirgen file
we want to run on. The second, `--test`, specifies that we want to run all the
tests defined in the file we're running. Typically, the output of Zirgen is a
generated Rust or C++ library that then needs to be integrated with the RISC
Zero proof system. For the sake of simplicity here and as a useful practice
during the development, it is easiest to experiment with Zirgen by writing tests
alongside your circuit code, which can be run in the builtin interpreter without
doing this integration work using the `--test` option. Now, the important part
of the file is the following:

```
test {
  Log("Hello world!");
}
```

The keyword `test` declares that the thing that follows (enclosed in curly
braces) is a test. Tests can be given an optional name, but if they aren't named
then they are labeled with sequential numbers. This causes the text "Running 0"
to be written to stdout, marking the beginning of the execution of that test.
The statement `Log("Hello world!");` is what causes the text "[0] Log: Hello
world!" to be written to stdout.

[Prev](README.md)
[Next](02_Conceptual_Overview.md)
