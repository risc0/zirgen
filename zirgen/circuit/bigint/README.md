These are the bigint programs we use for our algebraic precompiles, and the compiler code to generate them.

In our system, BigInt programs are initially written as MLIR programs in the [bigint Dialect](../../Dialect/BigInt/Overview.md).
Look in e.g. `rsa.cpp/.h` or `elliptic_curve.cpp/.h` for these programs in their original forms.

Then, they are compiled into a BIBC (BigInt Byte Code) BLOB format by the `bigint2c.cpp` compiler.
This is the format the RISC Zero zkVM can execute as a precompile or part of a precompile.

This directory also contains `bibc-exec.cpp`, which is code that can directly execute BIBC code, e.g. to allow for testing without use of the zkVM.

You may also be interested in the [bootstrap code](../../bootstrap/), which will repeatedly invoke bigint2c compiler to generate all the BigInt programs expected in the [`risc0` repository](https://github.com/risc0/risc0).
