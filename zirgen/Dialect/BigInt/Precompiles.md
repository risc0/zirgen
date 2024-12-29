# Creating a precompile using the BigInt2 system

Creating a new precompile using BigInt2 requires a series of related changes across files in two repositories. This is the process you should follow if you want to create a new bigint2 precompile and make it available within the zkVM.

To keep the demo simple, let's imagine that we want to create a trivial precompile which sums two 128-bit numbers, which we'll call "add128".


## Write a bigint program
First, create files under [`zirgen/circuit/bigint/`] which will hold the accelerator builder function. We might call them `add128.h` and `add128.cpp`, declaring and then defining a function like this:

In add128.cpp:
```
#include "zirgen/circuit/bigint/add128.h"

namespace zirgen::BigInt {
void genAdd128(mlir::OpBuilder& builder, mlir::Location loc) {
    auto lhs = builder.create<BigInt::LoadOp>(loc, 128, 11, 0);
    auto rhs = builder.create<BigInt::LoadOp>(loc, 128, 12, 0);
    auto sum = builder.create<BigInt::AddOp>(loc, lhs, rhs);  
    builder.create<BigInt::StoreOp>(loc, sum, 13, 0);
}
}
```

In add128.h
```
#include "zirgen/Dialect/BigInt/IR/BigInt.h"

namespace zirgen::BigInt {
void genAdd128(mlir::OpBuilder& builder, mlir::Location loc);
}
```

We might also like to create a test function here, but we'll gloss over that for now.

## Compile your bigint program to a bigint blob using bigint2c
To make use of the new function, we must add it as an option of the `bigint2c` compiler program, which generates the precompiled blobs we'll need to load into the zkVM. Ensure you include the new circuit header file:

```
#include "zirgen/circuit/bigint/add128.h"
```

Inside `zirgen/circuit/bigint/bigint2c.cpp`, we'll add a new command line option:

```
enum class Program {
  ModPow65537,
  EC_Double,
  EC_Add,
  ModAdd,
  ModInv,
  ModMul,
  ModSub,
  Add128
};
```

Make this option available by extending the arg parser's declaration:
```
   
               clEnumValN(Program::Add128, "add128", "Add128")),
```

Later, in the main function, we must handle the new option by invoking our new generator function:

```
case Program::Add128:
  zirgen::BigInt::genAdd128(builder, loc);
  break;
```

## Bootstrapping your precompile into the `risc0` repo

The `bigint2c` program can now compile our addition function into a "bibc" format blob. In order to use this blob from the `risc0` repo, we must add it to the bootstrap process, which means we must integrate blob-generation into the zirgen circuit build process. This happens in `zirgen/circuit/bigint/BUILD.bazel`, where we must create a new genrule for the new blob file:

```
genrule(
    name = "add128",
    outs = ["add128.blob"],
    exec_tools = [":bigint2c"],
    cmd = "$(location //zirgen/circuit/bigint:bigint2c) --program=add128 --bitwidth 128 > $(OUTS)"
)
```

Next we must add the new target name to the list of blob files to generate:

```
BLOBS = [
    "modpow65537_4096",
    "ec_double_256",
    "ec_add_256",
    "modadd_256",
    "modinv_256",
    "modmul_256",
    "modsub_256",
    "add128",
]
```

Add "add128.cpp" to `cc_library.srcs`. Add "add128.h" to `cc_library.hdrs`.

```
    srcs = [
        "elliptic_curve.cpp",
        "field.cpp",
        "rsa.cpp",
        "add128.cpp",
    ],
```

```
    hdrs = [
        "elliptic_curve.h",
        "field.h",
        "rsa.h",
        "add128.h",
        "//zirgen/circuit/recursion",
        "//zirgen/circuit/rv32im/v1/edsl:rv32im",
    ],
```

We can now `bazelisk build //zirgen/circuit/bigint/...` to produce our new blob along with the others. In order to make this available to the `risc0` world, we must add the new blob to the bootstrap process. In `zirgen/bootstrap/src/main.rs`, in the `bigint2` function, after a series of similar lines, we must add these lines to copy the new blob over:


```
        let add128_path = risc0_root.join("bigint2/src/add128");
        self.copy_file(&src_path, &add128_path, "add128.blob");
```

We may now bootstrap the risc0 repository, including the new blob file, like this (you may need to adjust the path):


```
cargo bootstrap bigint2 --output=`$HOME`/risc0/
```

## Invoking a bigint program
To invoke the new addition program, we move over to the `risc0` repository. Here we will create a new module, akin to `bigint2/src/ec` or `bigint2/src/rsa`, containing a new `bigint2/src/add128/mod.rs` file. Its contents should look like this:

```
use include_bytes_aligned::include_bytes_aligned;
use crate::ffi::sys_bigint2_3;

const BLOB: &[u8] = include_bytes_aligned!(4, "add128.blob");

pub fn add128(lhs: &[u32; 4], rhs: &[u32; 4], result: &mut [u32; 8]) {
    unsafe {
        sys_bigint2_3(
            BLOB.as_ptr(),
            lhs.as_ptr() as *const u32,
            rhs.as_ptr() as *const u32,
            result.as_mut_ptr() as *mut u32,
        );
    }
}
```

Notice that the size of the `result` buffer is larger than the input buffers -- this is because the sum of two 128-bit unsigned integers can be as large as 256 bits. It is important that each input and output buffer is sufficiently large, as otherwise the bigint program can read and/or write out of bounds (the `unsafe` keyword is warning us of this possibility of out-of-bounds access). You can look at the types used in the Load/Store ops to help determine appropriate buffer sizes.

One can now import the `add128` function to perform simple addition through the bigint2 accelerator.

## Utilizing the new precompile

The `add128` function is now available as part of the `risc0-bigint2` package. All that's left now is to add a dependency on this package and call your precompile!

The bigint system passes input and output data as arrays of `u32`s with the least significant digits first. For example, if we wanted to compute `4294967296 + 4` with our `add128` function, we would call `add128([0, 1, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0])` and after the call the result buffer would read `[4, 1, 0, 0, 0, 0, 0, 0]`. How you process your broader program's data into this format will be application-specific, but we have helper functions for some cases; see for instance the `num-bigint` and `num-bigint-dig` features of the `risc0-bigint2` crate.

For a relatively simple real-world example of utilizing a precompile, look at how we use the `modpow_65537` precompile to [patch RustCrypto's RSA crate][rustcrypto-rsa-patch].

[`zirgen/circuit/bigint/`]: https://github.com/risc0/zirgen/tree/main/zirgen/circuit/bigint
[rustcrypto-rsa-patch]: https://github.com/risc0/RustCrypto-RSA/pull/5/files
