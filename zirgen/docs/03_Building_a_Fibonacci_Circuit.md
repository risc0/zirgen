# Building a Fibonacci Circuit

Let's jump right in by walking through building a complete circuit from
beginning to end! We'll build something closely based on the Fibonacci circuit
from the [STARK by Hand](https://dev.risczero.com/proof-system/stark-by-hand)
document from our website, which will take us through a lot of basic language
features and integrating them with the RISC Zero proof system.

## Setting up the project

First off, we need to create a new Zirgen file for our circuit. Zirgen files
conventionally have the extension `.zir`, and the final source code for this
example lives in `/zirgen/dsl/examples/fibonacci.zir`. For now, we can create
just a trivial `Top` component in that file which doesn't really do anything to
satisfy the compiler:

```
component Top() {}
```

Next, we need to tell Bazel about our new circuit, and tell it how to build our
project. Create a `Build.bazel` file in the same directory as our new circuit
source file, with these contents:

```
package(
    default_visibility = ["//visibility:public"],
)

load("//bazel/rules/zirgen:dsl-defs.bzl", "zirgen_genfiles")

zirgen_genfiles(
    name = "FibonacciIncs",
    zir_file = ":fibonacci.zir",
    zirgen_outs = [
        (
            ["--emit=rust"],
            "fibonacci.rs.inc",
        ),
    ],
)
```

This adds a new Bazel build command called `GenerateFibonacciIncs` that
generates Rust library code related to the circuit. If you run it now, it will
create a file called `fibonacci.rs.inc` next to `fibonacci.zir`. For our circuit
code in the `examples` directory, we can generate this like so:

```
bazel run //zirgen/dsl/examples:GenerateFibonacciIncs
```

That library code doesn't really do anything now because the circuit doesn't do
anything yet, but at the end of the tutorial we'll talk about what's in the
generated code and how to integrate it with the proof system. We'll do the bulk
of our circuit development cycle without it though, using Zirgen tests and the
interpreter. The following command (replacing the path of `fibonacci.zir`` with
the correct path in your project) should quietly succeed:

```
bazel run //zirgen/dsl:zirgen -- $(pwd)/zirgen/dsl/examples/fibonacci.zir --test
```

## Building the circuit

### Inputs and outputs

Now that everything is set up, we can jump in and start building our circuit!
Let's start out by setting up our inputs and outputs. Our circuit will have 3
inputs: two initial terms for our Fibonacci sequence, `f0` and `f1`, and a
number of steps to compute, `steps`. It will also have a single output: the
`steps`-th element of the Fibonacci sequence after `f1`, `f_last`. None of
these inputs need to be secret, so we'll make all of them global constants
defined outside the circuit. They are global in the sense that any declaration
or definition prefixed with the `global` keyword will refer to the same
component, and they are constant in the sense that they have only a single value
over the entire execution of the circuit, so their value does not change from
cycle to cycle. In fact, they exist outside the witness, and are ultimately
revealed to the verifier, so they are public.

```
component Top() {
  global f0: Reg;
  global f1: Reg;
  global steps: Reg;
}
```

### Handling our base case

Now let's make the circuit do just one cycle of computation. On the first cycle,
we'll want to move the input values into columns of the witness and compute the
next term. Add the following code to the end of the `Top` component:

```
// Copy global inputs into the witness
d1 := Reg(f0);
d2 := Reg(f1);

// Compute the next Fibonacci term
d3 := Reg(d1 + d2);

// Write the next term as the output
global f_last := Reg(d3);
```

We can also write a test that supplies values to these globals and checks the
output:

```
test FirstCycle {
  // Supply inputs
  global f0 := Reg(1);
  global f1 := Reg(2);
  global steps := Reg(1);

  top := Top();

  // Check the output
  global f_last : Reg;
  f_last = 3;
}
```

Running this test with the above command, we should see that it passes! This
means `f_last` is being set to 3 as expected.

### Two cycles and beyond!

Now, in order to do multiple cycles, we're going to have to do a little more
work. On the first cycle, we needed to copy the initial terms into the witness.
But for all subsequent cycles, we want to compute subsequent Fibonacci terms
from the previous ones, so we need a mechanism to distinguish the first cycle
from other cycles. Conveniently, we can query the current cycle number using an
external component (extern) provided by the prover, which just needs to be
declared before it can be used:

```
extern GetCycle() : Val;
```

This says that `GetExtern` is a component whose constructor takes no arguments,
and has a supercomponent of type `Val`. The `extern` keyword indicates that the
implementation comes from the prover, which unfortunately means that a malicious
prover could use a different implementation to trick the verifier, so we'll have
to do more work to properly constrain the values that come from it. We'll
encapsulate this in a new component:

```
component CycleCounter() {
  global total_cycles := NondetReg(1);

  cycle := NondetReg(GetCycle());
  is_first_cycle := IsZero(cycle);

  [is_first_cycle, 1-is_first_cycle] -> ({
    // First cycle; previous cycle should be the last cycle.
    cycle@1 = total_cycles - 1;
  }, {
    // Not first cycle; cycle number should advance by one for every row.
    cycle = cycle@1 + 1;
  });
  cycle
}
```

There's a lot going on here, let's break it down. First, we introduce a new
global that stores the total number of cycles in the witness. It might seem like
this could be a place for the prover to break things by mismatching the total
number of cycles, but remember that the verifier knows all globals and will be
responsible for checking this later. Second, we query the prover for the cycle
number using our `GetCycle` extern, and record it into a register. All that's
left to do now is introduce constraints: we would like for the cycle number to
start at 0, increase by 1 every cycle. So if the cycle number is 0, we
constrain the previous cycle number (looping back to the end of the witness) to
be one less than the total number of cycles (since we start from 0), and for
all other cycles we constrain the current cycle number to be one more than the
previous. Figuring out if the cycle number is 0 is done using the `IsZero`
component, which we will define in a moment, and then we use a mux to introduce
one of our two constraints depending on whether or not the current cycle is 0.
Finally, in the last line, we declare `cycle` to be the supercomponent of our
`CycleCounter` component, so that it can be coerced to a `NondetReg` and `Val`.

Onward to `IsZero`! Zirgen defines a few builtin components that we'll make use
of here: `Isz`, which is 1 if its argument is 0, and 0 if its argument is
nonzero, and `Inv`, which computes the multiplicative inverse of its argument or
0 if its argument is 0. Neither of these create any constraints, so as with
`GetCycle` we're also on the hook to properly constrain these values. Without
further ado:

```
component IsZero(val: Val) {
  // Nondeterministically 'guess' the result
  isZero := NondetReg(Isz(val));

  // Compute the inverse (for non-zero values), for zero values, Inv returns 0
  inv := NondetReg(Inv(val));

  // isZero should be either 0 or 1
  isZero * (1 - isZero) = 0;
  // If isZero is 0 (i.e. nonzero) then val must have an inverse
  val * inv = 1 - isZero;
  // If isZero is 1, then val must be zero
  isZero * val = 0;
  // If isZero is 1, then inv must be zero
  isZero * inv = 0;
  isZero
}
```

It's also a good idea to add a test for the new component. Rerunning the tests,
we should see that both `IsZeroTest` and `FirstCycle` pass.

```
test IsZeroTest {
  IsZero(0) = 1;
  IsZero(1) = 0;
  IsZero(2) = 0;
}
```

Now we can go finish up our implementation of `Top`. First off, we need to
construct our cycle counter. Its constructor takes no arguments, so we simply
need to add this after our global declarations:

```
cycle := CycleCounter();
```

Next, we need to adjust our computation of the values for `d1` and `d2`
depending on whether we're on the first cycle or not. Conveniently, our cycle
counter already has this in a properly constrained register, so we can use it
to define a mux selector:

```
d2 : Reg;
d3 : Reg;
d1 := Reg([cycle.is_first_cycle, 1 - cycle.is_first_cycle] -> (f0, d2@1));
d2 := Reg([cycle.is_first_cycle, 1 - cycle.is_first_cycle] -> (f1, d3@1));
```

What is going on here? First off, we forward declare `d2` and `d3` so that we
can access them through "backs" --- this is how we refer to their values on
previous cycles. Then, we compute the values of `d1` and `d2` using a mux over
the `cycle.is_first_cycle` field. Recall that this register contains 1 if the
cycle is 0, and 1 otherwise. So on the first cycle, the selector array becomes `[1, 0]`, so our muxes evaluate as their first arms (`f0` and `f1` respectively),
and on all other cycles the selector array becomes `[0, 1]`, so they evaluate
as their second arms (`d2@1` and `d3@1` respectively).

This code is a bit verbose though, since `cycle.is_first_cycle` is repeated four
times. It might be a bit clearer to give this a shorter name. Note that even
though this expression has type `IsZero` and therefore has two registers
allocated for it, doing this simply "aliases" the component without introducing
any new columns to the witness. We can rewrite like so:

```
first := cycle.is_first_cycle;
d2 : Reg;
d3 : Reg;
d1 := Reg([first, 1-first] -> (f0, d2@1));
d2 := Reg([first, 1-first] -> (f1, d3@1));
```

Onto the last order of business: for the final step of our computation, we need
to copy the value of `d3` into our global output register `f_last`. Note that we
can only write to the global register once, because globals are not "per cycle,"
so we have to guard this assignment with another mux. As an added debugging
feature, we can log the assignment using the builtin `Log` extern. When we run
our tests, this shows us the value that gets written to that global, and on
which cycle of execution.

```
// If cycle = steps, write the next term to the output
terminate := IsZero(cycle - steps);
[terminate, 1 - terminate] -> ({
  global f_last := Reg(d3);
  Log("f_last = %u", f_last);
}, {});
```

So here's what the final implementation of `Top` should look like:

```
component Top() {
  global f0: Reg;
  global f1: Reg;
  global steps: Reg;

  cycle := CycleCounter();
  first := cycle.is_first_cycle;

  // Copy previous two terms forward
  d2 : Reg;
  d3 : Reg;
  d1 := Reg([first, 1-first] -> (f0, d2@1));
  d2 := Reg([first, 1-first] -> (f1, d3@1));

  // Compute the next Fibonacci term
  d3 := Reg(d1 + d2);

  // If cycle = steps - 1, write the next term to the output
  terminate := IsZero(cycle - steps + 1);
  [terminate, 1 - terminate] -> ({
    global f_last := Reg(d3);
  }, {});
}
```

### Adding some multi-cycle tests

TODO: the cycle counter doesn't work yet because the interpreter fails to check
the `cycle@1 = total_cycles - 1` constraint on the first cycle since the last
cycle hasn't been populated yet! I've commented out this constraint in the meantime, which compromises the integrity of the cycle counter.

Now we can write some tests that execute across multiple cycles. In order to do
this, we only need to change the `steps` value, and constrain the value written
to `f_last` matches the expected value. Note that constraints are checked
per-cycle, so we need to guard this constraint to only be checked on or after
the terminate cycle. There's a lot of shared structure here, so we can
abstract out test cases into a new component!

```
component FibTest(f0: Val, f1: Val, steps: Val, out: Val) {
  // Supply inputs
  global f0 := Reg(f0);
  global f1 := Reg(f1);
  global steps := Reg(steps);

  top := Top();

  // Check the output
  [top.terminate, 1 - top.terminate] -> ({
    global f_last : Reg;
    f_last = out;
  }, {});
}

test FirstCycle {
  FibTest(1, 2, 1, 3);
}

test SecondCycle {
  FibTest(1, 2, 2, 5);
}

test SixthCycle {
  FibTest(1, 2, 6, 34);
}
```

By default, invoking Zirgen on our
