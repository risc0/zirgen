# Conceptual Overview

Now that we've seen and run "hello world," we can start to really dig into the
substance of Zirgen. While the language certainly draws on ideas from
imperative, functional, and logic programming, the challenge of building
arithmetic circuits is sufficiently strange that a specialized model of
abstraction would be helpful. The RISC Zero proof system is STARK-based, and so
circuits for the proof system fundamentally need to reason about a grid of
finite field elements (the witness) with polynomial constraints between them,
and procedures for populating them based on program inputs and other values in
the witness (witness generation).

While in principle witness generation for a STARK can populate the values in any
order, Zirgen imposes the restriction that all columns for one row are populated
before moving on to the next one. This way, one can read a Zirgen program as a
procedure for generating the next row of the execution trace. This creates a
time metaphor for the execution trace, and we refer to rows of the STARK witness
as "cycles" in analogy to clock cycles of a microprocessor. The "duration" of a
computation (number of cycles) is configurable at run time, and is determined by
the prover before proving starts.

A particular circuit, then, is responsible for defining a few things:
* How many columns are there in the witness?
* What constraints exist between the columns?
* How is the witness populated?

## Components

The core unit of abstraction in Zirgen is a component. A component,
fundamentally, is an answer to these three questions. The whole circuit is a
special component called `Top` (compare the `main` function from Rust) which
must be self-contained, but the smaller components that make it up don't have to
be. This means that a component can add constraints to columns that it did not
allocate, but were instead allocated by another component. Components can also
store intermediate computed values that aren't part of the witness.

## Val

The `Val`, component type represents a finite field element. Notably, a `Val`
does not need to be written into the witness, but works more as an "intermediate
value" for all the computations done by the circuit. As such, a `Val` does not
allocate any columns in the witness, generates no constraints, and has no impact
on the witness. It is purely a value that exists "on the side" of the circuit
execution.

`Val`s can be represented syntactically with numeric literals like `42`
(decimal), `0xFF` (hexadecimal), or `0b10101` (binary) notation, and since they
are finite field elements they can be added, subtracted, multiplied, and divided
according to the rules of modular arithmetic. These operations are themselves
components that can be coerced to `Val`, and have "identifier" names (`Add`,
`Sub`, `Mul`, `Div`) as well as the more familiar infix operators (`+`, `-`,
`*`, `/`).

## Registers

Now, to actually put values into the witness, we introduce the notion of a
register: a register is a component that allocates and populates a single column
in the witness.

The most fundamental kind of register in Zirgen is the `NondetReg` component,
which allocates a single column, writes it with a given value, and applies no
constraints. The "nondet" is short for "nondeterministic", which in this context
means that the value is unconstrained so that the prover can in principle assign
it any value.

Another very common kind of register is the `Reg` component, which is also built
into the language. It is defined as a `NondetReg` that additionally constrains
the value of that column to equal the value passed to the constructor:

```
component Reg(v: Val) {
   reg := NondetReg(v);
   v = reg;
   reg
}
```

See Section 2.5 of the [Cairo paper][paper-cairo] for a more thorough explanation of
non-determinism.

# Constraints

Constraints describe conditions between column values that must be satisfied if
the prover creates a valid proof. As such, they must be polynomial equations
over constants and values recorded in the witness. The constraint equations must
be polynomials, so they must be expressed without division, and must not include
terms with degree greater than 5 due to the current parameters of the RISC Zero proof
system. For example, one may constrain a value to be either 0 or 1 using this
component:

```
component IsBit(v: Val) {
  v * (v - 1) = 0;
}

test zero_is_bit {
  r := NondetReg(0);
  IsBit(r);
}

test one_is_bit {
  r := NondetReg(1);
  IsBit(r);
}

test_fails two_is_not_bit {
  r := NondetReg(2);
  IsBit(r);
}
```

## Supercomponents

It is often useful to be able to use multiple related types in the same way. For
example, multiplication is defined for any two `Val`s, but when writing circuits
one often finds themselves wanting to multiply the value in a register. Now, if
one has a `r: Reg`, one could write `3 * r.reg`, but this is needlessly verbose.
Instead, it would be better to allow registers to be used as if they are `Val`s,
treating it like the value read from the witness, so one can instead write
`3 * r`. Zirgen implements this subtype relation through the idea of
supercomponents --- every component has a super component (with one exception),
any component can be implicitly coerced to the type of its supercomponent, or
the supercomponent of its supercomponent, recursively up to the trivial
component called `Component`, which is the only component without a
supercomponent. So one can define the "supercomponent chain" of a component as
the sequence of types that a particular component can be coerced to. For
example, the supercomponent chain of `Reg` is
`Reg -> NondetReg -> Val -> Component`.

[Prev](01_Getting_Started.md)
[Next](03_Building_a_Fibonacci_Circuit.md)

[paper-cairo]: https://eprint.iacr.org/2021/1063.pdf
