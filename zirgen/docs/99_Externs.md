# Externs

An _extern_ is a component whose implementation is provided by the prover at runtime
rather than defined in Zirgen source. Externs are the primary way that a circuit
interacts with the world outside the witness: reading inputs, writing outputs, and
accessing prover-computed "hints" that would be expensive or impossible to compute inside
the circuit itself.

## Declaring an extern

An extern is declared with the `extern` keyword, a name, a parameter list, and an
optional return type:

```
extern <Name>(<param1>: <Type1>, ...) ;
extern <Name>(<param1>: <Type1>, ...) : <ReturnType> ;
```

If the return type is omitted the extern has no useful return value (its super is the
trivial `Component` type). If a return type is given, the extern can be used anywhere an
expression of that type is expected.

Some examples:

```
// An extern that accepts a value but returns nothing (e.g. to write output)
extern OutputToUser(val: Val);

// An extern that returns a Val (e.g. to read input)
extern GetValFromUser() : Val;

// An extern that returns the current cycle number
extern GetCycle() : Val;

// An extern that accepts and returns structured component types
component PairVal(aArg: Val, bArg: Val) {
  a := aArg;
  b := bArg;
}
extern TakesPair(v: PairVal);
extern ReturnsPair() : PairVal;
```

## Calling an extern

Externs are called exactly like ordinary component constructors:

```
component Top() {
  x := GetValFromUser();         // call extern, bind result
  OutputToUser(x * 2);          // call extern, discard result
  TakesPair(ReturnsPair());     // nest extern calls
}
```

## Trust and constraints

The prover controls the implementation of every extern, which means a malicious prover
could supply any value it likes. An unconstrained extern return value gives the prover
a "free variable" to exploit. For this reason, **extern return values should almost
always be stored in a `NondetReg` and then constrained**:

```
extern GetCycle() : Val;

component CycleCounter() {
  cycle := NondetReg(GetCycle());   // record the prover's claim

  // Now constrain it: cycle must advance by 1 each row (except the first)
  ...
  cycle
}
```

Compare this with `Reg`, which both records and constrains in one step. `Reg` is
appropriate when the value is computed as a polynomial of other registers; `NondetReg`
followed by explicit constraints is appropriate for extern-supplied values.

The `Log` extern is a special case: it has no return value and produces no constraints,
so it is safe to call without any additional work. It is intended only for debugging
and has no effect on the circuit's validity.

## Built-in externs

Several externs are provided by the Zirgen runtime and do not need to be declared in
your source file. They are described in [Builtin Components](A1_Builtin_Components.md);
the most commonly used ones are listed here for reference.

| Extern | Signature | Description |
|--------|-----------|-------------|
| `GetCycle` | `() : Val` | Returns the current cycle number (0-indexed). |
| `IsFirstCycle` | `() : Val` | Returns 1 on cycle 0, 0 on all other cycles. |
| `Isz` | `(v: Val) : Val` | Returns 1 if `v` is zero, 0 otherwise. Nondeterministic. |
| `Inv` | `(v: Val) : Val` | Returns the multiplicative inverse of `v`, or 0 if `v` is 0. Nondeterministic. |
| `InRange` | `(a: Val, b: Val, c: Val) : Val` | Returns 1 if `a <= b < c` (integer comparison), 0 otherwise. |
| `BitAnd` | `(a: Val, b: Val) : Val` | Returns the bitwise AND of `a` and `b`. Nondeterministic. |
| `Log` | `(fmt: ...) ` | Prints a debug message. No constraints generated. |

All of the nondeterministic externs (`Isz`, `Inv`, `InRange`, `BitAnd`) require you to
add constraints in order to make their outputs trustworthy. See
[Builtin Components](A1_Builtin_Components.md) for patterns showing how to do this, and
the `IsZero` component in [Building a Fibonacci Circuit](03_Building_a_Fibonacci_Circuit.md)
for a worked example.

## Externs in multi-cycle circuits

When a circuit runs for multiple cycles each extern call happens once per cycle. The
prover is free to return different values on each cycle, which is how cycle-varying
inputs (like `GetCycle`) work. Externs that read from an external stream — user inputs,
file data, hash preimages — typically advance their internal pointer on each call, so
the order in which they are called within a cycle matters.

[Prev](05_Muxes.md)
