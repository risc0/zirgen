# Backs

A circuit in Zirgen describes one row of the execution trace at a time. The _back_
operator `@` lets you read column values from previous rows (previous cycles). This
is the primary mechanism for building stateful computations — accumulators, counters,
rolling hashes, and similar patterns all depend on it.

## Syntax

A back expression is written as a postfix `@` followed by a distance:

```
<expression>@<distance>
```

The distance must be a compile-time constant and indicates how many cycles back to
read. A distance of `1` reads from the immediately preceding cycle; a distance of `2`
reads from two cycles back, and so on.

```
reg@1    // value of reg on the previous cycle
reg@2    // value of reg two cycles ago
```

The `@` operator binds more tightly than any binary operator, so `x@1 + y` parses as
`(x@1) + y`, not `x@(1 + y)`.

## A simple counter

The prototypical use of backs is a counter that adds one to itself each cycle:

```
extern GetCycle() : Val;

component Count(first: Val) {
  public a : Reg;
  a := Reg((1 + a@1) * (1 - first));
}

test count {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Log("a = %u", c.a);
  // Cycle 0: a = 0
  // Cycle 1: a = 1
  // Cycle 2: a = 2
}
```

On the first cycle `first` is 1, so the product `(1 + a@1) * (1 - first)` evaluates to
zero regardless of `a@1`. On subsequent cycles `first` is 0, so the expression becomes
`1 + a@1` — the previous value plus one.

This pattern of gating on the first cycle is very common when using backs, because there
is no well-defined "previous" row on cycle 0. Reading too far back (asking for a back
that goes before cycle 0) wraps around to the end of the witness, which is intentional:
the circuit is laid out as a ring so that wrap-around constraints can enforce things like
a counter resetting at the correct time.

## Taking backs on components

The `@` operator can be applied to any expression, not just registers. When applied to a
component, it gives you the entire component from the previous cycle, and you can then
access its members with `.`:

```
component PrevCount(first: Val) {
  public c := Count(first);
  public prev := c@1;       // the Count component from the previous cycle
}

test prev_count {
  first := NondetReg(Isz(GetCycle()));
  c := PrevCount(first);
  Log("prev.a = %u", c@1.prev.a);  // Count value from two cycles ago
}
```

Member access on a back and taking a back of a member are equivalent, so `c@1.a` and
`(c.a)@1` refer to the same column value.

## Forward declarations and backs

To take a back on a register defined later in the same component, it must be forward
declared before use:

```
d2 : Reg;
d3 : Reg;
d1 := Reg([first, 1-first] -> (f0, d2@1));
d2 := Reg([first, 1-first] -> (f1, d3@1));
d3 := Reg(d1 + d2);
```

Here `d2` and `d3` are forward declared so that `d2@1` and `d3@1` can appear in the
expression defining `d1` and `d2` respectively. The forward declaration `: Reg`
allocates the column in the witness without yet populating it; the `:=` definition that
follows fills in the value.

This is the pattern used in the Fibonacci circuit from the
[Building a Fibonacci Circuit](03_Building_a_Fibonacci_Circuit.md) tutorial.

## Taking backs on arrays

The `@` operator also applies to arrays. `arr@1` refers to the entire array from the
previous cycle, and `arr@1[i]` reads a specific element:

```
arr := for i : 0..4 {
  Reg([first, 1-first] -> (
    i + 1,
    (i + 1) * arr@1[i]      // multiply by the previous-cycle element
  ))
};
```

You can also reduce over a previous-cycle array:

```
result := [first, 1 - first] -> (
  0,
  reduce base@1 init 0 with Add   // sum all elements from the previous cycle
);
```

See [Arrays and Loops](99_Arrays_and_Loops.md) for more on arrays.

## Backs and muxes

When taking a back on a mux expression, only registers that are part of the mux's
least-common-super layout are guaranteed to be well-defined. Registers that belong
exclusively to one arm of the mux may share columns with registers from another arm;
taking a back on those produces an undefined value. See [Muxes](05_Muxes.md) for
details on mux layout.

[Prev](05_Muxes.md)
