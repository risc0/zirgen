# Backs

Zirgen circuits execute over multiple _cycles_ (rows of the execution trace).
The back-reference operator `@N` reads a component's value from `N` cycles ago.

## Syntax

```
a@1    // value of 'a' from 1 cycle ago
a@2    // value of 'a' from 2 cycles ago
```

The operand can be any component expression, and `N` must be a compile-time
constant.

## Semantics

On cycle 0 (and whenever the back distance exceeds the number of cycles that
have executed), a back-reference returns zero. This is the expected base case
for inductive definitions.

Back-references are most commonly used to define a register whose current value
depends on its previous value, implementing counters, accumulators, and other
sequential state.

## Simple counter example

From `zirgen/dsl/test/back_counter.zir`:

```
extern GetCycle() : Val;
extern Output(v: Val);

component Count(first: Val) {
  public a : Reg;
  // On cycle 0 ('first' is 1) the register is forced to 0.
  // On subsequent cycles it increments by one.
  a := Reg((1 + a@1) * (1 - first));
}

test count {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Output(c.a);
  // [0] Output(0)
  // [1] Output(1)
  // [2] Output(2)
  // [3] Output(3)
}
```

The expression `(1 + a@1) * (1 - first)` evaluates to `0` when `first == 1`
(cycle 0), and to `a@1 + 1` on every subsequent cycle.

## Accessing backs of a component

You can take a back of a composite component and then access its fields:

```
component PrevCount(first: Val) {
  public c := Count(first);
  public prev := c@1;   // the entire Count component from the previous cycle
}
```

`c@1.a` gives the value of the `a` register **one** cycle back — `@1` applies
to `c`, so you read `a` from the `Count` instance from one cycle ago.

To reach **two** cycles back, chain through `prev`. From `back_counter.zir`
lines 41–43:

```
test prev_count {
  first := NondetReg(Isz(GetCycle()));
  c := PrevCount(first);
  Output(c@1.prev.a);
  // [0] Output(0) -> ()
  // [1] Output(0) -> ()
  // [2] Output(0) -> ()
  // [3] Output(1) -> ()
}
```

Here `c@1` is the previous cycle's `PrevCount`, and `.prev` is that
`PrevCount`'s `prev` field — itself `c@1` inside that component, i.e. the
`Count` from one cycle before that. So `.prev.a` reaches `a` two cycles back.

## Backs of arrays

The `@N` operator applies uniformly to `Array` types. `arr@1[i]` reads element
`i` of the array from the previous cycle.

From `zirgen/dsl/test/back_of_array.zir`:

```
extern GetCycle() : Val;

test {
  cycle := NondetReg(GetCycle());
  first := NondetReg(Isz(cycle));

  base := [Reg(cycle + 0), Reg(cycle + 1), Reg(cycle + 2), Reg(cycle + 3)];
  result := [first, 1 - first] -> (
    0,
    reduce base@1 init 0 with Add
  );

  Log("result = %u", result);
  // Cycle 0: result = 0
  // Cycle 1: result = 6   (0+1+2+3)
  // Cycle 2: result = 10  (1+2+3+4)
  // Cycle 3: result = 14  (2+3+4+5)
}
```

Here `base@1` is the entire four-element array from the previous cycle, which
is then fed into `reduce`.

## Common patterns

| Pattern | Meaning |
|---------|---------|
| `reg@1` | Register value from the previous cycle |
| `arr@1[i]` | Element `i` of an array from the previous cycle |
| `comp@1.field` | Field of a composite component from the previous cycle |
| `(1 + x@1) * (1 - first)` | Increment with zero base case on cycle 0 |

[Prev](99_Arrays_and_Loops.md)
[Next](99_Externs.md)
