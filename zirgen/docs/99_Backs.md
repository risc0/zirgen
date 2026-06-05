# Backs

Zirgen circuits are evaluated across multiple rows of an execution trace — one
row per _cycle_. A back-reference lets a component read the value that was
stored in a register (or a sub-component) on a previous cycle. This is the
primary mechanism for expressing multi-cycle state transitions and inductive
invariants in a circuit.

## Syntax

The back-reference operator is `@`:

```
<expr>@<distance>
```

`<distance>` is the number of cycles to look back. `@1` reads the value from
the immediately preceding cycle; `@2` reads two cycles back, and so on.

```
// Read the value of register `a` from the previous cycle.
prev := a@1;
```

## Back-Referencing a Register

The most common use-case is reading the previous cycle's register value to
compute the current one. The following implements a simple counter that
increments by one each cycle, starting at zero on the first cycle:

```
component Count(first: Val) {
  public a : Reg;
  a := Reg((1 + a@1) * (1 - first));
}
```

On the first cycle `first = 1`, so `a` is constrained to 0. On subsequent
cycles `first = 0`, so `a = a@1 + 1`.

## Back-Referencing a Component

You can apply `@` to any named component binding, not just primitive registers.
The result is the _whole component_ (including its public members) as it was on
the indicated prior cycle:

```
component PrevCount(first: Val) {
  public c    := Count(first);
  public prev := c@1;   // c as it was one cycle ago
}
```

Given `PrevCount`, `c@1.prev.a` reads the `a` field of the `Count` component
two cycles back (one step back from `c@1`'s own back-reference):

```
test prev_count {
  first := NondetReg(Isz(GetCycle()));
  c := PrevCount(first);
  Output(c@1.prev.a);   // value of `a` two cycles ago
}
```

## Back-Referencing an Array

`@` can also be applied to an array binding. After the `@` you have a full
`Array` in hand and can index into it normally:

```
// arr is Array<Reg, 4>; arr@1[i] is the i-th register from the prior cycle.
for i : 0..4 { arr[i] - arr@1[i] = 1; }
```

The `reduce … init … with …` form can also consume a back-referenced array:

```
result := [first, 1 - first] -> (
  0,
  reduce base@1 init 0 with Add
);
```

## Using `@0` (Self-Reference)

`@0` refers to the component _in the current scope_, which is useful when
passing a component as an argument that will take a back internally:

```
component DoubleBackOne(x: NondetReg) {
  NondetReg(2 * x@1)
}

component Top() {
  first := NondetReg(IsFirstCycle());
  public result : NondetReg;
  result := [first, 1 - first] -> (
    NondetReg(1),
    DoubleBackOne(result@0)   // pass the current result slot; DoubleBackOne reads @1
  );
}
```

## Backs Across Mux Arms

The [Muxes](05_Muxes.md) documentation covers an important edge case: taking a
back on a register that belongs only to one mux arm (not to the common super) is
undefined behavior. Always take backs on fields that are part of the _least
common super_ of the mux.

## Out-of-Range Backs

Reading further back than the available trace (e.g., `@1` on the very first
cycle) produces an undefined value. The interpreter will emit a warning in this
case. Guard back-references with a `first`-cycle flag:

```
a := Reg((1 + a@1) * (1 - first));
```

[Prev](05_Muxes.md)
