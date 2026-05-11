# Arrays and Loops

Arrays in Zirgen are fixed-size, homogeneous collections. The element type and
length are both part of the type: `Array<T, N>` is an array of `N` elements of
type `T`.

## Array literals

An array literal is written with square brackets:

```
seq := [x, x + 1, x + 2];
```

The type of `seq` above is `Array<Val, 3>`.

## Indexed access

Individual elements are read with `arr[i]`:

```
Output(seq[0]);   // first element
Output(seq[1]);   // second element
```

## The `for` map expression

The primary way to construct an array in Zirgen is with a `for` map expression,
which produces one output element per iteration:

```
arr := for i : 0..4 { Reg(i) };
```

This creates an `Array<Reg, 4>` whose `i`-th element is `Reg(i)`. The range
`0..N` is exclusive of the upper bound, so `0..4` yields indices 0, 1, 2, 3.

You can also declare the array type explicitly before assigning it:

```
arr : Array<Reg, 4>;
arr := for i : 0..4 { Reg(i) };
```

## Iterating over an existing array

A `for` expression can also range over an existing array rather than a numeric
range:

```
for x : arr { ... }
```

This is useful when passing array elements to a function or building a derived
array from an existing one. The `reduce` keyword accumulates an array into a
single value:

```
sum := reduce arr init 0 with Add;
```

## Full example — Sequence component

From `zirgen/dsl/test/test_array.zir`:

```
extern Output(v: Val);

component Sequence(x: Val) {
  [x, x + 1, x + 2]
}

test {
  seq := Sequence(2);
  Output(seq[0]);   // Output(2)
  Output(seq[1]);   // Output(3)
  Output(seq[2]);   // Output(4)
}
```

## Full example — Induction over arrays

Arrays interact naturally with back-references (see [Backs](99_Backs.md)). The
following circuit computes cumulative products across cycles. Each cycle, `arr[i]`
holds the product of `(i+1)` over all previous cycles.

From `zirgen/dsl/test/induction_on_arrays.zir`:

```
extern IsFirstCycle() : Val;

component Top() {
  first := NondetReg(IsFirstCycle());

  public arr : Array<Reg, 4>;
  arr := for i : 0..4 {
    Reg([first, 1-first] -> (
      i + 1,
      (i + 1) * arr@1[i]
    ))
  };
}

test {
  arr := Top().arr;
  Log("%u %u %u %u", arr[0], arr[1], arr[2], arr[3]);
}
// Cycle 0: 1 2 3 4
// Cycle 1: 1 4 9 16
// Cycle 2: 1 8 27 64
```

Note `arr@1[i]`: the `@1` reads the array from the previous cycle, then `[i]`
indexes into it. See [Backs](99_Backs.md) for details.

[Prev](05_Muxes.md)
[Next](99_Backs.md)
