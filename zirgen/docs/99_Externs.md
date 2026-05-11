# Externs

Externs are the interface between a Zirgen circuit and the outside world. An
extern declaration introduces a name that the host environment must provide at
runtime. Externs can be used to feed inputs into the circuit, emit outputs, or
query execution state such as the current cycle number.

## Syntax

An extern that takes no arguments and returns nothing:

```
extern Log(msg: Val);
```

An extern that returns a value:

```
extern GetCycle() : Val;
```

An extern that takes a component type:

```
component PairVal(aArg: Val, bArg: Val) {
  a := aArg;
  b := bArg;
}

extern TakesPair(v: PairVal);
extern ReturnsPair() : PairVal;
```

The argument and return types can be `Val` or any component type. Component
arguments are flattened to their individual field elements at the FFI boundary.

## Behavior in `--test` mode

When you run `zirgen --test`, extern calls are not provided by a real host.
Instead the interpreter:

- Prints each call and its return value to stdout, e.g. `[0] GetCycle() -> (0)`
- Returns zero (or a zero-initialized component) for any extern that produces a
  value

This makes it straightforward to unit-test circuits without a full host
implementation. The printed log is the basis for FileCheck assertions in the
test suite.

From `zirgen/dsl/test/externs.zir`:

```
test val {
  TakesArg(ReturnsVal());
  // [0] ReturnsVal() -> (0)
  // [0] TakesArg(0) -> ()
}

test pair {
  TakesPair(ReturnsPair());
  // [0] ReturnsPair() -> (0, 1)
  // [0] TakesPair(0, 1) -> ()
}
```

Note that `ReturnsPair()` returns `(0, 1)` in test mode because the two fields
of `PairVal` are assigned indices 0 and 1 sequentially.

## Standard externs

The following externs are commonly used in tests and simulation. Each must be
declared explicitly with `extern` before use:

| Extern | Signature | Description |
|--------|-----------|-------------|
| `Log` | `(fmt: Val, ...)` | Print a formatted log message during test/simulation |
| `Output` | `(v: Val)` | Emit a single field element as output |
| `IsFirstCycle` | `() : Val` | Returns 1 on cycle 0, 0 otherwise |
| `GetCycle` | `() : Val` | Returns the current cycle index |

`IsFirstCycle` and `GetCycle` are commonly used to initialize inductive
registers. `Output` is used in tests to assert specific values.

## Full example

```
extern Output(v: Val);
extern GetCycle() : Val;

component Count(first: Val) {
  public a : Reg;
  a := Reg((1 + a@1) * (1 - first));
}

test {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Output(c.a);
  // [0] Output(0) -> ()
  // [1] Output(1) -> ()
  // [2] Output(2) -> ()
}
```

[Prev](99_Backs.md)
[Next](A1_Builtin_Components.md)
