# Externs

An _extern_ is a named operation whose implementation is provided by the host
environment rather than expressed as Zirgen constraints. Externs are the escape
hatch through which a circuit communicates with the outside world: they can
produce _nondeterministic_ witness values (inputs that the prover supplies but
that must still be constrained), emit side-effects during simulation (e.g.
logging, assertions), or invoke host-side callbacks during proof generation.

Because externs are not constrained by definition, any value they return must be
explicitly constrained by the circuit before it can be trusted.

## Declaration Syntax

```
extern <Name>(<param>: <Type>, ...) [: <ReturnType>];
```

* If the extern produces no value, omit the return type annotation.
* If it returns a single `Val`, write `: Val`.
* If it returns a component type, write the component name.

**Examples:**

```
// Side-effecting extern — no return value.
extern Output(v: Val);

// Extern that returns a Val.
extern GetCycle() : Val;

// Extern that returns a composite component type.
extern ReturnsPair() : PairVal;

// Extern that accepts a composite type.
extern TakesPair(v: PairVal);
```

## Calling Externs

Externs are called exactly like component constructors:

```
Output(42);
cycle := GetCycle();
```

During simulation (the `--test` interpreter), the runtime provides default
implementations: externs that return a value yield `0` (or a zero-initialized
struct), and side-effecting externs print their arguments to stdout in a
canonical format, e.g.:

```
[0] Output(5) -> ()
[0] ReturnsVal() -> (0)
```

## Using Returned Values in Constraints

A value returned by an extern is nondeterministic — the prover supplies it. To
make the circuit _verify_ a returned value, record it in a `NondetReg` and then
constrain it explicitly:

```
// Wrong: the raw extern return value is unconstrained.
cycle := GetCycle();

// Right: register the value and constrain it.
cycle := NondetReg(GetCycle());
cycle = cycle@1 + 1;   // or whatever invariant must hold
```

As a rule of thumb from [Builtin Components](A1_Builtin_Components.md): _`Val`s
computed nondeterministically or returned by externs can't be used in constraints
without registerizing, so use `NondetReg` for such values._

## Common Built-in Externs

Several externs are provided by the Zirgen runtime and appear frequently in
tests and circuits:

| Extern | Return | Description |
|--------|--------|-------------|
| `GetCycle()` | `Val` | The index of the current row in the execution trace (0-based). |
| `IsFirstCycle()` | `Val` | 1 on cycle 0, 0 otherwise. |
| `Log(fmt, ...)` | — | Print a formatted message during simulation. |
| `Isz(v)` | `Val` | 1 if `v == 0`, 0 otherwise (nondeterministic; must be constrained). |

## Full Example

The following circuit uses `GetCycle` to initialize a counter and `Output` to
emit the result at each step. `GetCycle` returns a nondeterministic value, so it
is immediately wrapped in `NondetReg`:

```
extern Output(v: Val);
extern GetCycle() : Val;

component Count(first: Val) {
  public a : Reg;
  a := Reg((1 + a@1) * (1 - first));
}

test count {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Output(c.a);
}
```

Running this with `--test --test-cycles 4` produces:

```
[0] Output(0) -> ()
[1] Output(1) -> ()
[2] Output(2) -> ()
[3] Output(3) -> ()
```

## Externs vs. Constraints

| | Constraints | Externs |
|---|---|---|
| Implementation | Encoded in the circuit arithmetization | Provided by the host |
| Verified by verifier | Yes | No (only indirectly, through constraints on their outputs) |
| Typical use | Defining invariants | Supplying witness values, logging, I/O |

[Prev](05_Muxes.md)
