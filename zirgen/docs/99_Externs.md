# Externs

An *extern* declares a function that is implemented by the **prover host** rather than inside the circuit. Externs are the primary mechanism for a circuit to receive inputs (e.g. the current cycle number, program memory, or any oracle query) and to produce debug output.

## Key Properties

- **Nondeterministic** — the verifier never executes the extern; it only sees the value the prover wrote into the witness. The circuit author is responsible for adding constraints that prove the extern's return value is correct.
- **Call-once per cycle** — each extern call is recorded during witness generation and replayed during verification.
- **Not part of the proof** — extern calls themselves are not proven. Only the polynomial constraints that follow are verified.

## Declaration Syntax

```zir
extern FunctionName(param1: Type1, param2: Type2) : ReturnType;
```

If the extern returns nothing, omit the `: ReturnType` part:

```zir
extern Output(v: Val);
```

If the extern returns a component type (a struct-like aggregate), declare that component first:

```zir
component PairVal(aArg: Val, bArg: Val) {
  a := aArg;
  b := bArg;
}

extern ReturnsPair() : PairVal;
extern TakesPair(v: PairVal);
```

## Usage

Call an extern like any other component:

```zir
cycle := NondetReg(GetCycle());
Output(cycle);
TakesPair(ReturnsPair());
```

Externs that return values should be captured in a `NondetReg` so the prover-supplied value is written into the witness. The `NondetReg` provides the storage; additional constraints make the value trustworthy.

## Example: Cycle Counter

The canonical extern pattern — receiving the current cycle number from the host and constraining it to advance monotonically:

```zir
extern GetCycle() : Val;

component CycleCounter() {
  global total_cycles := NondetReg(6);

  cycle := NondetReg(GetCycle());       // prover supplies the value
  public is_first_cycle := IsZero(cycle);

  // Constrain: on non-first cycles, cycle must equal previous + 1
  [is_first_cycle, 1 - is_first_cycle] -> ({
    // (nothing extra on the first cycle)
  }, {
    cycle = cycle@1 + 1;                // polynomial constraint
  });

  cycle
}
```

Without the `cycle = cycle@1 + 1` constraint, a malicious prover could supply any cycle number.

## Example: Input and Output

```zir
extern GetCycle() : Val;
extern Output(v: Val);

test count {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Output(c.a);  // sends register value to the host
}
```

## Example: Aggregate Return Types

```zir
component PairVal(aArg: Val, bArg: Val) {
  a := aArg;
  b := bArg;
}

extern ReturnsPair() : PairVal;
extern TakesPair(v: PairVal);

test pair {
  TakesPair(ReturnsPair());
  // host sees: ReturnsPair() -> (0, 1)  then  TakesPair(0, 1) -> ()
}
```

## Built-in Externs

The zirgen standard preamble provides several commonly used externs:

| Extern | Signature | Purpose |
|--------|-----------|---------|
| `GetCycle` | `() : Val` | Returns the current cycle index |
| `IsFirstCycle` | `() : Val` | Returns 1 on cycle 0, else 0 |
| `Output` | `(v: Val)` | Sends a value to the host for inspection |
| `Log` | `(fmt: String, …)` | Prints a formatted debug string |

## Security Considerations

An extern return value has **no intrinsic trust** — the verifier cannot see it directly. Always add polynomial constraints to certify the extern's output:

```zir
// Bad: prover can return anything
x := NondetReg(GetInput());

// Good: constrain x to the valid range (e.g. a byte)
x := NondetReg(GetInput());
// use ByteReg, InRange, or explicit constraints to bound x
```

Failing to constrain an extern return allows a malicious prover to forge witnesses without detection.
