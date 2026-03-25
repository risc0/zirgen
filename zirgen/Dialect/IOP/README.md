# IOP Dialect

The **IOP (Interactive Oracle Proof)** dialect provides operations for verifier interaction with proof transcripts in zero-knowledge proof systems. It represents the verifier's side of the Fiat-Shamir transformed interactive protocol, enabling reading from proof transcripts, committing to hashes, and generating random challenges.

## Purpose

The IOP dialect provides:

- **Verifier transcript operations**: Read values from prover's transcript
- **Commitment operations**: Commit to hash digests in the transcript
- **Challenge generation**: Generate random field elements via Fiat-Shamir
- **Proof verification support**: Essential operations for ZK proof verification

## Position in Compilation Pipeline

The IOP dialect is used alongside Zll dialect operations to implement the verification phase of ZK proofs:

```
Circuit (Prover Side)
    ↓
Execution Trace
    ↓
Proof Generation
    ↓
[Proof Transcript]
    ↓
Verification (uses IOP dialect) ← YOU ARE HERE
    ↓
Accept/Reject
```

IOP operations are typically embedded within Zll functions that implement verification logic.

## Key Operations

Defined in `zirgen/Dialect/IOP/IR/Ops.td` (only 4 operations):

### Reading from Transcript
- `iop.read` - Read value(s) from the IOP
  - Syntax: `%vals = iop.read %iop : <type>`
  - Parameters: IOP handle, flip flag (boolean)
  - Returns: One or more scalar values (Val or Digest)
  - Reads prover-provided values from proof transcript

### Commitment
- `iop.commit` - Commit to a hash digest
  - Syntax: `iop.commit %iop, %digest : !zll.digest`
  - Parameters: IOP handle, digest to commit
  - Records commitment in transcript for challenge generation

### Random Challenge Generation
- `iop.rng_bits` - Get a random field element of specified bit width
  - Syntax: `%challenge = iop.rng_bits %iop, <bits> : !zll.val`
  - Parameters: IOP handle, number of bits
  - Returns: Random field element (via Fiat-Shamir)
  - Used for generating random challenges during verification

- `iop.rng_val` - Get a random field extension element
  - Syntax: `%challenge = iop.rng_val %iop : !zll.val`
  - Parameters: IOP handle
  - Returns: Random field element (possibly extension field)
  - Generates challenges in extension field if needed

## Type System

Defined in `zirgen/Dialect/IOP/IR/Types.td`:

### Core Type

**IOP** - Interactive Oracle Proof read stream for verifier
- Represents the verifier's view of the proof transcript
- Passed through verification functions as a handle
- Implements `CodegenTypeInterface` for code generation
- Syntax: `!iop.iop`

### Auxiliary Types

**AnyScalar** - Union of Val and Digest types
- Used for `iop.read` return values
- Can read either field elements (`!zll.val`) or digests (`!zll.digest`)

## Operation Traits

All IOP operations have these traits:
- **EvalOpAdaptor** - Evaluable in the interpreter
- **IsReduce** - Marked as reduction operations (affect state)

## Interactive Oracle Proof Protocol

IOP operations implement the Fiat-Shamir transformation of an interactive protocol:

1. **Prover commits** to values (represented as hashes)
2. **Verifier reads** committed values via `iop.read`
3. **Verifier commits** to hashes via `iop.commit`
4. **Verifier generates challenges** via `iop.rng_bits` or `iop.rng_val`
5. **Process repeats** for multiple rounds

The IOP handle maintains the transcript state, ensuring challenges are deterministically derived from prior commitments.

## Example

```mlir
func.func @verify(%iop: !iop.iop) {
  // Read prover's first message (field element)
  %val1 = iop.read %iop : !zll.val<BabyBear>

  // Read a digest commitment
  %digest1 = iop.read %iop : !zll.digest

  // Commit to a computed digest
  %computed = zll.hash %val1 : !zll.digest
  iop.commit %iop, %computed : !zll.digest

  // Generate random challenge (20 bits)
  %challenge = iop.rng_bits %iop, 20 : !zll.val<BabyBear>

  // Generate random extension field element
  %alpha = iop.rng_val %iop : !zll.val<BabyBear, 4>

  // Use challenges for verification...
  // ...

  return
}
```

## Fiat-Shamir Transformation

The IOP dialect implements the Fiat-Shamir heuristic:

**Interactive Protocol**:
```
Prover → value → Verifier
Prover ← challenge ← Verifier (random)
```

**Non-Interactive (IOP)**:
```
Prover generates transcript: [value, ...]
Verifier reads value: iop.read
Verifier derives challenge deterministically: iop.rng_bits
Challenge = Hash(transcript_so_far)
```

This transformation makes the protocol non-interactive while maintaining security.

## Integration with Zll

IOP operations are used within Zll functions:
- IOP handle passed as function argument
- Mixed with Zll operations for computation
- Enables verification logic implementation
- Coordinates transcript reading with constraint checking

```mlir
func.func @verify_step(%iop: !iop.iop, %buffer: !zll.buffer<...>) {
  // Read from IOP
  %witness = iop.read %iop : !zll.val<BabyBear>

  // Compute using Zll operations
  %computed = zll.add %witness, %const : !zll.val<BabyBear>

  // Generate challenge
  %challenge = iop.rng_bits %iop, 32 : !zll.val<BabyBear>

  // Continue verification...
}
```

## Code Generation

IOP operations implement `EvalOpAdaptor` and generate code for:
- **Rust**: Transcript reading/writing operations
- **C++**: Verification code with IOP handling
- **GPU**: May not be applicable (verification typically on CPU)

The `CodegenTypeInterface` on the IOP type ensures proper emission of IOP handle types across backends.

## Design Philosophy

1. **Minimal Interface**: Only 4 operations for complete IOP functionality
2. **Stateful Handle**: IOP maintains transcript state implicitly
3. **Type Safety**: Strongly typed reads and challenges
4. **Fiat-Shamir Native**: Built-in support for challenge derivation
5. **Composable**: Works seamlessly with Zll operations

## Use Cases

1. **Proof Verification**: Primary use case for ZK proof verifiers
2. **FRI Protocol**: Reading FRI query responses and generating challenges
3. **Polynomial Commitments**: Opening proofs and challenge generation
4. **Multi-round Protocols**: Supporting complex interactive protocols
5. **Recursion**: Verifying inner proofs in recursive constructions

## File Organization

```
zirgen/Dialect/IOP/
├── IR/
│   ├── Dialect.td      - Dialect definition (name: "iop")
│   ├── Ops.td          - 4 operation definitions
│   ├── Types.td        - IOP type definition
│   ├── IR.h            - Main header
│   ├── Dialect.cpp     - Dialect implementation
│   └── Ops.cpp         - Operation implementations
└── (No transformation passes - IOP used directly in Zll)
```

## Relationship to Other Dialects

### Works with Zll
- IOP operations embedded in Zll functions
- Uses Zll types (Val, Digest) for values
- Coordinates with Zll constraint checking

### Independent Functionality
- Not transformed by passes
- No lowering to other dialects
- Emitted directly to target code

## See Also

- [Zll Dialect](../Zll/README.md) - Host dialect for IOP operations
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [Code Generation](../../docs/CODEGEN.md) - Multi-target emission
- [Fiat-Shamir Transformation](https://en.wikipedia.org/wiki/Fiat%E2%80%93Shamir_heuristic) - Theoretical background
