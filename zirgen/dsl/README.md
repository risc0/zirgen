# ZIR DSL Front-end

This directory contains the front-end of the Zirgen compiler: the lexer, parser, AST, and lowering pass that translate `.zir` source files into the ZHL MLIR dialect.

## Pipeline Overview

```
.zir source file
      │
      ▼
  ┌────────┐
  │ Lexer  │  (lexer.h / lexer.cpp)
  └────────┘
      │  token stream
      ▼
  ┌────────┐
  │ Parser │  (parser.h / parser.cpp)
  └────────┘
      │  ast::Module
      ▼
  ┌──────────────────────┐
  │ Type inference/check │  (Analysis/Typing/)
  └──────────────────────┘
      │
      ▼
  ┌─────────┐
  │ Lowering│  (lower.h / lower.cpp)
  └─────────┘
      │  ZHL MLIR dialect
      ▼
  (continues in zirgen/Dialect/)
```

The main entry point for compilation is `driver.cpp`, which wires together the stages above and feeds the result into the dialect lowering pipeline.

## Lexer (`lexer.h`, `lexer.cpp`)

`Lexer` operates on an `llvm::SourceMgr` and produces a stream of `Token` values. Key features:

- **Import handling** — `import "foo.zir";` is resolved and the included file's tokens are interleaved transparently. Files are deduplicated.
- **Preamble injection** — the driver injects built-in definitions (e.g. `NondetReg`, `Reg`, arithmetic ops) via `addPreamble()` before parsing the user file.
- **`@` token** — lexed as `tok_back`; immediately follows an expression to form a back-reference (`x@1`).
- **`..` token** — lexed as `tok_range`; used in loop ranges (`0..4`).
- **`->`** — lexed as `tok_mux`; separates a selector from its arms.

Notable token kinds (from `enum Token`):

| Token | Meaning |
|-------|---------|
| `tok_component` | `component` keyword |
| `tok_extern` | `extern` keyword |
| `tok_for` | `for` keyword |
| `tok_reduce` | `reduce` keyword |
| `tok_back` (`@`) | Back-reference operator |
| `tok_range` (`..`) | Range operator |
| `tok_mux` (`->`) | Mux arm separator |
| `tok_define` (`:=`) | Definition operator |
| `tok_global` | `global` keyword |
| `tok_public` | `public` keyword |

## Parser (`parser.h`, `parser.cpp`)

`Parser` is a recursive-descent parser. Its only public entry point is `parseModule()`, which returns an `ast::Module` or reports errors via `getErrors()`.

Key parse rules:

| Method | Constructs |
|--------|-----------|
| `parseComponent()` | `component Foo<T: Type>(args…) { body }` |
| `parseExtern()` | `extern Foo(args…) : ReturnType;` |
| `parseTest()` | `test name { body }` |
| `parseBlock()` | `{ stmt; stmt; … }` |
| `parseMap()` | `for i : start..end { body }` |
| `parseReduce()` | `reduce arr init seed with Op` |
| `parseConditional()` | `[sel, …] -> (arm, arm, …)` |
| `parseBack()` | `expr@N` |
| `parseSubscript()` | `expr[i]` |
| `parseSpecialize()` | `Ident<T, N>` |

Type parameters use angle brackets: `component Foo<T: Type, N: Val>(…)`.

## AST (`ast.h`, `ast.cpp`)

The AST is a tree of `Node<T>` subclasses. All nodes carry an `SMLoc` for diagnostics.

### Expression nodes

| Class | Syntax |
|-------|--------|
| `Literal` | `42`, `0xff`, `0b101` |
| `StringLiteral` | `"text"` |
| `Ident` | `foo` |
| `Lookup` | `a.b` |
| `Subscript` | `a[i]` |
| `Specialize` | `Foo<T, N>` |
| `Construct` | `Foo(a, b)` |
| `Block` | `{ stmts… }` |
| `Map` | `for i : range { body }` |
| `Reduce` | `reduce arr init seed with Op` |
| `Switch` | `[sel] -> (arm, …)` |
| `Range` | `start..end` |
| `Back` | `expr@N` |
| `ArrayLiteral` | `[a, b, c]` |

### Statement nodes

| Class | Syntax |
|-------|--------|
| `Definition` | `name := expr` |
| `Declaration` | `name : Type` |
| `Constraint` | `lhs = rhs` |
| `Void` | standalone expression as statement |
| `Directive` | `#[attr]` |

### Top-level declarations

`Component` covers all top-level forms. Its `kind` field distinguishes:

| Kind | Example |
|------|---------|
| `Object` | `component Foo(…) { … }` |
| `Function` | `function foo(…) { … }` |
| `Extern` | `extern Foo(…) : T;` |
| `Argument` | type parameter `argument T: Type` |
| `Major` | `define Foo := Bar;` |

## Lowering (`lower.h`, `lower.cpp`)

`lower.cpp` walks the `ast::Module` and emits ZHL MLIR operations. After lowering, the module enters the dialect pipeline defined in `zirgen/Dialect/`.

## Directory Contents

```
dsl/
├── ast.h / ast.cpp         # AST node definitions
├── lexer.h / lexer.cpp     # Tokenizer
├── parser.h / parser.cpp   # Recursive-descent parser
├── lower.h / lower.cpp     # AST → ZHL lowering
├── driver.cpp              # Compilation pipeline glue
├── Analysis/               # Type inference and checking
├── passes/                 # DSL-level optimization passes
├── test/                   # .zir unit test suite (120+ files)
└── examples/               # fibonacci.zir, calculator/, …
```
