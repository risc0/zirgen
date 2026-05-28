# zirgen/dsl

The DSL frontend: lexer, parser, AST, and the lowering pass that converts the AST into the ZHL MLIR dialect.

## How .zir Files Flow Through the Frontend

```
.zir source text
      │
      ▼
   Lexer (lexer.h / lexer.cpp)
      │  tokenizes source; supports preamble injection for builtins
      ▼
   Parser (parser.h / parser.cpp)
      │  recursive-descent; builds AST nodes from ast.h
      │  addPreamble() injects builtin component definitions before user source
      ▼
   AST (ast.h / ast.cpp)
      │  Module, Component, Block, Expression subtypes
      ▼
   lower() (lower.h / lower.cpp)
      │  single-pass AST walker; emits ZHL dialect ops into an MLIRContext
      ▼
   ZHL ModuleOp  →  passed to Typing::typeCheck()
```

## Key Files

| File | Purpose |
|------|---------|
| `lexer.h` / `lexer.cpp` | Tokenizer; handles include file expansion via `llvm::SourceMgr` |
| `parser.h` / `parser.cpp` | Recursive-descent parser; entry point is `Parser::parseModule()` |
| `ast.h` / `ast.cpp` | AST node hierarchy (`Module`, `Component`, `Block`, all `Expression` subtypes) |
| `lower.h` / `lower.cpp` | AST → ZHL MLIR; entry point is `zirgen::dsl::lower()` |
| `stats.h` / `stats.cpp` | Collects circuit statistics (op/constraint counts) for analysis |
| `passes/` | DSL-level MLIR passes: `FieldDCE`, `ElideTrivialStructs`, `GenerateCheck`, `InlinePure`, `HoistInvariants`, `TopologicalShuffle` |
| `examples/` | Small `.zir` example circuits |
| `test/` | FileCheck-based tests for DSL parsing and lowering |

## Preamble Injection

`Parser::addPreamble()` prepends a block of ZIR source text before the user's file. The builtin preamble (from `zirgen/Conversions/Typing/BuiltinComponents.h`) defines core types like `Reg`, `NondetReg`, `Val`, and arithmetic operations. This makes builtins available to all user circuits without explicit imports.

## passes/

DSL-level passes that run after lowering but before or during the typing phase:

- **FieldDCE** — removes unused component fields
- **ElideTrivialStructs** — collapses single-field structs to their element type
- **GenerateCheck** — synthesizes the validity (`check`) function from `constraint` ops
- **InlinePure** — inlines pure (side-effect-free) components at their call sites
- **HoistInvariants** — moves loop-invariant computations out of `map`/`reduce` bodies
- **TopologicalShuffle** — reorders ops in check functions to improve CSE after `if`-multiply expansion
