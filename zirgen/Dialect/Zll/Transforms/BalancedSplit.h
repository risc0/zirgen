#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace zirgen::Zll {

// Calculates a reasonable way to split a block, and outlines parts of via func::CallOp
void balancedSplitBlock(mlir::Block* orig, size_t nsplit);

// Spltis the body of the given function, and invokes the second half as a call function.
void balancedSplitFunc(mlir::func::FuncOp func);

} // namespace zirgen::Zll
