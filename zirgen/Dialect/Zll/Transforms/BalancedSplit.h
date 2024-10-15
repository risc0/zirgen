#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace zirgen::Zll {

// Calculates a reasonable way to split a block, and moves the second half to a new block and
// returns it.
mlir::Block* balancedSplitBlock(mlir::Block* orig);

// Spltis the body of the given function, and invokes the second half as a call function.
void balancedSplitFunc(mlir::func::FuncOp func);

} // namespace zirgen::Zll
