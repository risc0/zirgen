#include "mlir/IR/BuiltinOps.h"

namespace zirgen {

void registerRunTestsCLOptions();
int runTests(mlir::ModuleOp module);

} // namespace zirgen
