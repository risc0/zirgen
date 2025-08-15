#include "mlir/Transforms/Passes.h"
#include "zirgen/Main/Target.h"

namespace zirgen {

std::unique_ptr<llvm::raw_ostream> openOutput(llvm::StringRef filename);

void emitPoly(mlir::ModuleOp mod, mlir::StringRef circuitName, llvm::StringRef protocolInfo);

void emitTarget(const zirgen::CodegenTarget& target,
                mlir::ModuleOp mod,
                mlir::ModuleOp stepFuncs,
                const zirgen::codegen::CodegenOptions& opts,
                unsigned stepSplitCount = 1);

} // namespace zirgen
