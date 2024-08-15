#ifndef ZHLT_TRANSFORM_PASSES
#define ZHLT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def HoistAllocs : Pass<"hoist-allocs", "mlir::ModuleOp"> {
  let summary = "Hoist alloc-only params to callers, then merge";
  let description = [{
    Removes some unnecessary allocations, allowing programs to conservatively
    invoke NondetReg without penalty. When the only use of some parameter is a
    call to NondetReg, that allocation will be hoisted to the callers of the
    component, and its parameter signature will be changed accordingly. The
    result may be a redundant allocation within the outer component, so nested
    calls to NondetReg will also be merged.
  }];
  let constructor = "zirgen::Zhlt::createHoistAllocsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def StripTests : Pass<"strip-tests", "mlir::ModuleOp"> {
  let summary = "Strip all tests for smaller generated code.";
  let constructor = "zirgen::Zhlt::createStripTestsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def GenerateSteps : Pass<"generate-steps", "mlir::ModuleOp"> {
  let summary = "Generate top-level StepOps for all externally callable functions ";
  let description = [{
     The following steps are generated:

     step$top: Constructs the component named "Top" and checks its constraints
     step$test$*: A StepOp to run each test
  }];
  let constructor = "zirgen::Zhlt::createStripTestsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

#endif // ZHLT_TRANSFORM_PASSES
