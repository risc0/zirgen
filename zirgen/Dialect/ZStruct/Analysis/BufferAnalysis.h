// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// Analyses what buffers are used by a module.

#include "mlir/Pass/AnalysisManager.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace zirgen::ZStruct {

// A description of a buffer used by ZStruct
struct BufferDesc {
  mlir::StringAttr name;
  Zll::BufferKind kind;

  // Tap register group id, if present.
  std::optional<size_t> regGroupId = std::nullopt;

  // True if this buffer is a global buffer, false if it is per-cycle.
  bool global = false;

  // Size of this buffer, in "Val"s.
  size_t regCount = 0;

  // Constructs the Zll::BufferType of this buffer.
  mlir::Type getType(mlir::MLIRContext* ctx) const;

  // The set of layouts in the circuit that apply to this buffer, i.e.
  // the "layout" field of all BoundBufferAttrs.
  llvm::SetVector<mlir::Attribute> layouts;
};

// This class provides access to the list of buffers used by a ZHLT
// module.  Currently the implementation is hardcoded, but we hope to
// have a more flexible buffer scheme in the future.  In preparation for that, we
// want to centralize everything that depends on types of buffers.
class BufferAnalysis {
public:
  BufferAnalysis(mlir::ModuleOp op);

  llvm::ArrayRef<BufferDesc> getTapBuffers() const { return tapBuffers; };

  llvm::SmallVector<BufferDesc> getAllBuffers() const {
    auto bufs = llvm::to_vector(llvm::make_second_range(buffers));
    // Make sure we return buffers in a determinstic order.
    llvm::sort(bufs, [&](auto a, auto b) { return a.name.strref() < b.name.strref(); });
    return bufs;
  }

  const BufferDesc& getBuffer(llvm::StringRef bufferName) const { return buffers.at(bufferName); }

  // Returns the operation defining the layout that we would provide to
  // the given top-level block argument.
  std::pair<ZStruct::GlobalConstOp, BufferDesc>
  getLayoutAndBufferForArgument(mlir::BlockArgument layoutArg);

private:
  llvm::DenseMap<llvm::StringRef, BufferDesc> buffers;
  llvm::SmallVector<BufferDesc> tapBuffers;
};

} // namespace zirgen::ZStruct
