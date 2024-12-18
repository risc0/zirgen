// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zirgen/Dialect/Zll/Analysis/MixPowerAnalysis.h"

#include <filesystem>
#include <fstream>
#include <unordered_set>

#include "mustache.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/include/llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace kainjow::mustache;
using namespace zirgen::Zll;

namespace fs = std::filesystem;

namespace zirgen {

namespace {

using PoolsSet = std::unordered_set<std::string>;
enum class FuncKind {
  Step,
  PolyFp,
  PolyExt,
};

// Apply fixups; apparently gpus use different names than cpus.
std::string gpuMapName(std::string name) {
  if (name == "code")
    return "ctrl";
  if (name == "global")
    return "out";
  return name;
}

class GpuStreamEmitterImpl : public GpuStreamEmitter {
  llvm::raw_ostream& ofs;
  std::string suffix;

public:
  GpuStreamEmitterImpl(llvm::raw_ostream& ofs, const std::string& suffix)
      : ofs(ofs), suffix(suffix) {}

  void emitStepFunc(const std::string& name, func::FuncOp func) override {
    // TODO: remove this hack once host-side recursion code is unified with rv32im.
    bool isRecursion = func.getName() == "recursion";
    if (isRecursion) {
      emitStepFuncRecursion(name, func);
    } else {
      mustache tmpl = openTemplate("zirgen/compiler/codegen/gpu/step.tmpl" + suffix);

      FileContext ctx;
      std::stringstream ss;
      for (auto arg : func.getArguments()) {
        ctx.vars[arg] = llvm::formatv("arg{0}", arg.getArgNumber()).str();
        ss << ", ";
        if (suffix == ".metal") {
          ss << "device ";
        }
        ss << "Fp* arg" << arg.getArgNumber();
      }

      list lines;
      PoolsSet pools;
      emitStepBlock(func.front(), ctx, lines, /*depth=*/0, isRecursion, pools);

      tmpl.render(
          object{
              {"name", func.getName().str()},
              {"args", ss.str()},
              {"fn", "step_" + name},
              {"body", lines},
          },
          ofs);
    }
  }

  // TODO: remove this hack once host-side recursion code is unified with rv32im.
  void emitStepFuncRecursion(const std::string& name, func::FuncOp func) {
    if (name != "compute_accum" && name != "verify_accum") {
      mustache tmpl = openTemplate("zirgen/compiler/codegen/gpu/recursion/step.tmpl" + suffix);
      tmpl.render(object{}, ofs);
      return;
    }

    mustache tmpl =
        openTemplate("zirgen/compiler/codegen/gpu/recursion/step_" + name + ".tmpl" + suffix);

    FileContext ctx;
    for (auto [argNum, arg] : llvm::enumerate(func.getArguments())) {
      if (auto name = func.getArgAttrOfType<StringAttr>(argNum, "zirgen.argName")) {
        ctx.vars[arg] = gpuMapName(name.str());
      }
    }
    list lines;
    PoolsSet pools;
    emitStepBlock(func.front(), ctx, lines, /*depth=*/0, true, pools);

    std::string poolArgs;
    for (std::string pool : pools) {
      if (!poolArgs.empty()) {
        poolArgs.append(", ");
      }
      std::string qualifier;
      if (suffix == ".metal") {
        qualifier = "device ";
      }
      poolArgs.append(llvm::formatv("{0}FpExt* {1}", qualifier, pool).str());
    }

    tmpl.render(object{{"name", func.getName().str()},
                       {"fn", "step_" + name},
                       {"body", lines},
                       {"pools", poolArgs}},
                ofs);
  }

  void
  emitPoly(mlir::func::FuncOp func, size_t splitIndex, size_t splitCount, bool declsOnly) override {
    MixPowAnalysis mixPows(func);

    auto circuitName = lookupModuleAttr<CircuitNameAttr>(func);

    mustache tmpl;
    bool isRecursion = func.getName() == "recursion";
    if (isRecursion && suffix == ".cu") {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/recursion/eval_check.tmpl" + suffix);
    } else {
      tmpl = openTemplate("zirgen/compiler/codegen/gpu/eval_check.tmpl" + suffix);
    }

    list funcProtos;
    list funcs;

    size_t curSplitIndex = 0;
    for (mlir::func::FuncOp calledFunc : mixPows.getCalledFuncs()) {
      FileContext ctx;
      std::string args;

      for (auto [argNum, arg] : llvm::enumerate(calledFunc.getArguments())) {
        std::string argName = llvm::formatv("arg{0}", argNum).str();
        ctx.vars[arg] = argName;

        args += ", ";

        TypeSwitch<Type>(arg.getType())
            .Case<ValType>([&](auto valType) {
              if (valType.getFieldK() > 1)
                args += "FpExt";
              else
                args += "Fp";
            })
            .Case<BufferType>([&](auto bufType) {
              if (bufType.getKind() != BufferKind::Temporary)
                args += "const ";
              if (bufType.getElement().getFieldK() > 1)
                args += "FpExt*";
              else
                args += "Fp*";
            })
            .Case<ConstraintType>([&](auto) { args += "FpExt"; })
            .Default([&](Type ty) {
              llvm::errs() << "Unknown type to pass to call: " << ty << "\n";
              assert(false);
            });
        args += " " + argName;
      }

      funcProtos.push_back(object{{"args", args}, {"fn", calledFunc.getName().str()}});

      if (declsOnly || (curSplitIndex++ % splitCount) != splitIndex)
        continue;

      list lines;
      for (Operation& op : calledFunc.front().without_terminator()) {
        emitOp(&op, ctx, lines, mixPows);
      }
      lines.push_back("return " + ctx.use(calledFunc.front().getTerminator()->getOperand(0)) + ";");

      funcs.push_back(object{
          {"args", args},
          {"fn", calledFunc.getName().str()},
          {"block", lines},
      });
    }

    // Main function
    FileContext ctx;
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      if (auto argName = func.getArgAttrOfType<StringAttr>(idx, "zirgen.argName")) {
        ctx.vars[arg] = gpuMapName(argName.str());
      }
    }

    std::string mainArgs = ", "
                           "const Fp* ctrl, "
                           "const Fp* out, "
                           "const Fp* data, "
                           "const Fp* mix, "
                           "const Fp* accum";

    funcProtos.push_back(object{{"args", mainArgs}, {"fn", "poly_fp"}});

    if (!declsOnly && (curSplitIndex++ % splitCount) == splitIndex) {
      list lines;
      for (Operation& op : func.front().without_terminator()) {
        emitOp(&op, ctx, lines, mixPows);
      }
      Value retVal = func.front().getTerminator()->getOperand(0);
      lines.push_back(llvm::formatv("return {0};", ctx.use(retVal)).str());

      funcs.push_back(object{{"args", mainArgs}, {"fn", "poly_fp"}, {"block", lines}});
    }
    if (declsOnly) {
      tmpl.render(
          object{
              {"decls", object{{"declFuncs", funcProtos}}},
              {"num_mix_powers", std::to_string(mixPows.getPowersNeeded().size())},
              {"cppNamespace", circuitName.getCppNamespace()},
          },
          ofs);
    } else {
      tmpl.render(
          object{
              {"defs", object{}},
              {"funcs", funcs},
              {"name", func.getName().str()},
              {"num_mix_powers", std::to_string(mixPows.getPowersNeeded().size())},
              {"cppNamespace", circuitName.getCppNamespace()},
          },
          ofs);
    }
  }

private:
  void emitStepBlock(Block& block,
                     FileContext& ctx,
                     list& lines,
                     size_t depth,
                     bool isRecursion,
                     PoolsSet& pools) {
    std::string indent(depth * 2, ' ');
    for (Operation& op : block.without_terminator()) {
      mlir::TypeSwitch<Operation*>(&op)
          .Case<NondetOp>([&](NondetOp op) {
            lines.push_back(indent + "{");
            emitStepBlock(op.getInner().front(), ctx, lines, depth + 1, isRecursion, pools);
            lines.push_back(indent + "}");
          })
          .Case<IfOp>([&](IfOp op) {
            lines.push_back(indent +
                            llvm::formatv("if ({0} != 0) {{", ctx.use(op.getCond())).str());
            emitStepBlock(op.getInner().front(), ctx, lines, depth + 1, isRecursion, pools);
            lines.push_back(indent + "}");
          })
          .Case<ExternOp>([&](ExternOp op) {
            // TODO: remove this hack once host-side recursion code is unified with rv32im.
            if (isRecursion) {
              emitExternRecursion(op, ctx, lines, depth, pools);
            } else {
              emitExtern(op, ctx, lines, depth);
            }
          })
          .Default([&](Operation* op) {
            if (isRecursion) {
              emitStepOp(op, ctx, lines, depth);
            } else {
              emitOperation(op, ctx, lines, depth, "Fp", FuncKind::Step);
            }
          });
    }
  }

  // TODO: remove this hack once host-side recursion code is unified with rv32im.
  void
  emitExternRecursion(ExternOp op, FileContext& ctx, list& lines, size_t depth, PoolsSet& pools) {
    std::string indent(depth * 2, ' ');
    std::string pool = op.getExtra().str();
    pools.insert(pool);
    if (op.getName() == "plonkWriteAccum") {
      lines.push_back(indent + llvm::formatv("{0}[cycle] = FpExt({1}, {2}, {3}, {4});",
                                             pool,
                                             emitOperand(op, ctx, 0),
                                             emitOperand(op, ctx, 1),
                                             emitOperand(op, ctx, 2),
                                             emitOperand(op, ctx, 3))
                                   .str());
    } else if (op.getName() == "plonkReadAccum") {
      for (size_t i = 0; i < kBabyBearExtSize; i++) {
        lines.push_back(
            indent +
            llvm::formatv("auto {0} = {1}[cycle].elems[{2}];", ctx.def(op.getResult(i)), pool, i)
                .str());
      }
    } else {
      for (size_t i = 0; i < op.getIn().size(); i++) {
        lines.push_back(
            indent + llvm::formatv("host_args.at({0}) = {1};", i, ctx.use(op.getOperand(i))).str());
      }
      lines.push_back(indent + llvm::formatv("host(ctx, {0}, {1}, host_args.data(), {2}, "
                                             "host_outs.data(), {3});",
                                             escapeString(op.getName()),
                                             escapeString(op.getExtra()),
                                             op.getIn().size(),
                                             op.getOut().size())
                                   .str());
      for (size_t i = 0; i < op.getOut().size(); i++) {
        lines.push_back(
            indent +
            llvm::formatv("auto {0} = host_outs.at({1});", ctx.def(op.getResult(i)), i).str());
      }
    }
  }

  void emitExtern(ExternOp op, FileContext& ctx, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');
    for (size_t i = 0; i < op.getIn().size(); i++) {
      lines.push_back(indent +
                      llvm::formatv("extern_args[{0}] = {1};", i, ctx.use(op.getOperand(i))).str());
    }
    std::string externSuffix;
    if (op.getName().starts_with("plonkRead") || op.getName().starts_with("plonkWrite")) {
      externSuffix = llvm::formatv("_{0}", op.getExtra()).str();
    }
    lines.push_back(indent +
                    llvm::formatv("extern_{0}{1}(ctx, cycle, {2}, extern_args, extern_outs);",
                                  op.getName(),
                                  externSuffix,
                                  escapeString(op.getExtra()))
                        .str());
    for (size_t i = 0; i < op.getOut().size(); i++) {
      lines.push_back(
          indent +
          llvm::formatv("auto {0} = extern_outs[{1}];", ctx.def(op.getResult(i)), i).str());
    }
  }

  void emitOperation(Operation* op,
                     FileContext& ctx,
                     list& lines,
                     size_t depth,
                     const char* type,
                     FuncKind kind,
                     MixPowAnalysis* mixPows = nullptr) {
    std::string indent(depth * 2, ' ');
    std::string locStr;
    llvm::raw_string_ostream locStrStream(locStr);
    op->getLoc()->print(locStrStream);
    lines.push_back(indent + "// " + locStrStream.str());
    mlir::TypeSwitch<Operation*>(op)
        .Case<ConstOp>([&](ConstOp op) {
          lines.push_back(indent + llvm::formatv("{0} {1}({2});",
                                                 type,
                                                 ctx.def(op.getOut()),
                                                 emitPolynomialAttr(op, "coefficients"))
                                       .str());
        })
        .Case<GetOp>([&](GetOp op) {
          auto out = ctx.def(op.getOut());
          if (kind == FuncKind::Step) {
            lines.push_back(indent +
                            llvm::formatv("auto {0} = {1}[{2} * steps + ((cycle - {3}) & mask)];",
                                          out,
                                          ctx.use(op->getOperand(0)),
                                          emitIntAttr(op, "offset"),
                                          emitIntAttr(op, "back"))
                                .str());
            if (op->hasAttr("unchecked")) {
              lines.push_back(indent +
                              llvm::formatv("if ({0} == Fp::invalid()) {0} = 0;", out).str());
            } else {
              lines.push_back(indent + llvm::formatv("assert({0} != Fp::invalid());", out).str());
            }
          } else {
            lines.push_back(
                indent +
                llvm::formatv("auto {0} = {1}[{2} * steps + ((cycle - kInvRate * {3}) & mask)];",
                              out,
                              ctx.use(op->getOperand(0)),
                              emitIntAttr(op, "offset"),
                              emitIntAttr(op, "back"))
                    .str());
          }
        })
        .Case<SetOp>([&](SetOp op) {
          std::string inner((depth + 1) * 2, ' ');
          lines.push_back(indent + "{");
          std::string specifier;
          if (suffix == ".metal") {
            specifier = "device ";
          }
          lines.push_back(inner + llvm::formatv("{2}auto& reg = {0}[{1} * steps + cycle];",
                                                ctx.use(op->getOperand(0)),
                                                emitIntAttr(op, "offset"),
                                                specifier)
                                      .str());
          lines.push_back(inner + llvm::formatv("assert(reg == Fp::invalid() || reg == {0});",
                                                ctx.use(op->getOperand(1)))
                                      .str());
          lines.push_back(inner + llvm::formatv("reg = {0};", ctx.use(op->getOperand(1))).str());
          lines.push_back(indent + "}");
        })
        .Case<GetGlobalOp>([&](GetGlobalOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1}[{2}];",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 emitIntAttr(op, "offset"))
                                       .str());
        })
        .Case<SetGlobalOp>([&](SetGlobalOp op) {
          lines.push_back(indent + llvm::formatv("{0}[{1}] = {2};",
                                                 ctx.use(op->getOperand(0)),
                                                 emitIntAttr(op, "offset"),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<EqualZeroOp>([&](EqualZeroOp op) {
          lines.push_back(indent + llvm::formatv("assert({0} == 0 && \"eqz failed at: {1}\");",
                                                 ctx.use(op->getOperand(0)),
                                                 emitLoc(op))
                                       .str());
        })
        .Case<IsZeroOp>([&](IsZeroOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = ({1} == 0) ? Fp(1) : Fp(0);",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)))
                                       .str());
        })
        .Case<InvOp>([&](InvOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = inv({1});",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)))
                                       .str());
        })
        .Case<AddOp>([&](AddOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} + {2};",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<SubOp>([&](SubOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} - {2};",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<NegOp>([&](NegOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = -{1};",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)))
                                       .str());
        })
        .Case<MulOp>([&](MulOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} * {2};",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<BitAndOp>([&](BitAndOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = Fp({1}.asUInt32() & {2}.asUInt32());",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<ModOp>([&](ModOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = Fp({1}.asUInt32() % {2}.asUInt32());",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)))
                                       .str());
        })
        .Case<TrueOp>([&](TrueOp op) {
          lines.push_back(indent +
                          llvm::formatv("FpExt {0} = FpExt(0);", ctx.def(op.getOut())).str());
        })
        .Case<AndEqzOp>([&](AndEqzOp op) {
          lines.push_back(indent + llvm::formatv("FpExt {0} = {1} + {2} * poly_mix[{3}];",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)),
                                                 mixPows->getMixPowIndex(op))
                                       .str());
        })
        .Case<AndCondOp>([&](AndCondOp op) {
          lines.push_back(indent + llvm::formatv("FpExt {0} = {1} + {2} * {3} * poly_mix[{4}];",
                                                 ctx.def(op.getOut()),
                                                 ctx.use(op->getOperand(0)),
                                                 ctx.use(op->getOperand(1)),
                                                 ctx.use(op->getOperand(2)),
                                                 mixPows->getMixPowIndex(op))
                                       .str());
        })
        .Default([&](Operation* op) -> std::string {
          llvm::errs() << "Found invalid op during codegen!\n";
          llvm::errs() << *op << "\n";
          throw std::runtime_error("invalid op");
        });
  }

  void emitStepOp(Operation* op, FileContext& ctx, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');
    mlir::TypeSwitch<Operation*>(op)
        .Case<ConstOp>([&](ConstOp op) {
          lines.push_back(indent + llvm::formatv("Fp {0}({1});",
                                                 ctx.def(op.getOut()),
                                                 emitPolynomialAttr(op, "coefficients"))
                                       .str());
        })
        .Case<GetOp>([&](GetOp op) {
          lines.push_back(indent +
                          llvm::formatv("auto {0} = {1}[{2} * steps + ((cycle - {3}) & mask)];",
                                        ctx.def(op.getOut()),
                                        emitOperand(op, ctx, 0),
                                        emitIntAttr(op, "offset"),
                                        emitIntAttr(op, "back"))
                              .str());
        })
        .Case<GetGlobalOp>([&](GetGlobalOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1}[{2}];",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0),
                                                 emitIntAttr(op, "offset"))
                                       .str());
        })
        .Case<AddOp>([&](AddOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} + {2};",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0),
                                                 emitOperand(op, ctx, 1))
                                       .str());
        })
        .Case<SubOp>([&](SubOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} - {2};",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0),
                                                 emitOperand(op, ctx, 1))
                                       .str());
        })
        .Case<MulOp>([&](MulOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = {1} * {2};",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0),
                                                 emitOperand(op, ctx, 1))
                                       .str());
        })
        .Case<InvOp>([&](InvOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = inv({1});",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0))
                                       .str());
        })
        .Case<NegOp>([&](NegOp op) {
          lines.push_back(indent + llvm::formatv("auto {0} = -{1};",
                                                 ctx.def(op.getOut()),
                                                 emitOperand(op, ctx, 0))
                                       .str());
        })
        .Case<SetOp>([&](SetOp op) {
          lines.push_back(indent + llvm::formatv("{0}[{1} * steps + cycle] = {2};",
                                                 emitOperand(op, ctx, 0),
                                                 emitIntAttr(op, "offset"),
                                                 emitOperand(op, ctx, 1))
                                       .str());
        })
        .Default([&](Operation* op) -> std::string {
          llvm::errs() << "Found invalid op during codegen!\n";
          llvm::errs() << *op << "\n";
          throw std::runtime_error("invalid op");
        });
  }

  void emitOp(Operation* op, FileContext& ctx, list& lines, MixPowAnalysis& mixPows) {
    std::stringstream ss;
    const char* outType = "Fp";
    if (op->getNumResults() == 1) {
      auto valType = llvm::dyn_cast<ValType>(op->getResults()[0].getType());
      if (valType && valType.getFieldK() > 1) {
        outType = "FpExt";
      }
    }

    mlir::TypeSwitch<Operation*>(op)
        .Case<ConstOp>([&](ConstOp op) {
          if (op.getType().getFieldK() > 1)
            ss << "FpExt " << ctx.def(op.getOut()) << emitPolynomialAttr(op, "coefficients") << ";";
          else
            ss << "Fp " << ctx.def(op.getOut()) << "(" << emitPolynomialAttr(op, "coefficients")
               << ");";
        })
        .Case<MakeTemporaryBufferOp>([&](MakeTemporaryBufferOp op) {
          StringRef typeName;
          if (op.getType().getElement().getFieldK() > 1)
            typeName = "FpExt";
          else
            typeName = "Fp";
          ss << llvm::formatv(
                    "{0} {1}[{2}];\n", typeName, ctx.def(op.getOut()), op.getType().getSize())
                    .str();
        })
        .Case<func::CallOp>([&](func::CallOp op) {
          auto out = ctx.def(op.getResult(0));
          ss << llvm::formatv("auto {0} = {1}(idx, size", out, op.getCallee()).str();

          for (mlir::Value arg : op.getOperands()) {
            ss << ", " << ctx.use(arg);
          }
          ss << ");\n";
        })
        .Case<GetOp>([&](GetOp op) {
          auto buf = emitOperand(op, ctx, 0);
          auto reg = emitIntAttr(op, "offset");
          auto back = emitIntAttr(op, "back");
          ss << outType << " " << ctx.def(op.getOut()) << " = " << buf << "[" << reg
             << " * size + ((idx - INV_RATE * " << back << ") & mask)];";
        })
        .Case<SetGlobalOp>([&](SetGlobalOp op) {
          ss << llvm::formatv("{0}[{1}] = {2};",
                              emitOperand(op, ctx, 0),
                              emitIntAttr(op, "offset"),
                              emitOperand(op, ctx, 1))
                    .str();
        })
        .Case<GetGlobalOp>([&](GetGlobalOp op) {
          auto global = emitOperand(op, ctx, 0);
          auto reg = emitIntAttr(op, "offset");
          ss << outType << " " << ctx.def(op.getOut()) << " = " << global << "[" << reg << "];";
        })
        .Case<AddOp>([&](AddOp op) {
          ss << outType << " " << ctx.def(op.getOut()) << " = " << emitOperand(op, ctx, 0) << " + "
             << emitOperand(op, ctx, 1) << ";";
        })
        .Case<SubOp>([&](SubOp op) {
          ss << outType << " " << ctx.def(op.getOut()) << " = " << emitOperand(op, ctx, 0) << " - "
             << emitOperand(op, ctx, 1) << ";";
        })
        .Case<MulOp>([&](MulOp op) {
          ss << outType << " " << ctx.def(op.getOut()) << " = " << emitOperand(op, ctx, 0) << " * "
             << emitOperand(op, ctx, 1) << ";";
        })
        .Case<TrueOp>([&](TrueOp op) { ss << "FpExt " << ctx.def(op.getOut()) << " = FpExt(0);"; })
        .Case<AndEqzOp>([&](AndEqzOp op) {
          auto x = emitOperand(op, ctx, 0);
          auto val = emitOperand(op, ctx, 1);

          ss << "FpExt " << ctx.def(op.getOut()) << " = " << x << " + poly_mix["
             << mixPows.getMixPowIndex(op) << "] * " << val << ";";
        })
        .Case<AndCondOp>([&](AndCondOp op) {
          auto x = emitOperand(op, ctx, 0);
          auto cond = emitOperand(op, ctx, 1);
          auto inner = emitOperand(op, ctx, 2);
          ss << "FpExt " << ctx.def(op.getOut()) << " = " << x << " + " << cond << " * " << inner
             << " * poly_mix[" << mixPows.getMixPowIndex(op) << "];";
        })
        .Default([&](Operation* op) -> std::string {
          llvm::errs() << "Found invalid op during poly codegen!\n";
          llvm::errs() << *op << "\n";
          throw std::runtime_error("invalid op");
        });

    lines.push_back(ss.str());
  }

  std::string emitOperand(Operation* op, const FileContext& ctx, size_t idx) {
    return ctx.use(op->getOperand(idx));
  }

  std::string emitLoc(Operation* op) {
    if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc())) {
      return llvm::formatv("{0}:{1}", loc.getFilename().str(), loc.getLine()).str();
    }
    return "\"unknown\"";
  }

  std::string emitIntAttr(Operation* op, const char* attrName) {
    auto attr = op->getAttrOfType<IntegerAttr>(attrName);
    return std::to_string(attr.getUInt());
  }

  std::string emitPolynomialAttr(Operation* op, const char* attrName) {
    auto attr = op->getAttrOfType<PolynomialAttr>(attrName);
    if (attr.size() == 1) {
      return std::to_string(attr[0]);
    } else {
      std::string out;
      llvm::raw_string_ostream os(out);
      os << "{";
      llvm::interleaveComma(attr.asArrayRef(), os);
      os << "}";
      return out;
    }
  }

  mustache openTemplate(const std::string& path) {
    fs::path fs_path(path);
    if (!fs::exists(fs_path)) {
      throw std::runtime_error(llvm::formatv("File does not exist: {0}", path));
    }

    std::ifstream ifs(path);
    ifs.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    std::string str(std::istreambuf_iterator<char>{ifs}, {});
    mustache tmpl(str);
    tmpl.set_custom_escape([](const std::string& str) { return str; });
    return tmpl;
  }
};

} // namespace

std::unique_ptr<GpuStreamEmitter> createGpuStreamEmitter(llvm::raw_ostream& ofs,
                                                         const std::string& suffix) {
  return std::make_unique<GpuStreamEmitterImpl>(ofs, suffix);
}

} // namespace zirgen
