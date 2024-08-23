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

#include "zirgen/compiler/codegen/codegen.h"

#include <filesystem>
#include <fstream>

#include "mlir/Support/DebugStringHelper.h"
#include "mustache.h"
#include "zirgen/Dialect/Zll/Analysis/MixPowerAnalysis.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/include/llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace kainjow::mustache;
using namespace zirgen::Zll;

namespace fs = std::filesystem;

namespace zirgen {

namespace {

enum class FuncKind {
  Step,
  PolyFp,
  PolyExt,
};

struct Vars {
  llvm::DenseMap<Value, size_t> vars;

  size_t use(Value value) const {
    auto it = vars.find(value);
    if (it == vars.end()) {
      throw std::runtime_error("Missing use");
    }
    return it->second;
  }

  size_t var(Value value) {
    size_t id = vars.size();
    vars[value] = id;
    return id;
  }
};

struct PolyContext {
  Vars args;
  Vars fp;
  Vars mix;
};

class RustStreamEmitterImpl : public RustStreamEmitter {
  llvm::raw_ostream& ofs;

public:
  RustStreamEmitterImpl(llvm::raw_ostream& ofs) : ofs(ofs) {}
  ~RustStreamEmitterImpl() = default;

  void emitStepFunc(const std::string& name, func::FuncOp func) override {
    mustache tmpl;
    // TODO: remove this hack once host-side recursion code is unified with rv32im.
    bool isRecursion = func.getName() == "recursion";
    if (isRecursion) {
      tmpl = openTemplate("zirgen/compiler/codegen/cpp/recursion/step.tmpl.cpp");
    } else {
      tmpl = openTemplate("zirgen/compiler/codegen/cpp/step.tmpl.cpp");
    }

    FileContext ctx;
    ctx.vars[func.getArgument(0)] = "args[0]";
    ctx.vars[func.getArgument(1)] = "args[1]";
    ctx.vars[func.getArgument(2)] = "args[2]";
    ctx.vars[func.getArgument(3)] = "args[3]";
    ctx.vars[func.getArgument(4)] = "args[4]";

    list lines;
    emitStepBlock(func.front(), ctx, lines, /*depth=*/0, isRecursion);
    Value retVal = func.front().getTerminator()->getOperand(0);
    lines.push_back(llvm::formatv("return {0};", ctx.use(retVal)).str());

    tmpl.render(
        object{
            {"name", func.getName().str()},
            {"fn", "step_" + name},
            {"body", lines},
        },
        ofs);
  }

  void emitPolyFunc(const std::string& fn, func::FuncOp func) override {
    MixPowAnalysis mixPows(func);

    mustache tmpl = openTemplate("zirgen/compiler/codegen/cpp/poly.tmpl.cpp");

    FileContext ctx;
    ctx.vars[func.getArgument(0)] = "args[0]";
    ctx.vars[func.getArgument(1)] = "args[1]";
    ctx.vars[func.getArgument(2)] = "args[2]";
    ctx.vars[func.getArgument(3)] = "args[3]";
    ctx.vars[func.getArgument(4)] = "args[4]";

    list lines;
    for (Operation& op : func.front().without_terminator()) {
      emitOperation(&op, ctx, lines, /*depth=*/0, "Fp", FuncKind::PolyFp, &mixPows);
    }
    Value retVal = func.front().getTerminator()->getOperand(0);
    lines.push_back(llvm::formatv("return {0};", ctx.use(retVal)).str());

    tmpl.render(
        object{
            {"name", func.getName().str()},
            {"fn", fn},
            {"body", lines},
        },
        ofs);
  }

private:
  void emitStepBlock(Block& block, FileContext& ctx, list& lines, size_t depth, bool isRecursion) {
    std::string indent(depth * 2, ' ');
    for (Operation& op : block.without_terminator()) {
      mlir::TypeSwitch<Operation*>(&op)
          .Case<NondetOp>([&](NondetOp op) {
            lines.push_back(indent + "{");
            emitStepBlock(op.getInner().front(), ctx, lines, depth + 1, isRecursion);
            lines.push_back(indent + "}");
          })
          .Case<IfOp>([&](IfOp op) {
            lines.push_back(indent +
                            llvm::formatv("if ({0} != 0) {{", ctx.use(op.getCond())).str());
            emitStepBlock(op.getInner().front(), ctx, lines, depth + 1, isRecursion);
            lines.push_back(indent + "}");
          })
          .Case<ExternOp>([&](ExternOp op) {
            // TODO: remove this hack once host-side recursion code is unified with rv32im.
            if (isRecursion) {
              emitExternRecursion(op, ctx, lines, depth);
            } else {
              emitExtern(op, ctx, lines, depth);
            }
          })
          .Default(
              [&](Operation* op) { emitOperation(op, ctx, lines, depth, "Fp", FuncKind::Step); });
    }
  }

  // TODO: remove this hack once host-side recursion code is unified with rv32im.
  void emitExternRecursion(ExternOp op, FileContext& ctx, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');
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

  void emitExtern(ExternOp op, FileContext& ctx, list& lines, size_t depth) {
    std::string indent(depth * 2, ' ');
    std::stringstream ss;

    if (op.getOut().size()) {
      ss << "auto ";
    }
    if (op.getOut().size() > 1) {
      ss << "[";
    }

    for (size_t i = 0; i < op.getOut().size(); i++) {
      if (i) {
        ss << ", ";
      }
      ss << ctx.def(op.getResult(i));
    }

    if (op.getOut().size() > 1) {
      ss << "]";
    }
    if (op.getOut().size()) {
      ss << " = ";
    }

    ss << "extern_";
    ss << op.getName();
    if (op.getName().starts_with("plonkRead") || op.getName().starts_with("plonkWrite")) {
      ss << "_" << op.getExtra().str();
    }

    ss << "(ctx, cycle, " << escapeString(op.getExtra()) << ", {";
    for (size_t i = 0; i < op.getIn().size(); i++) {
      if (i) {
        ss << ", ";
      }
      ss << ctx.use(op.getOperand(i));
    }
    ss << "});";

    lines.push_back(indent + ss.str());
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
          lines.push_back(inner + llvm::formatv("auto& reg = {0}[{1} * steps + cycle];",
                                                ctx.use(op->getOperand(0)),
                                                emitIntAttr(op, "offset"))
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
          lines.push_back(
              indent +
              llvm::formatv("if ({0} != 0) throw std::runtime_error(\"eqz failed at: {1}\");",
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

  std::string emitLoc(Operation* op) {
    if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>()) {
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
    assert(attr.size() == 1 && "not yet unsupported");
    return std::to_string(attr[0]);
  }

  std::string emitOperand(Operation* op, Vars& vars, size_t idx) {
    return std::to_string(vars.use(op->getOperand(idx)));
  }

  std::string emitPolyExtBlock(Block& block, PolyContext& ctx) {
    std::stringstream ss;
    ss << "&[";
    for (Operation& origOp : block.without_terminator()) {
      mlir::TypeSwitch<Operation*>(&origOp)
          .Case<ConstOp>([&](ConstOp op) {
            ss << "PolyExtStep::Const(" << emitPolynomialAttr(op, "coefficients") << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<GetOp>([&](GetOp op) {
            ss << "PolyExtStep::Get(" << emitIntAttr(op, "tap") << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<GetGlobalOp>([&](GetGlobalOp op) {
            ss << "PolyExtStep::GetGlobal(" << emitOperand(op, ctx.args, 0) << ", "
               << emitIntAttr(op, "offset") << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<AddOp>([&](AddOp op) {
            ss << "PolyExtStep::Add(" << emitOperand(op, ctx.fp, 0) << ", "
               << emitOperand(op, ctx.fp, 1) << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<SubOp>([&](SubOp op) {
            ss << "PolyExtStep::Sub(" << emitOperand(op, ctx.fp, 0) << ", "
               << emitOperand(op, ctx.fp, 1) << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<MulOp>([&](MulOp op) {
            ss << "PolyExtStep::Mul(" << emitOperand(op, ctx.fp, 0) << ", "
               << emitOperand(op, ctx.fp, 1) << ")";
            ctx.fp.var(op.getOut());
          })
          .Case<TrueOp>([&](TrueOp op) {
            ss << "PolyExtStep::True";
            ctx.mix.var(op.getOut());
          })
          .Case<AndEqzOp>([&](AndEqzOp op) {
            ss << "PolyExtStep::AndEqz(" << emitOperand(op, ctx.mix, 0) << ", "
               << emitOperand(op, ctx.fp, 1) << ")";
            ctx.mix.var(op.getOut());
          })
          .Case<AndCondOp>([&](AndCondOp op) {
            ss << "PolyExtStep::AndCond(" << emitOperand(op, ctx.mix, 0) << ", "
               << emitOperand(op, ctx.fp, 1) << ", " << emitOperand(op, ctx.mix, 2) << ")";
            ctx.mix.var(op.getOut());
          });
      ss << ", // " << getLocString(origOp.getLoc()) << "\n";
    }
    ss << "]";
    return ss.str();
  }

public:
  void emitPolyExtFunc(func::FuncOp func) override {
    mustache tmpl = openTemplate("zirgen/compiler/codegen/rust/poly_ext_def.tmpl.rs");

    PolyContext ctx;
    ctx.args.vars[func.getArgument(1)] = 0;
    ctx.args.vars[func.getArgument(3)] = 1;

    std::string block = emitPolyExtBlock(func.front(), ctx);
    Value ret = func.front().getTerminator()->getOperand(0);

    tmpl.render(
        object{
            {"block", block},
            {"ret", std::to_string(ctx.mix.use(ret))},
        },
        ofs);
  }

  void emitTaps(func::FuncOp func) override {
    mustache tmpl = openTemplate("zirgen/compiler/codegen/rust/taps.tmpl.rs");

    auto regsAttr = func->getAttrOfType<ArrayAttr>("tapRegs");
    auto combosAttr = func->getAttrOfType<ArrayAttr>("tapCombos");

    // Flatten regs to taps, tracking start offsets of each reg group.
    list taps;
    list groupBegin;
    for (auto reg : regsAttr.getAsRange<TapRegAttr>()) {
      while (groupBegin.size() <= reg.getRegGroupId()) {
        groupBegin.push_back(std::to_string(taps.size()));
      }
      for (uint32_t back : reg.getBacks()) {
        taps.push_back(object{{"group", std::to_string(reg.getRegGroupId())},
                              {"offset", std::to_string(reg.getOffset())},
                              {"back", std::to_string(back)},
                              {"combo", std::to_string(reg.getComboId())},
                              {"skip", std::to_string(reg.getBacks().size())}});
      }
    }
    groupBegin.push_back(std::to_string(taps.size()));

    // Flatten combos, tracking start offset of each combo.
    list comboBacks;
    list comboBegin;

    for (auto backs : combosAttr.getAsRange<ArrayAttr>()) {
      comboBegin.push_back(std::to_string(comboBacks.size()));
      for (auto back : backs.getAsRange<IntegerAttr>()) {
        comboBacks.push_back(std::to_string(back.getUInt()));
      }
    }
    comboBegin.push_back(std::to_string(comboBacks.size()));

    tmpl.render(object{{"taps", taps},
                       {"combo_taps", comboBacks},
                       {"combo_begin", comboBegin},
                       {"group_begin", groupBegin},
                       {"combos_count", std::to_string(combosAttr.size())},
                       {"reg_count", std::to_string(regsAttr.size())},
                       {"tot_combo_backs", std::to_string(comboBacks.size())}},
                ofs);
  }

  void emitInfo(func::FuncOp func, ProtocolInfo info) override {
    mustache tmpl = openTemplate("zirgen/compiler/codegen/rust/info.tmpl.rs");

    size_t out_size = func.getArgument(1).getType().dyn_cast<BufferType>().getSize();
    size_t mix_size = func.getArgument(3).getType().dyn_cast<BufferType>().getSize();

    MixPowAnalysis mixPows(func);
    list poly_mix_powers;
    llvm::append_range(poly_mix_powers, llvm::map_range(mixPows.getPowersNeeded(), [](auto val) {
                         return std::to_string(val);
                       }));
    size_t num_poly_mix_powers = poly_mix_powers.size();

    tmpl.render(
        object{
            {"circuit_info", std::string(info.data())},
            {"output_size", std::to_string(out_size)},
            {"mix_size", std::to_string(mix_size)},
            {"num_poly_mix_powers", std::to_string(num_poly_mix_powers)},
            {"poly_mix_powers", poly_mix_powers},
        },
        ofs);
  }

private:
  mustache openTemplate(const std::string& path) {
    fs::path fs_path(path);
    if (!fs::exists(fs_path)) {
      if (fs::exists("../" + path)) {
        // Some lit tests put us in the "zirgen" subdirectory, so try up
        // one level.  TODO: Get rid of this lit test directory
        // confusion
        fs_path = fs::path("../" + path);
      } else
        throw std::runtime_error(llvm::formatv("File does not exist: {0}", path));
    }

    std::ifstream ifs(fs_path);
    ifs.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    std::string str(std::istreambuf_iterator<char>{ifs}, {});
    mustache tmpl(str);
    tmpl.set_custom_escape([](const std::string& str) { return str; });
    return tmpl;
  }
};

} // namespace

std::unique_ptr<RustStreamEmitter> createRustStreamEmitter(llvm::raw_ostream& ofs) {
  return std::make_unique<RustStreamEmitterImpl>(ofs);
}

} // namespace zirgen
