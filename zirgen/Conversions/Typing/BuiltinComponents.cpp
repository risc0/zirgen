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

#include "zirgen/Conversions/Typing/BuiltinComponents.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"

using namespace mlir;
using namespace zirgen::Zhl;
namespace zirgen::Typing {

namespace {

struct Builtins {
  Builtins(OpBuilder& builder)
      : builder(builder)
      , ctx(builder.getContext())
      , valType(Zhlt::getValType(ctx))
      , loc(builder.getUnknownLoc()) {}

  void addBuiltins();
  void genTrivial(StringRef name, Type type);
  void genArray(StringRef mangledName,
                ZStruct::ArrayType arrayType,
                ZStruct::LayoutArrayType layoutType,
                TypeRange ctorParams);

private:
  void makeBuiltin(StringRef name,
                   Type valueType,
                   TypeRange constructParams,
                   Type layoutType,
                   const std::function<void(/*args=*/ValueRange)>& buildBody);
  template <typename OpT> void makeBinValOp(StringRef name);
  template <typename OpT> void makeUnaryValOp(StringRef name);
  template <typename OpT> void makeSpecialBuiltin(StringRef name);
  void genNondetReg();
  void genComponent();
  void genInRange();

  OpBuilder& builder;
  MLIRContext* ctx;

  Type valType;
  Location loc;
};

void Builtins::makeBuiltin(StringRef name,
                           Type valueType,
                           TypeRange constructParams,
                           Type layoutType,
                           const std::function<void(ValueRange)>& buildBody) {
  auto op = builder.create<Zhlt::ComponentOp>(
      builder.getUnknownLoc(), name, valueType, constructParams, layoutType);

  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(op.addEntryBlock());
  buildBody(op.getArguments());
}

template <typename OpT> void Builtins::makeBinValOp(StringRef name) {
  makeBuiltin(name,
              /*valueType=*/valType,
              /*constructParams=*/{valType, valType},
              /*layout=*/Type(),
              [&](ValueRange args) {
                auto op = builder.create<OpT>(loc, args[0], args[1]);
                builder.create<Zhlt::ReturnOp>(loc, op);
              });
}

template <typename OpT> void Builtins::makeUnaryValOp(StringRef name) {
  makeBuiltin(name,
              /*valueType=*/valType,
              /*constructParams=*/{valType},
              /*layout=*/Type(),
              [&](ValueRange args) {
                auto op = builder.create<OpT>(loc, args[0]);
                builder.create<Zhlt::ReturnOp>(loc, op);
              });
}

void Builtins::genNondetReg() {
  auto returnType = Zhlt::getNondetRegType(builder.getContext());
  auto refType = Zhlt::getNondetRegLayoutType(builder.getContext());
  makeBuiltin("NondetReg",
              /*valueType=*/returnType,
              /*constructParams=*/{valType},
              /*layout=*/refType,
              [&](ValueRange args) {
                Value val = args[0];
                Value ref = builder.create<ZStruct::LookupOp>(loc, args[1], "@super");
                builder.create<ZStruct::StoreOp>(loc, ref, val);
                Value zero = builder.create<arith::ConstantOp>(
                    loc, builder.getIndexType(), builder.getIndexAttr(0));
                mlir::Value loaded =
                    builder.create<ZStruct::LoadOp>(loc, valType, ref, /*distance=*/zero);
                mlir::Value packed = builder.create<ZStruct::PackOp>(
                    loc, Zhlt::getNondetRegType(ctx), /*members=*/ValueRange{loaded});
                builder.create<Zhlt::ReturnOp>(loc, packed);
              });
}

void Builtins::genComponent() {
  makeBuiltin("Component",
              /*valueType=*/Zhlt::getComponentType(ctx),
              /*constructParams=*/ValueRange{},
              /*layout=*/Type(),
              [&](ValueRange args) {
                auto packed = builder.create<ZStruct::PackOp>(
                    loc, Zhlt::getComponentType(ctx), /*members=*/ValueRange{});
                builder.create<Zhlt::ReturnOp>(loc, packed);
              });
}

void Builtins::genInRange() {
  auto val = Zhlt::getValType(ctx);
  makeBuiltin("InRange",
              /*valueType=*/val,
              /*constructParams*/ {val, val, val},
              /*layout=*/Type(),
              [&](ValueRange args) {
                auto op = builder.create<Zll::InRangeOp>(loc, args[0], args[1], args[2]);
                builder.create<Zhlt::ReturnOp>(loc, op);
              });
}

void Builtins::genTrivial(StringRef name, Type type) {
  makeBuiltin(name,
              /*valueType=*/type,
              /*constructParams=*/{type},
              /*layout=*/Type(),
              [&](ValueRange args) { builder.create<Zhlt::ReturnOp>(loc, args[0]); });
}

template <typename OpT> void Builtins::makeSpecialBuiltin(StringRef name) {
  builder.create<OpT>(builder.getUnknownLoc(), name);
}

void Builtins::addBuiltins() {
  makeUnaryValOp<Zll::InvOp>("Inv");
  makeUnaryValOp<Zll::IsZeroOp>("Isz");
  makeUnaryValOp<Zll::NegOp>("Neg");

  makeBinValOp<Zll::BitAndOp>("BitAnd");
  makeBinValOp<Zll::AddOp>("Add");
  makeBinValOp<Zll::SubOp>("Sub");
  makeBinValOp<Zll::MulOp>("Mul");

  genNondetReg();
  genComponent();
  genInRange();

  genTrivial("Type", Zhlt::getTypeType(ctx));
  genTrivial("Val", valType);
  genTrivial("String", Zhlt::getStringType(ctx));

  makeSpecialBuiltin<Zhlt::BuiltinArrayOp>("Array");
  makeSpecialBuiltin<Zhlt::BuiltinArrayOp>("ConcatArray");
  makeSpecialBuiltin<Zhlt::BuiltinLogOp>("Log");
}

// Builtins that are defined using the DSL.
static llvm::StringLiteral zirPreamble = R"(

component Reg(v: Val) {
   reg := NondetReg(v);
   v = reg;
   reg
}

function Div(lhs: Val, rhs: Val) {
   reciprocal := Inv(rhs);
   reciprocal * rhs = 1;

   reciprocal * lhs
}

)";

} // namespace

void addBuiltins(OpBuilder& builder) {
  Builtins builtins(builder);
  builtins.addBuiltins();
}

StringRef getBuiltinPreamble() {
  return zirPreamble;
}

} // namespace zirgen::Typing
