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
      , extValType(Zhlt::getExtValType(ctx))
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
  template <typename OpT> void makeBinExtValOp(StringRef name);
  template <typename OpT> void makeUnaryValOp(StringRef name);
  template <typename OpT> void makeUnaryExtValOp(StringRef name);
  void genNondetReg();
  void genNondetExtReg();
  void genMakeExt();
  void genEqzExt();
  void genComponent();
  void genInRange();

  OpBuilder& builder;
  MLIRContext* ctx;

  Zll::ValType valType;
  Zll::ValType extValType;
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

template <typename OpT> void Builtins::makeBinExtValOp(StringRef name) {
  makeBuiltin(name,
              /*valueType=*/extValType,
              /*constructParams=*/{extValType, extValType},
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

template <typename OpT> void Builtins::makeUnaryExtValOp(StringRef name) {
  makeBuiltin(name,
              /*valueType=*/extValType,
              /*constructParams=*/{extValType},
              /*layout=*/Type(),
              [&](ValueRange args) {
                auto op = builder.create<OpT>(loc, args[0]);
                builder.create<Zhlt::ReturnOp>(loc, op);
              });
}

void Builtins::genNondetReg() {
  auto returnType = Zhlt::getNondetRegType(builder.getContext());
  auto refType = Zhlt::getNondetRegLayoutType(builder.getContext());
  makeBuiltin(
      "NondetReg",
      /*valueType=*/returnType,
      /*constructParams=*/{valType},
      /*layout=*/refType,
      [&](ValueRange args) {
        Value val = args[0];
        Value ref = builder.create<ZStruct::LookupOp>(loc, args[1], "@super");
        builder.create<ZStruct::StoreOp>(loc, ref, val);
        Value zero =
            builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
        mlir::Value loaded = builder.create<ZStruct::LoadOp>(loc, valType, ref, /*distance=*/zero);
        mlir::Value packed = builder.create<ZStruct::PackOp>(
            loc, Zhlt::getNondetRegType(ctx), /*members=*/ValueRange{loaded, /*layout=*/args[1]});
        builder.create<Zhlt::ReturnOp>(loc, packed);
      });
}

void Builtins::genNondetExtReg() {
  auto returnType = Zhlt::getNondetExtRegType(builder.getContext());
  auto refType = Zhlt::getNondetExtRegLayoutType(builder.getContext());
  makeBuiltin("NondetExtReg",
              /*valueType=*/returnType,
              /*constructParams=*/{extValType},
              /*layout=*/refType,
              [&](ValueRange args) {
                Value val = args[0];
                Value ref = builder.create<ZStruct::LookupOp>(loc, args[1], "@super");
                builder.create<ZStruct::StoreOp>(loc, ref, val);
                Value zero = builder.create<arith::ConstantOp>(
                    loc, builder.getIndexType(), builder.getIndexAttr(0));
                mlir::Value loaded =
                    builder.create<ZStruct::LoadOp>(loc, extValType, ref, /*distance=*/zero);
                mlir::Value packed = builder.create<ZStruct::PackOp>(
                    loc, Zhlt::getNondetExtRegType(ctx), /*members=*/ValueRange{loaded});
                builder.create<Zhlt::ReturnOp>(loc, packed);
              });
}

void Builtins::genMakeExt() {
  makeBuiltin("MakeExt",
              /*valueType=*/extValType,
              /*constructParams=*/{valType},
              /*layout=*/Type(),
              [&](ValueRange args) {
                // Convert from Fp -> FpExt by adding an Ext of 0
                Value val = args[0];
                SmallVector<uint64_t> zero(extValType.getFieldK(), 0);
                Value zConst = builder.create<Zll::ConstOp>(loc, extValType, zero);
                Value result = builder.create<Zll::AddOp>(loc, val, zConst);
                builder.create<Zhlt::ReturnOp>(loc, result);
              });
}

void Builtins::genEqzExt() {
  makeBuiltin("EqzExt",
              /*valueType=*/Zhlt::getComponentType(ctx),
              /*constructParams=*/{extValType},
              /*layout=*/Type(),
              [&](ValueRange args) {
                builder.create<Zll::EqualZeroOp>(loc, args[0]);
                auto packed = builder.create<ZStruct::PackOp>(
                    loc, Zhlt::getComponentType(ctx), /*members=*/ValueRange{});
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

void Builtins::genArray(StringRef mangledName,
                        ZStruct::ArrayType arrayType,
                        ZStruct::LayoutArrayType layoutType,
                        TypeRange ctorParams) {
  auto build = [&](ValueRange args) {
    Type elemType = arrayType.getElement();
    SmallVector<Value> elements;
    for (size_t i = 0; i < arrayType.getSize(); i++) {
      Value idx = builder.create<Zll::ConstOp>(loc, i);
      ValueRange ctorParamArgs = args;
      Value elemLayout;
      if (layoutType) {
        ctorParamArgs = args.drop_back();
        elemLayout = builder.create<ZStruct::SubscriptOp>(loc, args.back(), idx);
      }
      Value element = builder.create<Zhlt::ConstructOp>(loc,
                                                        Zhlt::mangledTypeName(elemType),
                                                        elemType,
                                                        /*constructParams=*/ctorParamArgs,
                                                        /*layout=*/elemLayout);
      elements.push_back(element);
    }
    Value array = builder.create<ZStruct::ArrayOp>(loc, elements);
    builder.create<Zhlt::ReturnOp>(loc, array);
  };
  makeBuiltin(mangledName,
              /*valueType=*/arrayType,
              /*constructParams=*/ctorParams,
              /*layout=*/layoutType,
              /*buildBody=*/build);
}

void Builtins::addBuiltins() {
  makeUnaryValOp<Zll::InvOp>("Inv");
  makeUnaryValOp<Zll::IsZeroOp>("Isz");
  makeUnaryValOp<Zll::NegOp>("Neg");

  makeBinValOp<Zll::BitAndOp>("BitAnd");
  makeBinValOp<Zll::ModOp>("Mod");
  makeBinValOp<Zll::AddOp>("Add");
  makeBinValOp<Zll::SubOp>("Sub");
  makeBinValOp<Zll::MulOp>("Mul");

  makeUnaryExtValOp<Zll::InvOp>("ExtInv");

  makeBinExtValOp<Zll::AddOp>("ExtAdd");
  makeBinExtValOp<Zll::SubOp>("ExtSub");
  makeBinExtValOp<Zll::MulOp>("ExtMul");

  genNondetReg();
  genNondetExtReg();
  genMakeExt();
  genEqzExt();
  genComponent();
  genInRange();

  genTrivial("Type", Zhlt::getTypeType(ctx));
  genTrivial("Val", valType);
  genTrivial("ExtVal", extValType);
  genTrivial("String", Zhlt::getStringType(ctx));
}

// Builtins that are defined using the DSL.
static llvm::StringLiteral zirPreamble = R"(

component Reg(v: Val) {
   reg := NondetReg(v);
   v = reg;
   reg
}

component ExtReg(v: ExtVal) {
   reg := NondetExtReg(v);
   EqzExt(ExtSub(reg, v));
   reg
}

function Div(lhs: Val, rhs: Val) {
   reciprocal := Inv(rhs);
   reciprocal * rhs = 1;

   reciprocal * lhs
}

extern Log(message: String, vals: Val...);
extern Abort();
extern Assert(x: Val, message: String);

)";

} // namespace

void addBuiltins(OpBuilder& builder) {
  Builtins builtins(builder);
  builtins.addBuiltins();
}

StringRef getBuiltinPreamble() {
  return zirPreamble;
}

void addArrayCtor(OpBuilder& builder,
                  StringRef mangledName,
                  ZStruct::ArrayType arrayType,
                  ZStruct::LayoutArrayType layoutType,
                  TypeRange ctorParams) {
  Builtins builtins(builder);
  builtins.genArray(mangledName, arrayType, layoutType, ctorParams);
}

} // namespace zirgen::Typing
