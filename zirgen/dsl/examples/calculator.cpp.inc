SET_FIELD(BabyBear);
constexpr size_t kRegCountAccum = 1;
constexpr size_t kRegCountCode = 1;
constexpr size_t kRegCountData = 6;
constexpr size_t kRegCountGlobal = 4;
constexpr size_t kRegCountMix = 4;
using Tap8Array = std::array<Tap, 8>;
struct NondetRegLayout {
  Reg _super;
};
using NondetRegLayout2LayoutArray = std::array<NondetRegLayout, 2>;
struct OneHotLayout {
  NondetRegLayout2LayoutArray _super;
};
struct TopResultLayout {
  NondetRegLayout _super;
  NondetRegLayout arm0;
  NondetRegLayout arm1;
};
struct TopLayout {
  NondetRegLayout op;
  NondetRegLayout left;
  NondetRegLayout right;
  OneHotLayout _0;
  TopResultLayout result;
};
struct _globalLayout {
  NondetRegLayout result;
};
struct NondetRegStruct {
  Val _super;
};
struct ComponentStruct {};
struct RegStruct {
  NondetRegStruct _super;
  NondetRegStruct reg;
};
struct GetValFromUserStruct {
  Val _super;
};
using Val2Array = std::array<Val, 2>;
using NondetRegStruct2Array = std::array<NondetRegStruct, 2>;
using ComponentStruct2Array = std::array<ComponentStruct, 2>;
struct OneHot__0_SuperStruct {
  Val _super;
};
using OneHot__0_SuperStruct2Array = std::array<OneHot__0_SuperStruct, 2>;
struct OneHotStruct {
  NondetRegStruct2Array _super;
  Val n;
  NondetRegStruct2Array bits;
};
struct TopStruct {
  ComponentStruct _super;
  NondetRegStruct op;
  NondetRegStruct left;
  NondetRegStruct right;
  RegStruct result;
};
using ExtVal8Array = std::array<ExtVal, 8>;
constexpr Tap8Array kTapList = Tap8Array{MAKE_TAP(0, 0, 0),
                                         MAKE_TAP(1, 0, 0),
                                         MAKE_TAP(2, 0, 0),
                                         MAKE_TAP(2, 1, 0),
                                         MAKE_TAP(2, 2, 0),
                                         MAKE_TAP(2, 3, 0),
                                         MAKE_TAP(2, 4, 0),
                                         MAKE_TAP(2, 5, 0)};
constexpr TopLayout kLayout_Top = TopLayout{
    .op = NondetRegLayout{._super = /*offset=*/0},
    .left = NondetRegLayout{._super = /*offset=*/1},
    .right = NondetRegLayout{._super = /*offset=*/2},
    ._0 = OneHotLayout{._super =
                           NondetRegLayout2LayoutArray{NondetRegLayout{._super = /*offset=*/3},
                                                       NondetRegLayout{._super = /*offset=*/4}}},
    .result = TopResultLayout{._super = NondetRegLayout{._super = /*offset=*/5},
                              .arm0 = NondetRegLayout{._super = /*offset=*/5},
                              .arm1 = NondetRegLayout{._super = /*offset=*/5}}};
constexpr _globalLayout kLayoutGlobal =
    _globalLayout{.result = NondetRegLayout{._super = /*offset=*/0}};
Val exec_Isz(ExecContext& ctx, Val arg0) {
  return isz(arg0);
}
Val exec_Add(ExecContext& ctx, Val arg0, Val arg1) {
  return (arg0 + arg1);
}
Val exec_Sub(ExecContext& ctx, Val arg0, Val arg1) {
  return (arg0 - arg1);
}
Val exec_Mul(ExecContext& ctx, Val arg0, Val arg1) {
  return (arg0 * arg1);
}
NondetRegStruct
back_NondetReg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  return NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout1, _super), distance0)};
}
NondetRegStruct exec_NondetReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE(x2, arg0);
  return NondetRegStruct{._super = LOAD(x2, 0)};
}
ComponentStruct exec_Component(ExecContext& ctx) {
  return ComponentStruct{};
}
RegStruct back_Reg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  // Reg(<preamble>:4)
  NondetRegStruct x2 = back_NondetReg(ctx, distance0, layout1);
  return RegStruct{._super = x2, .reg = x2};
}
RegStruct exec_Reg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  NondetRegStruct x2 = exec_NondetReg(ctx, arg0, layout1);
  // Reg(<preamble>:5)
  EQZ((arg0 - x2._super), "Reg(<preamble>:5)");
  return RegStruct{._super = x2, .reg = x2};
}
ComponentStruct exec_Log(ExecContext& ctx, std::string arg0, std::initializer_list<Val> arg1) {
  // Log(<preamble>:16)
  INVOKE_EXTERN(ctx, log, arg0, arg1);
  return ComponentStruct{};
}
GetValFromUserStruct exec_GetValFromUser(ExecContext& ctx) {
  // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
  Val x0 = INVOKE_EXTERN(ctx, getValFromUser);
  return GetValFromUserStruct{._super = x0};
}
ComponentStruct exec_OutputToUser(ExecContext& ctx, Val arg0) {
  // OutputToUser(zirgen/dsl/examples/calculator.zir:6)
  INVOKE_EXTERN(ctx, outputToUser, arg0);
  return ComponentStruct{};
}
OneHotStruct exec_OneHot(ExecContext& ctx, Val arg0, BoundLayout<OneHotLayout> layout1) {
  // OneHot(zirgen/dsl/examples/calculator.zir:13)
  Val2Array x2 = Val2Array{Val(0), Val(1)};
  NondetRegStruct2Array x3 =
      map(x2,
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val2Array::value_type x4,
                            BoundLayout<NondetRegLayout2LayoutArray::value_type> x5) {
            Val x6 = exec_Sub(ctx, x4, arg0);
            Val x7 = exec_Isz(ctx, x6);
            NondetRegStruct x8 = exec_NondetReg(ctx, x7, x5);
            return x8;
          }));
  // OneHot(zirgen/dsl/examples/calculator.zir:15)
  ComponentStruct2Array x9 = map(x3, std::function([&](NondetRegStruct2Array::value_type x10) {
                                   Val x11 = x10._super;
                                   Val x12 = exec_Sub(ctx, Val(1), x11);
                                   Val x13 = exec_Mul(ctx, x11, x12);
                                   EQZ(x13, "OneHot(zirgen/dsl/examples/calculator.zir:15)");
                                   ComponentStruct x14 = exec_Component(ctx);
                                   return x14;
                                 }));
  // OneHot(zirgen/dsl/examples/calculator.zir:17)
  Val x15 = reduce(x3, Val(0), std::function([&](Val x16, NondetRegStruct2Array::value_type x17) {
                     Val x18 = exec_Add(ctx, x16, x17._super);
                     return x18;
                   }));
  EQZ((x15 - Val(1)), "OneHot(zirgen/dsl/examples/calculator.zir:17)");
  // OneHot(zirgen/dsl/examples/calculator.zir:19)
  OneHot__0_SuperStruct2Array x19 = map(x2, std::function([&](Val2Array::value_type x20) {
                                          Val x21 = exec_Mul(ctx, x3[to_size_t(x20)]._super, x20);
                                          return OneHot__0_SuperStruct{._super = x21};
                                        }));
  Val x22 =
      reduce(x19, Val(0), std::function([&](Val x23, OneHot__0_SuperStruct2Array::value_type x24) {
               Val x25 = exec_Add(ctx, x23, x24._super);
               return x25;
             }));
  EQZ((x22 - arg0), "OneHot(zirgen/dsl/examples/calculator.zir:19)");
  return OneHotStruct{._super = x3, .n = Val(2), .bits = x3};
}
RegStruct exec_GetGlobalResult(ExecContext& ctx) {
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
  BoundLayout<_globalLayout> x0 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx, global));
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:24)
  RegStruct x1 = back_Reg(ctx, 0, LAYOUT_LOOKUP(x0, result));
  return x1;
}
ComponentStruct exec_SetGlobalResult(ExecContext& ctx, Val arg0) {
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  BoundLayout<_globalLayout> x1 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx, global));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
  RegStruct x2 = exec_Reg(ctx, arg0, LAYOUT_LOOKUP(x1, result));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  ComponentStruct x3 = exec_Component(ctx);
  return x3;
}
TopStruct exec_Top(ExecContext& ctx, BoundLayout<TopLayout> layout0) {
  // Top(zirgen/dsl/examples/calculator.zir:36)
  BoundLayout<TopResultLayout> x1 = LAYOUT_LOOKUP(layout0, result);
  // Top(zirgen/dsl/examples/calculator.zir:33)
  GetValFromUserStruct x2 = exec_GetValFromUser(ctx);
  NondetRegStruct x3 = exec_NondetReg(ctx, x2._super, LAYOUT_LOOKUP(layout0, op));
  // Top(zirgen/dsl/examples/calculator.zir:34)
  GetValFromUserStruct x4 = exec_GetValFromUser(ctx);
  NondetRegStruct x5 = exec_NondetReg(ctx, x4._super, LAYOUT_LOOKUP(layout0, left));
  // Top(zirgen/dsl/examples/calculator.zir:35)
  GetValFromUserStruct x6 = exec_GetValFromUser(ctx);
  NondetRegStruct x7 = exec_NondetReg(ctx, x6._super, LAYOUT_LOOKUP(layout0, right));
  // Top(zirgen/dsl/examples/calculator.zir:36)
  OneHotStruct x8 = exec_OneHot(ctx, x3._super, LAYOUT_LOOKUP(layout0, _0));
  Val2Array x9 = map(
      x8._super, std::function([&](NondetRegStruct2Array::value_type x10) { return x10._super; }));
  BoundLayout<NondetRegLayout> x11 = LAYOUT_LOOKUP(x1, _super);
  RegStruct x12;
  if (to_size_t(x9[0])) {
    // Top(zirgen/dsl/examples/calculator.zir:37)
    Val x13 = exec_Add(ctx, x5._super, x7._super);
    RegStruct x14 = exec_Reg(ctx, x13, LAYOUT_LOOKUP(x1, arm0));
    x12 = x14;
  } else if (to_size_t(x9[1])) {
    // Top(zirgen/dsl/examples/calculator.zir:38)
    Val x15 = exec_Sub(ctx, x5._super, x7._super);
    RegStruct x16 = exec_Reg(ctx, x15, LAYOUT_LOOKUP(x1, arm1));
    x12 = x16;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x17 = back_Reg(ctx, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:40)
  ComponentStruct x18 = exec_Log(ctx, "result = %u", std::initializer_list<Val>{x17._super._super});
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x19 = back_Reg(ctx, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:41)
  ComponentStruct x20 = exec_SetGlobalResult(ctx, x19._super._super);
  // Top(zirgen/dsl/examples/calculator.zir:42)
  RegStruct x21 = exec_GetGlobalResult(ctx);
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x22 = back_Reg(ctx, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:42)
  EQZ((x21._super._super - x22._super._super), "Top(zirgen/dsl/examples/calculator.zir:42)");
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x23 = back_Reg(ctx, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:43)
  ComponentStruct x24 = exec_OutputToUser(ctx, x23._super._super);
  // Top(zirgen/dsl/examples/calculator.zir:32)
  ComponentStruct x25 = exec_Component(ctx);
  return TopStruct{._super = x25, .op = x3, .left = x5, .right = x7, .result = x17};
}
void step_Top(ExecContext& ctx) {
  BoundLayout<TopLayout> x0 = BIND_LAYOUT(kLayout_Top, GET_BUFFER(ctx, data));
  TopStruct x1 = exec_Top(ctx, x0);
  return;
}
MixState validityTaps_(ValidityTapsContext& ctx, PolyMix polyMix0, ExtVal8Array taps1) {
  // Top(zirgen/dsl/examples/calculator.zir:36)
  MixState x2 = trivialConstraint();
  // OneHot(zirgen/dsl/examples/calculator.zir:19)
  MixState x3 = andEqzExt(
      polyMix0,
      andEqzExt(polyMix0,
                andEqzExt(polyMix0,
                          andEqzExt(polyMix0, x2, (taps1[5] * (Val4{1, 0, 0, 0} - taps1[5]))),
                          (taps1[6] * (Val4{1, 0, 0, 0} - taps1[6]))),
                ((taps1[5] + taps1[6]) - Val4{1, 0, 0, 0})),
      (taps1[6] - taps1[2]));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  // Top(zirgen/dsl/examples/calculator.zir:41)
  BoundLayout<_globalLayout> x4 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx, global));
  // Top(zirgen/dsl/examples/calculator.zir:42)
  MixState x5 = andEqzExt(
      polyMix0,
      andEqzExt(
          polyMix0,
          andCondExt(
              andCondExt(x3, taps1[5], andEqzExt(polyMix0, x2, ((taps1[3] + taps1[4]) - taps1[7]))),
              taps1[6],
              andEqzExt(polyMix0, x2, ((taps1[3] - taps1[4]) - taps1[7]))),
          (taps1[7] - LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, result), _super), 0))),
      (LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, result), _super), 0) - taps1[7]));
  return x5;
}
MixState validityRegs_(ValidityRegsContext& ctx, PolyMix polyMix0) {
  BoundLayout<TopLayout> x1 = BIND_LAYOUT(kLayout_Top, GET_BUFFER(ctx, data));
  // Top(zirgen/dsl/examples/calculator.zir:36)
  BoundLayout<TopResultLayout> x2 = LAYOUT_LOOKUP(x1, result);
  // OneHot(zirgen/dsl/examples/calculator.zir:13)
  BoundLayout<NondetRegLayout2LayoutArray> x3 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, _0), _super);
  // OneHot(zirgen/dsl/examples/calculator.zir:15)
  MixState x4 = andEqz(polyMix0,
                       trivialConstraint(),
                       (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 0), _super), 0) *
                        (Val(1) - LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 0), _super), 0))));
  // OneHot(zirgen/dsl/examples/calculator.zir:17)
  MixState x5 = andEqz(polyMix0,
                       andEqz(polyMix0,
                              x4,
                              (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 1), _super), 0) *
                               (Val(1) - LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 1), _super), 0)))),
                       ((LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 0), _super), 0) +
                         LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 1), _super), 0)) -
                        Val(1)));
  BoundLayout<NondetRegLayout> x6 = LAYOUT_LOOKUP(x2, _super);
  // Reg(<preamble>:5)
  // Top(zirgen/dsl/examples/calculator.zir:37)
  MixState x7 = andEqz(polyMix0,
                       trivialConstraint(),
                       ((LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, left), _super), 0) +
                         LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, right), _super), 0)) -
                        LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x2, arm0), _super), 0)));
  // Top(zirgen/dsl/examples/calculator.zir:38)
  MixState x8 = andEqz(polyMix0,
                       trivialConstraint(),
                       ((LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, left), _super), 0) -
                         LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, right), _super), 0)) -
                        LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x2, arm1), _super), 0)));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  // Top(zirgen/dsl/examples/calculator.zir:41)
  BoundLayout<_globalLayout> x9 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx, global));
  // Reg(<preamble>:5)
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
  MixState x10 =
      andEqz(polyMix0,
             andCond(andCond(andEqz(polyMix0,
                                    x5,
                                    (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 1), _super), 0) -
                                     LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x1, op), _super), 0))),
                             LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 0), _super), 0),
                             x7),
                     LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x3, 1), _super), 0),
                     x8),
             (LOAD(LAYOUT_LOOKUP(x6, _super), 0) -
              LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, result), _super), 0)));
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
  // Top(zirgen/dsl/examples/calculator.zir:42)
  BoundLayout<_globalLayout> x11 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx, global));
  return andEqz(polyMix0,
                x10,
                (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, result), _super), 0) -
                 LOAD(LAYOUT_LOOKUP(x6, _super), 0)));
}
