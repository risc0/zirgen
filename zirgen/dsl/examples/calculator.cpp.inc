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
    .op = NondetRegLayout{._super = MAKE_REF(0)},
    .left = NondetRegLayout{._super = MAKE_REF(1)},
    .right = NondetRegLayout{._super = MAKE_REF(2)},
    ._0 =
        OneHotLayout{._super = NondetRegLayout2LayoutArray{NondetRegLayout{._super = MAKE_REF(3)},
                                                           NondetRegLayout{._super = MAKE_REF(4)}}},
    .result = TopResultLayout{._super = NondetRegLayout{._super = MAKE_REF(5)},
                              .arm0 = NondetRegLayout{._super = MAKE_REF(5)},
                              .arm1 = NondetRegLayout{._super = MAKE_REF(5)}}};
constexpr _globalLayout kLayoutGlobal =
    _globalLayout{.result = NondetRegLayout{._super = MAKE_REF(0)}};
Val exec_Isz(ExecContext ctx0, Val arg1) {
  return isz(arg1);
}
Val exec_Add(ExecContext ctx0, Val arg1, Val arg2) {
  return (arg1 + arg2);
}
Val exec_Sub(ExecContext ctx0, Val arg1, Val arg2) {
  return (arg1 - arg2);
}
Val exec_Mul(ExecContext ctx0, Val arg1, Val arg2) {
  return (arg1 * arg2);
}
NondetRegStruct
back_NondetReg(ExecContext ctx0, Index distance1, BoundLayout<NondetRegLayout> layout2) {
  return NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout2, _super), distance1)};
}
NondetRegStruct exec_NondetReg(ExecContext ctx0, Val arg1, BoundLayout<NondetRegLayout> layout2) {
  BoundLayout<Reg> x3 = LAYOUT_LOOKUP(layout2, _super);
  STORE(x3, arg1);
  return NondetRegStruct{._super = LOAD(x3, 0)};
}
ComponentStruct exec_Component(ExecContext ctx0) {
  return ComponentStruct{};
}
RegStruct back_Reg(ExecContext ctx0, Index distance1, BoundLayout<NondetRegLayout> layout2) {
  // Reg(<preamble>:4)
  NondetRegStruct x3 = back_NondetReg(ctx0, distance1, layout2);
  return RegStruct{._super = x3, .reg = x3};
}
RegStruct exec_Reg(ExecContext ctx0, Val arg1, BoundLayout<NondetRegLayout> layout2) {
  NondetRegStruct x3 = exec_NondetReg(ctx0, arg1, layout2);
  // Reg(<preamble>:5)
  EQZ((arg1 - x3._super), "Reg(<preamble>:5)");
  return RegStruct{._super = x3, .reg = x3};
}
ComponentStruct exec_Log(ExecContext ctx0, std::string arg1, std::initializer_list<Val> arg2) {
  // Log(<preamble>:16)
  INVOKE_EXTERN(ctx0, log, arg1, arg2);
  return ComponentStruct{};
}
GetValFromUserStruct exec_GetValFromUser(ExecContext ctx0) {
  // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
  Val x1 = INVOKE_EXTERN(ctx0, getValFromUser);
  return GetValFromUserStruct{._super = x1};
}
ComponentStruct exec_OutputToUser(ExecContext ctx0, Val arg1) {
  // OutputToUser(zirgen/dsl/examples/calculator.zir:6)
  INVOKE_EXTERN(ctx0, outputToUser, arg1);
  return ComponentStruct{};
}
OneHotStruct exec_OneHot(ExecContext ctx0, Val arg1, BoundLayout<OneHotLayout> layout2) {
  // OneHot(zirgen/dsl/examples/calculator.zir:13)
  Val2Array x3 = Val2Array{MAKE_VAL(0), MAKE_VAL(1)};
  NondetRegStruct2Array x4 =
      map(x3,
          LAYOUT_LOOKUP(layout2, _super),
          std::function([&](Val2Array::value_type x5,
                            BoundLayout<NondetRegLayout2LayoutArray::value_type> x6) {
            Val x7 = exec_Sub(ctx0, x5, arg1);
            Val x8 = exec_Isz(ctx0, x7);
            NondetRegStruct x9 = exec_NondetReg(ctx0, x8, x6);
            return x9;
          }));
  // OneHot(zirgen/dsl/examples/calculator.zir:15)
  ComponentStruct2Array x10 = map(x4, std::function([&](NondetRegStruct2Array::value_type x11) {
                                    Val x12 = x11._super;
                                    Val x13 = exec_Sub(ctx0, MAKE_VAL(1), x12);
                                    Val x14 = exec_Mul(ctx0, x12, x13);
                                    EQZ(x14, "OneHot(zirgen/dsl/examples/calculator.zir:15)");
                                    ComponentStruct x15 = exec_Component(ctx0);
                                    return x15;
                                  }));
  // OneHot(zirgen/dsl/examples/calculator.zir:17)
  Val x16 =
      reduce(x4, MAKE_VAL(0), std::function([&](Val x17, NondetRegStruct2Array::value_type x18) {
               Val x19 = exec_Add(ctx0, x17, x18._super);
               return x19;
             }));
  EQZ((x16 - MAKE_VAL(1)), "OneHot(zirgen/dsl/examples/calculator.zir:17)");
  // OneHot(zirgen/dsl/examples/calculator.zir:19)
  OneHot__0_SuperStruct2Array x20 = map(x3, std::function([&](Val2Array::value_type x21) {
                                          Val x22 = exec_Mul(ctx0, x4[to_size_t(x21)]._super, x21);
                                          return OneHot__0_SuperStruct{._super = x22};
                                        }));
  Val x23 = reduce(
      x20, MAKE_VAL(0), std::function([&](Val x24, OneHot__0_SuperStruct2Array::value_type x25) {
        Val x26 = exec_Add(ctx0, x24, x25._super);
        return x26;
      }));
  EQZ((x23 - arg1), "OneHot(zirgen/dsl/examples/calculator.zir:19)");
  return OneHotStruct{._super = x4, .n = MAKE_VAL(2), .bits = x4};
}
RegStruct exec_GetGlobalResult(ExecContext ctx0) {
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
  BoundLayout<_globalLayout> x1 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx0, global));
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:24)
  RegStruct x2 = back_Reg(ctx0, 0, LAYOUT_LOOKUP(x1, result));
  return x2;
}
ComponentStruct exec_SetGlobalResult(ExecContext ctx0, Val arg1) {
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  BoundLayout<_globalLayout> x2 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx0, global));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
  RegStruct x3 = exec_Reg(ctx0, arg1, LAYOUT_LOOKUP(x2, result));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  ComponentStruct x4 = exec_Component(ctx0);
  return x4;
}
TopStruct exec_Top(ExecContext ctx0, BoundLayout<TopLayout> layout1) {
  // Top(zirgen/dsl/examples/calculator.zir:36)
  BoundLayout<TopResultLayout> x2 = LAYOUT_LOOKUP(layout1, result);
  // Top(zirgen/dsl/examples/calculator.zir:33)
  GetValFromUserStruct x3 = exec_GetValFromUser(ctx0);
  NondetRegStruct x4 = exec_NondetReg(ctx0, x3._super, LAYOUT_LOOKUP(layout1, op));
  // Top(zirgen/dsl/examples/calculator.zir:34)
  GetValFromUserStruct x5 = exec_GetValFromUser(ctx0);
  NondetRegStruct x6 = exec_NondetReg(ctx0, x5._super, LAYOUT_LOOKUP(layout1, left));
  // Top(zirgen/dsl/examples/calculator.zir:35)
  GetValFromUserStruct x7 = exec_GetValFromUser(ctx0);
  NondetRegStruct x8 = exec_NondetReg(ctx0, x7._super, LAYOUT_LOOKUP(layout1, right));
  // Top(zirgen/dsl/examples/calculator.zir:36)
  OneHotStruct x9 = exec_OneHot(ctx0, x4._super, LAYOUT_LOOKUP(layout1, _0));
  NondetRegStruct2Array x10 = x9._super;
  BoundLayout<NondetRegLayout> x11 = LAYOUT_LOOKUP(x2, _super);
  RegStruct x12;
  if (to_size_t(x10[0]._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:37)
    Val x13 = exec_Add(ctx0, x6._super, x8._super);
    RegStruct x14 = exec_Reg(ctx0, x13, LAYOUT_LOOKUP(x2, arm0));
    x12 = x14;
  } else if (to_size_t(x10[1]._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:38)
    Val x15 = exec_Sub(ctx0, x6._super, x8._super);
    RegStruct x16 = exec_Reg(ctx0, x15, LAYOUT_LOOKUP(x2, arm1));
    x12 = x16;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x17 = back_Reg(ctx0, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:40)
  ComponentStruct x18 =
      exec_Log(ctx0, "result = %u", std::initializer_list<Val>{x17._super._super});
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x19 = back_Reg(ctx0, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:41)
  ComponentStruct x20 = exec_SetGlobalResult(ctx0, x19._super._super);
  // Top(zirgen/dsl/examples/calculator.zir:42)
  RegStruct x21 = exec_GetGlobalResult(ctx0);
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x22 = back_Reg(ctx0, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:42)
  EQZ((x21._super._super - x22._super._super), "Top(zirgen/dsl/examples/calculator.zir:42)");
  // Top(zirgen/dsl/examples/calculator.zir:36)
  RegStruct x23 = back_Reg(ctx0, 0, x11);
  // Top(zirgen/dsl/examples/calculator.zir:43)
  ComponentStruct x24 = exec_OutputToUser(ctx0, x23._super._super);
  // Top(zirgen/dsl/examples/calculator.zir:32)
  ComponentStruct x25 = exec_Component(ctx0);
  return TopStruct{._super = x25, .op = x4, .left = x6, .right = x8, .result = x17};
}
void step_Top(ExecContext ctx0) {
  BoundLayout<TopLayout> x1 = BIND_LAYOUT(kLayout_Top, GET_BUFFER(ctx0, data));
  TopStruct x2 = exec_Top(ctx0, x1);
  return;
}
MixState validityTaps_(ValidityTapsContext ctx0, PolyMix polyMix1, ExtVal8Array taps2) {
  // Top(zirgen/dsl/examples/calculator.zir:36)
  MixState x3 = trivialConstraint();
  // OneHot(zirgen/dsl/examples/calculator.zir:19)
  MixState x4 = andEqzExt(
      polyMix1,
      andEqzExt(
          polyMix1,
          andEqzExt(polyMix1,
                    andEqzExt(polyMix1, x3, (taps2[5] * (MAKE_VAL_EXT(1, 0, 0, 0) - taps2[5]))),
                    (taps2[6] * (MAKE_VAL_EXT(1, 0, 0, 0) - taps2[6]))),
          ((taps2[5] + taps2[6]) - MAKE_VAL_EXT(1, 0, 0, 0))),
      (taps2[6] - taps2[2]));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  // Top(zirgen/dsl/examples/calculator.zir:41)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx0, global));
  // Top(zirgen/dsl/examples/calculator.zir:42)
  MixState x6 = andEqzExt(
      polyMix1,
      andEqzExt(
          polyMix1,
          andCondExt(
              andCondExt(x4, taps2[5], andEqzExt(polyMix1, x3, ((taps2[3] + taps2[4]) - taps2[7]))),
              taps2[6],
              andEqzExt(polyMix1, x3, ((taps2[3] - taps2[4]) - taps2[7]))),
          (taps2[7] - LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x5, result), _super), 0))),
      (LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x5, result), _super), 0) - taps2[7]));
  return x6;
}
MixState validityRegs_(ValidityRegsContext ctx0, PolyMix polyMix1) {
  // Top(zirgen/dsl/examples/calculator.zir:36)
  MixState x2 = trivialConstraint();
  BoundLayout<TopLayout> x3 = BIND_LAYOUT(kLayout_Top, GET_BUFFER(ctx0, data));
  // Top(zirgen/dsl/examples/calculator.zir:36)
  BoundLayout<TopResultLayout> x4 = LAYOUT_LOOKUP(x3, result);
  // OneHot(zirgen/dsl/examples/calculator.zir:13)
  BoundLayout<NondetRegLayout2LayoutArray> x5 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, _0), _super);
  // OneHot(zirgen/dsl/examples/calculator.zir:15)
  MixState x6 =
      andEqz(polyMix1,
             andEqz(polyMix1,
                    x2,
                    (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 0), _super), 0) *
                     (MAKE_VAL(1) - LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 0), _super), 0)))),
             (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 1), _super), 0) *
              (MAKE_VAL(1) - LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 1), _super), 0))));
  MixState x7 = andCond(andEqz(polyMix1,
                               andEqz(polyMix1,
                                      x6,
                                      ((LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 0), _super), 0) +
                                        LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 1), _super), 0)) -
                                       MAKE_VAL(1))),
                               (LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 1), _super), 0) -
                                LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, op), _super), 0))),
                        LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 0), _super), 0),
                        andEqz(polyMix1,
                               x2,
                               ((LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, left), _super), 0) +
                                 LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, right), _super), 0)) -
                                LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, arm0), _super), 0))));
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
  // Top(zirgen/dsl/examples/calculator.zir:41)
  BoundLayout<_globalLayout> x8 = BIND_LAYOUT(kLayoutGlobal, GET_BUFFER(ctx0, global));
  // Top(zirgen/dsl/examples/calculator.zir:42)
  MixState x9 =
      andEqz(polyMix1,
             andEqz(polyMix1,
                    andCond(x7,
                            LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x5, 1), _super), 0),
                            andEqz(polyMix1,
                                   x2,
                                   ((LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, left), _super), 0) -
                                     LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x3, right), _super), 0)) -
                                    LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, arm1), _super), 0)))),
                    (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, _super), _super), 0) -
                     LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, result), _super), 0))),
             (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, result), _super), 0) -
              LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, _super), _super), 0)));
  return x9;
}
