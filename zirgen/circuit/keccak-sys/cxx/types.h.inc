struct NondetRegLayout {
  Reg _super;
};
using NondetRegLayout2LayoutArray = std::array<NondetRegLayout, 2>;
struct MyArgumentLayout {
  NondetRegLayout a;
  NondetRegLayout b;
};
struct Top__0Layout {
  NondetRegLayout2LayoutArray _selector;
  MyArgumentLayout arm0;
  NondetRegLayout arm1;
};
using Reg1LayoutArray = std::array<Reg, 1>;
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
  NondetRegLayout _1;
  NondetRegLayout _2;
  Top__0Layout _3;
};
struct _globalLayout {
  NondetRegLayout result;
};
struct Arg_MyArgumentLayout {
  Reg b;
};
struct _accumLayout {
  Arg_MyArgumentLayout myArgument;
  Reg _offset;
};
struct _mixLayout {
  _accumLayout randomness;
};
struct NondetRegStruct {
  Val _super;
};
using Val2Array = std::array<Val, 2>;
using NondetRegStruct2Array = std::array<NondetRegStruct, 2>;
struct OneHotStruct {
  NondetRegStruct2Array _super;
};
struct ComponentStruct {
};
struct TopStruct {
};
