struct NondetRegLayout {
  Reg _super;
};
struct MyArgumentLayout {
  NondetRegLayout a;
  NondetRegLayout b;
};
using NondetRegLayout2LayoutArray = std::array<NondetRegLayout, 2>;
struct Top__0Layout {
  MyArgumentLayout _super;
  NondetRegLayout2LayoutArray _selector;
  MyArgumentLayout arm0;
  MyArgumentLayout arm1;
};
using Reg1LayoutArray = std::array<Reg, 1>;
struct TopLayout {
  NondetRegLayout odd;
  NondetRegLayout _0;
  Top__0Layout _1;
};
struct _globalLayout {
  NondetRegLayout globzero;
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
struct MyArgumentStruct {
};
struct TopStruct {
};
struct TestSucc_0Struct {
};
struct ComponentStruct {
};
