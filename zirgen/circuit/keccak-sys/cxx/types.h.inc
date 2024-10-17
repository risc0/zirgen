struct NondetRegLayout {
  Reg _super;
};
struct NondetMyArgumentLayout {
  NondetRegLayout a;
  NondetRegLayout b;
};
using NondetRegLayout2LayoutArray = std::array<NondetRegLayout, 2>;
struct Top__0Layout {
  NondetMyArgumentLayout _super;
  NondetRegLayout2LayoutArray _selector;
  NondetMyArgumentLayout arm0;
  NondetMyArgumentLayout arm1;
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
struct Arg_NondetMyArgumentLayout {
  Reg b;
};
struct _accumLayout {
  Arg_NondetMyArgumentLayout nondetMyArgument;
  Reg _offset;
};
struct _mixLayout {
  _accumLayout randomness;
};
struct NondetRegStruct {
  Val _super;
};
struct NondetMyArgumentStruct {
};
struct MyArgumentStruct {
};
struct TopStruct {
};
struct ComponentStruct {
};
