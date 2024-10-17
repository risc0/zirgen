constexpr Top__0Layout kLayout__0 =
    Top__0Layout{._super = MyArgumentLayout{.a = NondetRegLayout{._super = /*offset=*/2},
                                            .b = NondetRegLayout{._super = /*offset=*/3}},
                 ._selector = NondetRegLayout2LayoutArray{NondetRegLayout{._super = /*offset=*/0},
                                                          NondetRegLayout{._super = /*offset=*/1}},
                 .arm0 = MyArgumentLayout{.a = NondetRegLayout{._super = /*offset=*/2},
                                          .b = NondetRegLayout{._super = /*offset=*/3}},
                 .arm1 = MyArgumentLayout{.a = NondetRegLayout{._super = /*offset=*/2},
                                          .b = NondetRegLayout{._super = /*offset=*/3}}};
constexpr Reg1LayoutArray kLayoutTestSucc_0Accum = Reg1LayoutArray{/*offset=*/0};
constexpr Reg1LayoutArray kLayout_TopAccum = Reg1LayoutArray{/*offset=*/0};
constexpr TopLayout kLayoutTestSucc_0 = TopLayout{.odd = NondetRegLayout{._super = /*offset=*/0},
                                                  ._0 = NondetRegLayout{._super = /*offset=*/1},
                                                  ._1 = kLayout__0};
constexpr TopLayout kLayout_Top = TopLayout{.odd = NondetRegLayout{._super = /*offset=*/0},
                                            ._0 = NondetRegLayout{._super = /*offset=*/1},
                                            ._1 = kLayout__0};
constexpr _globalLayout kLayoutGlobal =
    _globalLayout{.globzero = NondetRegLayout{._super = /*offset=*/0}};
constexpr _mixLayout kLayoutMix =
    _mixLayout{.randomness = _accumLayout{.myArgument = Arg_MyArgumentLayout{.b = /*offset=*/0},
                                          ._offset = /*offset=*/4}};
