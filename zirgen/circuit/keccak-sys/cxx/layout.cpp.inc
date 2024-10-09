constexpr Top__0Layout kLayout__0 =
    Top__0Layout{._selector = NondetRegLayout2LayoutArray{NondetRegLayout{._super = /*offset=*/0},
                                                          NondetRegLayout{._super = /*offset=*/1}},
                 .arm0 = MyArgumentLayout{.a = NondetRegLayout{._super = /*offset=*/8},
                                          .b = NondetRegLayout{._super = /*offset=*/9}},
                 .arm1 = NondetRegLayout{._super = /*offset=*/8}};
constexpr Reg1LayoutArray kLayout_TopAccum = Reg1LayoutArray{/*offset=*/0};
constexpr TopLayout kLayout_Top = TopLayout{
    .op = NondetRegLayout{._super = /*offset=*/2},
    .left = NondetRegLayout{._super = /*offset=*/3},
    .right = NondetRegLayout{._super = /*offset=*/4},
    ._0 = OneHotLayout{._super =
                           NondetRegLayout2LayoutArray{NondetRegLayout{._super = /*offset=*/5},
                                                       NondetRegLayout{._super = /*offset=*/6}}},
    .result = TopResultLayout{._super = NondetRegLayout{._super = /*offset=*/7},
                              .arm0 = NondetRegLayout{._super = /*offset=*/7},
                              .arm1 = NondetRegLayout{._super = /*offset=*/7}},
    ._1 = NondetRegLayout{._super = /*offset=*/0},
    ._2 = NondetRegLayout{._super = /*offset=*/1},
    ._3 = kLayout__0};
constexpr _globalLayout kLayoutGlobal =
    _globalLayout{.result = NondetRegLayout{._super = /*offset=*/0}};
constexpr _mixLayout kLayoutMix =
    _mixLayout{.randomness = _accumLayout{.myArgument = Arg_MyArgumentLayout{.b = /*offset=*/0},
                                          ._offset = /*offset=*/4}};
