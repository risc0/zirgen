pub const LAYOUT__0: &Top__0Layout = &Top__0Layout {
    _super: &NondetMyArgumentLayout {
        a: &NondetRegLayout {
            _super: &Reg { offset: 2 },
        },
        b: &NondetRegLayout {
            _super: &Reg { offset: 3 },
        },
    },
    _selector: &[
        &NondetRegLayout {
            _super: &Reg { offset: 0 },
        },
        &NondetRegLayout {
            _super: &Reg { offset: 1 },
        },
    ],
    arm0: &NondetMyArgumentLayout {
        a: &NondetRegLayout {
            _super: &Reg { offset: 2 },
        },
        b: &NondetRegLayout {
            _super: &Reg { offset: 3 },
        },
    },
    arm1: &NondetMyArgumentLayout {
        a: &NondetRegLayout {
            _super: &Reg { offset: 2 },
        },
        b: &NondetRegLayout {
            _super: &Reg { offset: 3 },
        },
    },
};
pub const LAYOUT_TEST_SUCC_0_ACCUM: &Reg1LayoutArray = &[&Reg { offset: 0 }];
pub const LAYOUT_TOP_ACCUM: &Reg1LayoutArray = &[&Reg { offset: 0 }];
pub const LAYOUT_TEST_SUCC_0: &TopLayout = &TopLayout {
    odd: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    _0: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    _1: LAYOUT__0,
};
pub const LAYOUT_TOP: &TopLayout = &TopLayout {
    odd: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    _0: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    _1: LAYOUT__0,
};
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    globzero: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
};
pub const LAYOUT_MIX: &_mixLayout = &_mixLayout {
    randomness: &_accumLayout {
        nondet_my_argument: &Arg_NondetMyArgumentLayout {
            b: &Reg { offset: 0 },
        },
        _offset: &Reg { offset: 4 },
    },
};
