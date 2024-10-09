pub const LAYOUT__0: &Top__0Layout = &Top__0Layout {
    _selector: &[
        &NondetRegLayout {
            _super: &Reg { offset: 0 },
        },
        &NondetRegLayout {
            _super: &Reg { offset: 1 },
        },
    ],
    arm0: &MyArgumentLayout {
        a: &NondetRegLayout {
            _super: &Reg { offset: 8 },
        },
        b: &NondetRegLayout {
            _super: &Reg { offset: 9 },
        },
    },
    arm1: &NondetRegLayout {
        _super: &Reg { offset: 8 },
    },
};
pub const LAYOUT_TOP_ACCUM: &Reg1LayoutArray = &[&Reg { offset: 0 }];
pub const LAYOUT_TOP: &TopLayout = &TopLayout {
    op: &NondetRegLayout {
        _super: &Reg { offset: 2 },
    },
    left: &NondetRegLayout {
        _super: &Reg { offset: 3 },
    },
    right: &NondetRegLayout {
        _super: &Reg { offset: 4 },
    },
    _0: &OneHotLayout {
        _super: &[
            &NondetRegLayout {
                _super: &Reg { offset: 5 },
            },
            &NondetRegLayout {
                _super: &Reg { offset: 6 },
            },
        ],
    },
    result: &TopResultLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 7 },
        },
        arm0: &NondetRegLayout {
            _super: &Reg { offset: 7 },
        },
        arm1: &NondetRegLayout {
            _super: &Reg { offset: 7 },
        },
    },
    _1: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    _2: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    _3: LAYOUT__0,
};
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    result: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
};
pub const LAYOUT_MIX: &_mixLayout = &_mixLayout {
    randomness: &_accumLayout {
        my_argument: &Arg_MyArgumentLayout {
            b: &Reg { offset: 0 },
        },
        _offset: &Reg { offset: 4 },
    },
};
