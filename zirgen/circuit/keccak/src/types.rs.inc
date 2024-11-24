pub struct NondetRegLayout {
    pub _super: &'static Reg,
}
impl risc0_zkp::layout::Component for NondetRegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout22LayoutArray = [&'static NondetRegLayout; 22];
pub type NondetRegLayout24LayoutArray = [&'static NondetRegLayout; 24];
pub type NondetRegLayout16LayoutArray = [&'static NondetRegLayout; 16];
pub struct RetTupleLayout {
    pub arr_a: &'static NondetRegLayout22LayoutArray,
    pub arr_b: &'static NondetRegLayout22LayoutArray,
    pub arr_c: &'static NondetRegLayout22LayoutArray,
    pub arr_d: &'static NondetRegLayout24LayoutArray,
    pub arr_e: &'static NondetRegLayout22LayoutArray,
    pub arr_f: &'static NondetRegLayout22LayoutArray,
    pub minor: &'static NondetRegLayout,
    pub major: &'static NondetRegLayout,
    pub rnd: &'static NondetRegLayout,
    pub blk: &'static NondetRegLayout,
    pub midx: &'static NondetRegLayout,
    pub auxr: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for RetTupleLayout {
    fn ty_name(&self) -> &'static str {
        "RetTupleLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arr_a", self.arr_a)?;
        v.visit_component("arr_b", self.arr_b)?;
        v.visit_component("arr_c", self.arr_c)?;
        v.visit_component("arr_d", self.arr_d)?;
        v.visit_component("arr_e", self.arr_e)?;
        v.visit_component("arr_f", self.arr_f)?;
        v.visit_component("minor", self.minor)?;
        v.visit_component("major", self.major)?;
        v.visit_component("rnd", self.rnd)?;
        v.visit_component("blk", self.blk)?;
        v.visit_component("midx", self.midx)?;
        v.visit_component("auxr", self.auxr)?;
        Ok(())
    }
}
pub struct OneHot_22_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_22_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_22_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout5LayoutArray = [&'static NondetRegLayout; 5];
pub struct OneHotU_5_Layout {
    pub _super: &'static NondetRegLayout5LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_5_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ToBits_22_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ToBits_22_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBits_22_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ToBitsU_22_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_22_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_22_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsResult_0_Layout {
    pub minor_onehot: &'static OneHotU_5_Layout,
    pub unpack_a: &'static ToBits_22_Layout,
    pub unpack_b: &'static ToBits_22_Layout,
    pub unpack_c: &'static ToBits_22_Layout,
    pub unpack_d: &'static ToBits_22_Layout,
    pub unpack_e: &'static ToBitsU_22_Layout,
    pub resbits: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Xor5wordsResult_0_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsResult_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_b", self.unpack_b)?;
        v.visit_component("unpack_c", self.unpack_c)?;
        v.visit_component("unpack_d", self.unpack_d)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("resbits", self.resbits)?;
        Ok(())
    }
}
pub struct Concatenate_22__2_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_22__2_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_22__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_5__11_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_5__11_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_5__11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout2LayoutArray = [&'static NondetRegLayout; 2];
pub struct Xor5wordsMinor0Layout {
    pub _super: &'static RetTupleLayout,
    pub r: &'static Xor5wordsResult_0_Layout,
    pub result: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout2LayoutArray,
    pub withr_d: &'static Concatenate_22__2_Layout,
    pub aux_regs: &'static ConcatZeros_5__11_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsMinor0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsMinor0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("r", self.r)?;
        v.visit_component("result", self.result)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub type NondetRegLayout21LayoutArray = [&'static NondetRegLayout; 21];
pub struct ToBits_21_Layout {
    pub _super: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for ToBits_21_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBits_21_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ToBitsU_21_Layout {
    pub _super: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_21_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_21_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsResult_1_Layout {
    pub minor_onehot: &'static OneHotU_5_Layout,
    pub unpack_a: &'static ToBits_21_Layout,
    pub unpack_b: &'static ToBits_21_Layout,
    pub unpack_c: &'static ToBits_21_Layout,
    pub unpack_d: &'static ToBits_21_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub resbits: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for Xor5wordsResult_1_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsResult_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_b", self.unpack_b)?;
        v.visit_component("unpack_c", self.unpack_c)?;
        v.visit_component("unpack_d", self.unpack_d)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("resbits", self.resbits)?;
        Ok(())
    }
}
pub struct ConcatElem_21_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatElem_21_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatElem_21_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_21__1_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_21__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_21__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_21__3_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_21__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_21__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsMinor1Layout {
    pub _super: &'static RetTupleLayout,
    pub r: &'static Xor5wordsResult_1_Layout,
    pub result: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub zc: &'static ConcatZeros_21__1_Layout,
    pub zd: &'static ConcatZeros_21__3_Layout,
    pub ze: &'static ConcatZeros_21__1_Layout,
    pub zx: &'static ConcatZeros_21__1_Layout,
    pub aux_regs: &'static ConcatZeros_5__11_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsMinor1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsMinor1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("r", self.r)?;
        v.visit_component("result", self.result)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("zc", self.zc)?;
        v.visit_component("zd", self.zd)?;
        v.visit_component("ze", self.ze)?;
        v.visit_component("zx", self.zx)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct Xor5wordsResult_2_Layout {
    pub minor_onehot: &'static OneHotU_5_Layout,
    pub unpack_a: &'static ToBits_21_Layout,
    pub unpack_b: &'static ToBits_21_Layout,
    pub unpack_c: &'static ToBits_21_Layout,
    pub unpack_d: &'static ToBits_21_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub resbits: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for Xor5wordsResult_2_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsResult_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_b", self.unpack_b)?;
        v.visit_component("unpack_c", self.unpack_c)?;
        v.visit_component("unpack_d", self.unpack_d)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("resbits", self.resbits)?;
        Ok(())
    }
}
pub struct Xor5wordsMinor2Layout {
    pub _super: &'static RetTupleLayout,
    pub r: &'static Xor5wordsResult_2_Layout,
    pub result: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
    pub zd: &'static ConcatZeros_21__3_Layout,
    pub ze: &'static ConcatZeros_21__1_Layout,
    pub zx: &'static ConcatZeros_21__1_Layout,
    pub aux_regs: &'static ConcatZeros_5__11_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsMinor2Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsMinor2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("r", self.r)?;
        v.visit_component("result", self.result)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        v.visit_component("zd", self.zd)?;
        v.visit_component("ze", self.ze)?;
        v.visit_component("zx", self.zx)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct DummyBLayout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for DummyBLayout {
    fn ty_name(&self) -> &'static str {
        "DummyBLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_0__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_0__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_0__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_0__SuperArm0Layout,
    pub arm1: &'static Xor5wordsM3impl_0__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RGZ_22_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for RGZ_22_Layout {
    fn ty_name(&self) -> &'static str {
        "RGZ_22_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout9LayoutArray = [&'static NondetRegLayout; 9];
pub type NondetRegLayout12LayoutArray = [&'static NondetRegLayout; 12];
pub struct Concatenate_9__3_Layout {
    pub _super: &'static NondetRegLayout12LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_9__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_9__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_12__12_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_12__12_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_12__12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout3LayoutArray = [&'static NondetRegLayout; 3];
pub struct Xor5wordsM3impl_0_Layout {
    pub _super: &'static Xor5wordsM3impl_0__SuperLayout,
    pub inp_a: &'static RGZ_22_Layout,
    pub inp_b: &'static RGZ_22_Layout,
    pub inp_c: &'static RGZ_22_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout3LayoutArray,
    pub tmp_1: &'static NondetRegLayout9LayoutArray,
    pub tmp_2: &'static Concatenate_9__3_Layout,
    pub inp_d: &'static ConcatZeros_12__12_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_0_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inp_a", self.inp_a)?;
        v.visit_component("inp_b", self.inp_b)?;
        v.visit_component("inp_c", self.inp_c)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component("tmp_1", self.tmp_1)?;
        v.visit_component("tmp_2", self.tmp_2)?;
        v.visit_component("inp_d", self.inp_d)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_1__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_1__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_1__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_1__SuperArm0Layout,
    pub arm1: &'static Xor5wordsM3impl_1__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type NondetRegLayout15LayoutArray = [&'static NondetRegLayout; 15];
pub struct Concatenate_12__3_Layout {
    pub _super: &'static NondetRegLayout15LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_12__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_12__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_15__9_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_15__9_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_15__9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_1_Layout {
    pub _super: &'static Xor5wordsM3impl_1__SuperLayout,
    pub inp_a: &'static RGZ_22_Layout,
    pub inp_b: &'static RGZ_22_Layout,
    pub inp_c: &'static RGZ_22_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout3LayoutArray,
    pub tmp_1: &'static NondetRegLayout12LayoutArray,
    pub tmp_2: &'static Concatenate_12__3_Layout,
    pub inp_d: &'static ConcatZeros_15__9_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_1_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inp_a", self.inp_a)?;
        v.visit_component("inp_b", self.inp_b)?;
        v.visit_component("inp_c", self.inp_c)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component("tmp_1", self.tmp_1)?;
        v.visit_component("tmp_2", self.tmp_2)?;
        v.visit_component("inp_d", self.inp_d)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_2__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_2__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_2__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_2__SuperArm0Layout,
    pub arm1: &'static Xor5wordsM3impl_2__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type NondetRegLayout18LayoutArray = [&'static NondetRegLayout; 18];
pub struct Concatenate_15__3_Layout {
    pub _super: &'static NondetRegLayout18LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_15__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_15__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_18__6_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_18__6_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_18__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_2_Layout {
    pub _super: &'static Xor5wordsM3impl_2__SuperLayout,
    pub inp_a: &'static RGZ_22_Layout,
    pub inp_b: &'static RGZ_22_Layout,
    pub inp_c: &'static RGZ_22_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout3LayoutArray,
    pub tmp_1: &'static NondetRegLayout15LayoutArray,
    pub tmp_2: &'static Concatenate_15__3_Layout,
    pub inp_d: &'static ConcatZeros_18__6_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_2_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inp_a", self.inp_a)?;
        v.visit_component("inp_b", self.inp_b)?;
        v.visit_component("inp_c", self.inp_c)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component("tmp_1", self.tmp_1)?;
        v.visit_component("tmp_2", self.tmp_2)?;
        v.visit_component("inp_d", self.inp_d)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_3__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_3__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_3__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_3__SuperArm0Layout,
    pub arm1: &'static Xor5wordsM3impl_3__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Concatenate_18__3_Layout {
    pub _super: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_18__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_18__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_3_Layout {
    pub _super: &'static Xor5wordsM3impl_3__SuperLayout,
    pub inp_a: &'static RGZ_22_Layout,
    pub inp_b: &'static RGZ_22_Layout,
    pub inp_c: &'static RGZ_22_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout3LayoutArray,
    pub tmp_1: &'static NondetRegLayout18LayoutArray,
    pub tmp_2: &'static Concatenate_18__3_Layout,
    pub inp_d: &'static ConcatZeros_21__3_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_3_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inp_a", self.inp_a)?;
        v.visit_component("inp_b", self.inp_b)?;
        v.visit_component("inp_c", self.inp_c)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component("tmp_1", self.tmp_1)?;
        v.visit_component("tmp_2", self.tmp_2)?;
        v.visit_component("inp_d", self.inp_d)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_4__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_4__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_4__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_4__SuperArm0Layout,
    pub arm1: &'static Xor5wordsM3impl_4__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Concatenate_21__3_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_21__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_21__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_24__0_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_24__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_24__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Xor5wordsM3impl_4_Layout {
    pub _super: &'static Xor5wordsM3impl_4__SuperLayout,
    pub inp_a: &'static RGZ_22_Layout,
    pub inp_b: &'static RGZ_22_Layout,
    pub inp_c: &'static RGZ_22_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout3LayoutArray,
    pub tmp_1: &'static NondetRegLayout21LayoutArray,
    pub tmp_2: &'static Concatenate_21__3_Layout,
    pub inp_d: &'static ConcatZeros_24__0_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsM3impl_4_Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsM3impl_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inp_a", self.inp_a)?;
        v.visit_component("inp_b", self.inp_b)?;
        v.visit_component("inp_c", self.inp_c)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component("tmp_1", self.tmp_1)?;
        v.visit_component("tmp_2", self.tmp_2)?;
        v.visit_component("inp_d", self.inp_d)?;
        Ok(())
    }
}
pub struct Xor5wordsMinor3_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Xor5wordsM3impl_0_Layout,
    pub arm1: &'static Xor5wordsM3impl_1_Layout,
    pub arm2: &'static Xor5wordsM3impl_2_Layout,
    pub arm3: &'static Xor5wordsM3impl_3_Layout,
    pub arm4: &'static Xor5wordsM3impl_4_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsMinor3_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsMinor3_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        Ok(())
    }
}
pub struct Xor5wordsMinor3Layout {
    pub _super: &'static Xor5wordsMinor3_SuperLayout,
    pub minor_onehot: &'static OneHotU_5_Layout,
    pub aux_regs: &'static ConcatZeros_5__11_Layout,
}
impl risc0_zkp::layout::Component for Xor5wordsMinor3Layout {
    fn ty_name(&self) -> &'static str {
        "Xor5wordsMinor3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct RotLeft3impl_22__21__21__1_Layout {
    pub ra: &'static NondetRegLayout22LayoutArray,
    pub rb: &'static NondetRegLayout21LayoutArray,
    pub rc: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for RotLeft3impl_22__21__21__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RotLeft3impl_22__21__21__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("ra", self.ra)?;
        v.visit_component("rb", self.rb)?;
        v.visit_component("rc", self.rc)?;
        Ok(())
    }
}
pub struct ConcatZeros_22__2_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_22__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_22__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RhoMinor0_0__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_0__0__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_0__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__0_Layout {
    pub _super: &'static RhoMinor1_0__0__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_0__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_2__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_2__0__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_2__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__0_Layout {
    pub _super: &'static RhoMinor1_2__0__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_2__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_4__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_4__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_4__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_4__0__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_4__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__0_Layout {
    pub _super: &'static RhoMinor1_4__0__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub type NondetRegLayout6LayoutArray = [&'static NondetRegLayout; 6];
pub struct Concatenate_3__3_Layout {
    pub _super: &'static NondetRegLayout6LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_3__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_3__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_6__3_Layout {
    pub _super: &'static NondetRegLayout9LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_6__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_6__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout7LayoutArray = [&'static NondetRegLayout; 7];
pub struct Concatenate_15__7_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_15__7_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_15__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RGZ_24_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for RGZ_24_Layout {
    fn ty_name(&self) -> &'static str {
        "RGZ_24_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _1: &'static NondetRegLayout3LayoutArray,
    pub _3: &'static NondetRegLayout3LayoutArray,
    pub _4: &'static Concatenate_3__3_Layout,
    pub _6: &'static NondetRegLayout3LayoutArray,
    pub _7: &'static Concatenate_6__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub _10: &'static Concatenate_9__3_Layout,
    pub _12: &'static NondetRegLayout3LayoutArray,
    pub rs1: &'static Concatenate_12__3_Layout,
    pub _14: &'static NondetRegLayout7LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _15: &'static RGZ_22_Layout,
    pub _16: &'static RGZ_22_Layout,
    pub _17: &'static RGZ_24_Layout,
    pub _18: &'static DummyBLayout,
    pub _19: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        Ok(())
    }
}
pub type NondetRegLayout8LayoutArray = [&'static NondetRegLayout; 8];
pub type NondetRegLayout14LayoutArray = [&'static NondetRegLayout; 14];
pub struct Concatenate_8__14_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_8__14_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_8__14_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct Concatenate_1__21_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_1__21_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_1__21_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout1LayoutArray = [&'static NondetRegLayout; 1];
pub struct RhoMinor3_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _17: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct Concatenate_1__15_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_1__15_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_1__15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_16__6_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_16__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_16__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout1LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _18: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
    pub _21: &'static RGZ_24_Layout,
    pub _22: &'static DummyBLayout,
    pub _23: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static NondetRegLayout,
    pub _45: &'static NondetRegLayout,
    pub _46: &'static NondetRegLayout,
    pub _47: &'static NondetRegLayout,
    pub _48: &'static NondetRegLayout,
    pub _49: &'static NondetRegLayout,
    pub _50: &'static NondetRegLayout,
    pub _51: &'static NondetRegLayout,
    pub _52: &'static NondetRegLayout,
    pub _53: &'static NondetRegLayout,
    pub _54: &'static NondetRegLayout,
    pub _55: &'static NondetRegLayout,
    pub _56: &'static NondetRegLayout,
    pub _57: &'static NondetRegLayout,
    pub _58: &'static NondetRegLayout,
    pub _59: &'static NondetRegLayout,
    pub _60: &'static NondetRegLayout,
    pub _61: &'static NondetRegLayout,
    pub _62: &'static NondetRegLayout,
    pub _63: &'static NondetRegLayout,
    pub _64: &'static NondetRegLayout,
    pub _65: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _66: &'static NondetRegLayout,
    pub _67: &'static NondetRegLayout,
    pub _68: &'static NondetRegLayout,
    pub _69: &'static NondetRegLayout,
    pub _70: &'static NondetRegLayout,
    pub _71: &'static NondetRegLayout,
    pub _72: &'static NondetRegLayout,
    pub _73: &'static NondetRegLayout,
    pub _74: &'static NondetRegLayout,
    pub _75: &'static NondetRegLayout,
    pub _76: &'static NondetRegLayout,
    pub _77: &'static NondetRegLayout,
    pub _78: &'static NondetRegLayout,
    pub _79: &'static NondetRegLayout,
    pub _80: &'static NondetRegLayout,
    pub _81: &'static NondetRegLayout,
    pub _82: &'static NondetRegLayout,
    pub _83: &'static NondetRegLayout,
    pub _84: &'static NondetRegLayout,
    pub _85: &'static NondetRegLayout,
    pub _86: &'static NondetRegLayout,
    pub _87: &'static NondetRegLayout,
    pub _88: &'static NondetRegLayout,
    pub _89: &'static NondetRegLayout,
    pub d: &'static NondetRegLayout24LayoutArray,
    pub _90: &'static DummyBLayout,
    pub _91: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        v.visit_component("_48", self._48)?;
        v.visit_component("_49", self._49)?;
        v.visit_component("_50", self._50)?;
        v.visit_component("_51", self._51)?;
        v.visit_component("_52", self._52)?;
        v.visit_component("_53", self._53)?;
        v.visit_component("_54", self._54)?;
        v.visit_component("_55", self._55)?;
        v.visit_component("_56", self._56)?;
        v.visit_component("_57", self._57)?;
        v.visit_component("_58", self._58)?;
        v.visit_component("_59", self._59)?;
        v.visit_component("_60", self._60)?;
        v.visit_component("_61", self._61)?;
        v.visit_component("_62", self._62)?;
        v.visit_component("_63", self._63)?;
        v.visit_component("_64", self._64)?;
        v.visit_component("_65", self._65)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_66", self._66)?;
        v.visit_component("_67", self._67)?;
        v.visit_component("_68", self._68)?;
        v.visit_component("_69", self._69)?;
        v.visit_component("_70", self._70)?;
        v.visit_component("_71", self._71)?;
        v.visit_component("_72", self._72)?;
        v.visit_component("_73", self._73)?;
        v.visit_component("_74", self._74)?;
        v.visit_component("_75", self._75)?;
        v.visit_component("_76", self._76)?;
        v.visit_component("_77", self._77)?;
        v.visit_component("_78", self._78)?;
        v.visit_component("_79", self._79)?;
        v.visit_component("_80", self._80)?;
        v.visit_component("_81", self._81)?;
        v.visit_component("_82", self._82)?;
        v.visit_component("_83", self._83)?;
        v.visit_component("_84", self._84)?;
        v.visit_component("_85", self._85)?;
        v.visit_component("_86", self._86)?;
        v.visit_component("_87", self._87)?;
        v.visit_component("_88", self._88)?;
        v.visit_component("_89", self._89)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_90", self._90)?;
        v.visit_component("_91", self._91)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_0__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_0__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_0__SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMajor_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor0_0__0_Layout,
    pub arm1: &'static RhoMinor1_0__0_Layout,
    pub arm2: &'static RhoMinor2_0__0_Layout,
    pub arm3: &'static RhoMinor0_2__0_Layout,
    pub arm4: &'static RhoMinor1_2__0_Layout,
    pub arm5: &'static RhoMinor2_2__0_Layout,
    pub arm6: &'static RhoMinor0_4__0_Layout,
    pub arm7: &'static RhoMinor1_4__0_Layout,
    pub arm8: &'static RhoMinor3_0__SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMajor_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct OneHotU_9_Layout {
    pub _super: &'static NondetRegLayout9LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_9_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_9__7_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_9__7_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_9__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct RhoMajor_0_Layout {
    pub _super: &'static RhoMajor_0__SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__7_Layout,
}
impl risc0_zkp::layout::Component for RhoMajor_0_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct RhoMinor0_0__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_0__1__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_0__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__1_Layout {
    pub _super: &'static RhoMinor1_0__1__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_0__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_2__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_2__1__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_2__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__1_Layout {
    pub _super: &'static RhoMinor1_2__1__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_2__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_4__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_4__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_4__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_4__1__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_4__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__1_Layout {
    pub _super: &'static RhoMinor1_4__1__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _1: &'static NondetRegLayout3LayoutArray,
    pub _3: &'static NondetRegLayout3LayoutArray,
    pub _4: &'static Concatenate_3__3_Layout,
    pub _6: &'static NondetRegLayout3LayoutArray,
    pub _7: &'static Concatenate_6__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub _10: &'static Concatenate_9__3_Layout,
    pub _12: &'static NondetRegLayout3LayoutArray,
    pub rs1: &'static Concatenate_12__3_Layout,
    pub _14: &'static NondetRegLayout7LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _15: &'static RGZ_22_Layout,
    pub _16: &'static RGZ_22_Layout,
    pub _17: &'static RGZ_24_Layout,
    pub _18: &'static DummyBLayout,
    pub _19: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _17: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout1LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _18: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
    pub _21: &'static RGZ_24_Layout,
    pub _22: &'static DummyBLayout,
    pub _23: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static NondetRegLayout,
    pub _45: &'static NondetRegLayout,
    pub _46: &'static NondetRegLayout,
    pub _47: &'static NondetRegLayout,
    pub _48: &'static NondetRegLayout,
    pub _49: &'static NondetRegLayout,
    pub _50: &'static NondetRegLayout,
    pub _51: &'static NondetRegLayout,
    pub _52: &'static NondetRegLayout,
    pub _53: &'static NondetRegLayout,
    pub _54: &'static NondetRegLayout,
    pub _55: &'static NondetRegLayout,
    pub _56: &'static NondetRegLayout,
    pub _57: &'static NondetRegLayout,
    pub _58: &'static NondetRegLayout,
    pub _59: &'static NondetRegLayout,
    pub _60: &'static NondetRegLayout,
    pub _61: &'static NondetRegLayout,
    pub _62: &'static NondetRegLayout,
    pub _63: &'static NondetRegLayout,
    pub _64: &'static NondetRegLayout,
    pub _65: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _66: &'static NondetRegLayout,
    pub _67: &'static NondetRegLayout,
    pub _68: &'static NondetRegLayout,
    pub _69: &'static NondetRegLayout,
    pub _70: &'static NondetRegLayout,
    pub _71: &'static NondetRegLayout,
    pub _72: &'static NondetRegLayout,
    pub _73: &'static NondetRegLayout,
    pub _74: &'static NondetRegLayout,
    pub _75: &'static NondetRegLayout,
    pub _76: &'static NondetRegLayout,
    pub _77: &'static NondetRegLayout,
    pub _78: &'static NondetRegLayout,
    pub _79: &'static NondetRegLayout,
    pub _80: &'static NondetRegLayout,
    pub _81: &'static NondetRegLayout,
    pub _82: &'static NondetRegLayout,
    pub _83: &'static NondetRegLayout,
    pub _84: &'static NondetRegLayout,
    pub _85: &'static NondetRegLayout,
    pub _86: &'static NondetRegLayout,
    pub _87: &'static NondetRegLayout,
    pub _88: &'static NondetRegLayout,
    pub _89: &'static NondetRegLayout,
    pub d: &'static NondetRegLayout24LayoutArray,
    pub _90: &'static DummyBLayout,
    pub _91: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        v.visit_component("_48", self._48)?;
        v.visit_component("_49", self._49)?;
        v.visit_component("_50", self._50)?;
        v.visit_component("_51", self._51)?;
        v.visit_component("_52", self._52)?;
        v.visit_component("_53", self._53)?;
        v.visit_component("_54", self._54)?;
        v.visit_component("_55", self._55)?;
        v.visit_component("_56", self._56)?;
        v.visit_component("_57", self._57)?;
        v.visit_component("_58", self._58)?;
        v.visit_component("_59", self._59)?;
        v.visit_component("_60", self._60)?;
        v.visit_component("_61", self._61)?;
        v.visit_component("_62", self._62)?;
        v.visit_component("_63", self._63)?;
        v.visit_component("_64", self._64)?;
        v.visit_component("_65", self._65)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_66", self._66)?;
        v.visit_component("_67", self._67)?;
        v.visit_component("_68", self._68)?;
        v.visit_component("_69", self._69)?;
        v.visit_component("_70", self._70)?;
        v.visit_component("_71", self._71)?;
        v.visit_component("_72", self._72)?;
        v.visit_component("_73", self._73)?;
        v.visit_component("_74", self._74)?;
        v.visit_component("_75", self._75)?;
        v.visit_component("_76", self._76)?;
        v.visit_component("_77", self._77)?;
        v.visit_component("_78", self._78)?;
        v.visit_component("_79", self._79)?;
        v.visit_component("_80", self._80)?;
        v.visit_component("_81", self._81)?;
        v.visit_component("_82", self._82)?;
        v.visit_component("_83", self._83)?;
        v.visit_component("_84", self._84)?;
        v.visit_component("_85", self._85)?;
        v.visit_component("_86", self._86)?;
        v.visit_component("_87", self._87)?;
        v.visit_component("_88", self._88)?;
        v.visit_component("_89", self._89)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_90", self._90)?;
        v.visit_component("_91", self._91)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_1__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_1__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_1__SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMajor_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor0_0__1_Layout,
    pub arm1: &'static RhoMinor1_0__1_Layout,
    pub arm2: &'static RhoMinor2_0__1_Layout,
    pub arm3: &'static RhoMinor0_2__1_Layout,
    pub arm4: &'static RhoMinor1_2__1_Layout,
    pub arm5: &'static RhoMinor2_2__1_Layout,
    pub arm6: &'static RhoMinor0_4__1_Layout,
    pub arm7: &'static RhoMinor1_4__1_Layout,
    pub arm8: &'static RhoMinor3_1__SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMajor_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct RhoMajor_1_Layout {
    pub _super: &'static RhoMajor_1__SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__7_Layout,
}
impl risc0_zkp::layout::Component for RhoMajor_1_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct RhoMinor0_0__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_0__2__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_0__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__2_Layout {
    pub _super: &'static RhoMinor1_0__2__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_0__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_2__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_2__2__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_2__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__2_Layout {
    pub _super: &'static RhoMinor1_2__2__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_2__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_4__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_4__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_4__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_4__2__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_4__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__2_Layout {
    pub _super: &'static RhoMinor1_4__2__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _1: &'static NondetRegLayout3LayoutArray,
    pub _3: &'static NondetRegLayout3LayoutArray,
    pub _4: &'static Concatenate_3__3_Layout,
    pub _6: &'static NondetRegLayout3LayoutArray,
    pub _7: &'static Concatenate_6__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub _10: &'static Concatenate_9__3_Layout,
    pub _12: &'static NondetRegLayout3LayoutArray,
    pub rs1: &'static Concatenate_12__3_Layout,
    pub _14: &'static NondetRegLayout7LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _15: &'static RGZ_22_Layout,
    pub _16: &'static RGZ_22_Layout,
    pub _17: &'static RGZ_24_Layout,
    pub _18: &'static DummyBLayout,
    pub _19: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _17: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout1LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _18: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
    pub _21: &'static RGZ_24_Layout,
    pub _22: &'static DummyBLayout,
    pub _23: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static NondetRegLayout,
    pub _45: &'static NondetRegLayout,
    pub _46: &'static NondetRegLayout,
    pub _47: &'static NondetRegLayout,
    pub _48: &'static NondetRegLayout,
    pub _49: &'static NondetRegLayout,
    pub _50: &'static NondetRegLayout,
    pub _51: &'static NondetRegLayout,
    pub _52: &'static NondetRegLayout,
    pub _53: &'static NondetRegLayout,
    pub _54: &'static NondetRegLayout,
    pub _55: &'static NondetRegLayout,
    pub _56: &'static NondetRegLayout,
    pub _57: &'static NondetRegLayout,
    pub _58: &'static NondetRegLayout,
    pub _59: &'static NondetRegLayout,
    pub _60: &'static NondetRegLayout,
    pub _61: &'static NondetRegLayout,
    pub _62: &'static NondetRegLayout,
    pub _63: &'static NondetRegLayout,
    pub _64: &'static NondetRegLayout,
    pub _65: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _66: &'static NondetRegLayout,
    pub _67: &'static NondetRegLayout,
    pub _68: &'static NondetRegLayout,
    pub _69: &'static NondetRegLayout,
    pub _70: &'static NondetRegLayout,
    pub _71: &'static NondetRegLayout,
    pub _72: &'static NondetRegLayout,
    pub _73: &'static NondetRegLayout,
    pub _74: &'static NondetRegLayout,
    pub _75: &'static NondetRegLayout,
    pub _76: &'static NondetRegLayout,
    pub _77: &'static NondetRegLayout,
    pub _78: &'static NondetRegLayout,
    pub _79: &'static NondetRegLayout,
    pub _80: &'static NondetRegLayout,
    pub _81: &'static NondetRegLayout,
    pub _82: &'static NondetRegLayout,
    pub _83: &'static NondetRegLayout,
    pub _84: &'static NondetRegLayout,
    pub _85: &'static NondetRegLayout,
    pub _86: &'static NondetRegLayout,
    pub _87: &'static NondetRegLayout,
    pub _88: &'static NondetRegLayout,
    pub _89: &'static NondetRegLayout,
    pub d: &'static NondetRegLayout24LayoutArray,
    pub _90: &'static DummyBLayout,
    pub _91: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        v.visit_component("_48", self._48)?;
        v.visit_component("_49", self._49)?;
        v.visit_component("_50", self._50)?;
        v.visit_component("_51", self._51)?;
        v.visit_component("_52", self._52)?;
        v.visit_component("_53", self._53)?;
        v.visit_component("_54", self._54)?;
        v.visit_component("_55", self._55)?;
        v.visit_component("_56", self._56)?;
        v.visit_component("_57", self._57)?;
        v.visit_component("_58", self._58)?;
        v.visit_component("_59", self._59)?;
        v.visit_component("_60", self._60)?;
        v.visit_component("_61", self._61)?;
        v.visit_component("_62", self._62)?;
        v.visit_component("_63", self._63)?;
        v.visit_component("_64", self._64)?;
        v.visit_component("_65", self._65)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_66", self._66)?;
        v.visit_component("_67", self._67)?;
        v.visit_component("_68", self._68)?;
        v.visit_component("_69", self._69)?;
        v.visit_component("_70", self._70)?;
        v.visit_component("_71", self._71)?;
        v.visit_component("_72", self._72)?;
        v.visit_component("_73", self._73)?;
        v.visit_component("_74", self._74)?;
        v.visit_component("_75", self._75)?;
        v.visit_component("_76", self._76)?;
        v.visit_component("_77", self._77)?;
        v.visit_component("_78", self._78)?;
        v.visit_component("_79", self._79)?;
        v.visit_component("_80", self._80)?;
        v.visit_component("_81", self._81)?;
        v.visit_component("_82", self._82)?;
        v.visit_component("_83", self._83)?;
        v.visit_component("_84", self._84)?;
        v.visit_component("_85", self._85)?;
        v.visit_component("_86", self._86)?;
        v.visit_component("_87", self._87)?;
        v.visit_component("_88", self._88)?;
        v.visit_component("_89", self._89)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_90", self._90)?;
        v.visit_component("_91", self._91)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_2__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_2__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_2__SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMajor_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor0_0__2_Layout,
    pub arm1: &'static RhoMinor1_0__2_Layout,
    pub arm2: &'static RhoMinor2_0__2_Layout,
    pub arm3: &'static RhoMinor0_2__2_Layout,
    pub arm4: &'static RhoMinor1_2__2_Layout,
    pub arm5: &'static RhoMinor2_2__2_Layout,
    pub arm6: &'static RhoMinor0_4__2_Layout,
    pub arm7: &'static RhoMinor1_4__2_Layout,
    pub arm8: &'static RhoMinor3_2__SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMajor_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct RhoMajor_2_Layout {
    pub _super: &'static RhoMajor_2__SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__7_Layout,
}
impl risc0_zkp::layout::Component for RhoMajor_2_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct RhoMinor0_0__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_0__3__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_0__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__3_Layout {
    pub _super: &'static RhoMinor1_0__3__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_0__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_2__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_2__3__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_2__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__3_Layout {
    pub _super: &'static RhoMinor1_2__3__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_2__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_4__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_4__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_4__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_4__3__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_4__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__3_Layout {
    pub _super: &'static RhoMinor1_4__3__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _1: &'static NondetRegLayout3LayoutArray,
    pub _3: &'static NondetRegLayout3LayoutArray,
    pub _4: &'static Concatenate_3__3_Layout,
    pub _6: &'static NondetRegLayout3LayoutArray,
    pub _7: &'static Concatenate_6__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub _10: &'static Concatenate_9__3_Layout,
    pub _12: &'static NondetRegLayout3LayoutArray,
    pub rs1: &'static Concatenate_12__3_Layout,
    pub _14: &'static NondetRegLayout7LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _15: &'static RGZ_22_Layout,
    pub _16: &'static RGZ_22_Layout,
    pub _17: &'static RGZ_24_Layout,
    pub _18: &'static DummyBLayout,
    pub _19: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _17: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout1LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _18: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
    pub _21: &'static RGZ_24_Layout,
    pub _22: &'static DummyBLayout,
    pub _23: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static NondetRegLayout,
    pub _45: &'static NondetRegLayout,
    pub _46: &'static NondetRegLayout,
    pub _47: &'static NondetRegLayout,
    pub _48: &'static NondetRegLayout,
    pub _49: &'static NondetRegLayout,
    pub _50: &'static NondetRegLayout,
    pub _51: &'static NondetRegLayout,
    pub _52: &'static NondetRegLayout,
    pub _53: &'static NondetRegLayout,
    pub _54: &'static NondetRegLayout,
    pub _55: &'static NondetRegLayout,
    pub _56: &'static NondetRegLayout,
    pub _57: &'static NondetRegLayout,
    pub _58: &'static NondetRegLayout,
    pub _59: &'static NondetRegLayout,
    pub _60: &'static NondetRegLayout,
    pub _61: &'static NondetRegLayout,
    pub _62: &'static NondetRegLayout,
    pub _63: &'static NondetRegLayout,
    pub _64: &'static NondetRegLayout,
    pub _65: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _66: &'static NondetRegLayout,
    pub _67: &'static NondetRegLayout,
    pub _68: &'static NondetRegLayout,
    pub _69: &'static NondetRegLayout,
    pub _70: &'static NondetRegLayout,
    pub _71: &'static NondetRegLayout,
    pub _72: &'static NondetRegLayout,
    pub _73: &'static NondetRegLayout,
    pub _74: &'static NondetRegLayout,
    pub _75: &'static NondetRegLayout,
    pub _76: &'static NondetRegLayout,
    pub _77: &'static NondetRegLayout,
    pub _78: &'static NondetRegLayout,
    pub _79: &'static NondetRegLayout,
    pub _80: &'static NondetRegLayout,
    pub _81: &'static NondetRegLayout,
    pub _82: &'static NondetRegLayout,
    pub _83: &'static NondetRegLayout,
    pub _84: &'static NondetRegLayout,
    pub _85: &'static NondetRegLayout,
    pub _86: &'static NondetRegLayout,
    pub _87: &'static NondetRegLayout,
    pub _88: &'static NondetRegLayout,
    pub _89: &'static NondetRegLayout,
    pub d: &'static NondetRegLayout24LayoutArray,
    pub _90: &'static DummyBLayout,
    pub _91: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        v.visit_component("_48", self._48)?;
        v.visit_component("_49", self._49)?;
        v.visit_component("_50", self._50)?;
        v.visit_component("_51", self._51)?;
        v.visit_component("_52", self._52)?;
        v.visit_component("_53", self._53)?;
        v.visit_component("_54", self._54)?;
        v.visit_component("_55", self._55)?;
        v.visit_component("_56", self._56)?;
        v.visit_component("_57", self._57)?;
        v.visit_component("_58", self._58)?;
        v.visit_component("_59", self._59)?;
        v.visit_component("_60", self._60)?;
        v.visit_component("_61", self._61)?;
        v.visit_component("_62", self._62)?;
        v.visit_component("_63", self._63)?;
        v.visit_component("_64", self._64)?;
        v.visit_component("_65", self._65)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_66", self._66)?;
        v.visit_component("_67", self._67)?;
        v.visit_component("_68", self._68)?;
        v.visit_component("_69", self._69)?;
        v.visit_component("_70", self._70)?;
        v.visit_component("_71", self._71)?;
        v.visit_component("_72", self._72)?;
        v.visit_component("_73", self._73)?;
        v.visit_component("_74", self._74)?;
        v.visit_component("_75", self._75)?;
        v.visit_component("_76", self._76)?;
        v.visit_component("_77", self._77)?;
        v.visit_component("_78", self._78)?;
        v.visit_component("_79", self._79)?;
        v.visit_component("_80", self._80)?;
        v.visit_component("_81", self._81)?;
        v.visit_component("_82", self._82)?;
        v.visit_component("_83", self._83)?;
        v.visit_component("_84", self._84)?;
        v.visit_component("_85", self._85)?;
        v.visit_component("_86", self._86)?;
        v.visit_component("_87", self._87)?;
        v.visit_component("_88", self._88)?;
        v.visit_component("_89", self._89)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_90", self._90)?;
        v.visit_component("_91", self._91)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_3__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_3__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_3__SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMajor_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor0_0__3_Layout,
    pub arm1: &'static RhoMinor1_0__3_Layout,
    pub arm2: &'static RhoMinor2_0__3_Layout,
    pub arm3: &'static RhoMinor0_2__3_Layout,
    pub arm4: &'static RhoMinor1_2__3_Layout,
    pub arm5: &'static RhoMinor2_2__3_Layout,
    pub arm6: &'static RhoMinor0_4__3_Layout,
    pub arm7: &'static RhoMinor1_4__3_Layout,
    pub arm8: &'static RhoMinor3_3__SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMajor_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct RhoMajor_3_Layout {
    pub _super: &'static RhoMajor_3__SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__7_Layout,
}
impl risc0_zkp::layout::Component for RhoMajor_3_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct RhoMinor0_0__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_0__4__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_0__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_0__4_Layout {
    pub _super: &'static RhoMinor1_0__4__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_0__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_2__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_2__4__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_2__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_2__4_Layout {
    pub _super: &'static RhoMinor1_2__4__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor2_2__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
    pub padded_p2: &'static ConcatZeros_21__1_Layout,
    pub padded_p3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor2_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor2_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        v.visit_component("padded_p2", self.padded_p2)?;
        v.visit_component("padded_p3", self.padded_p3)?;
        Ok(())
    }
}
pub struct RhoMinor0_4__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
    pub unpack_p1: &'static ToBitsU_22_Layout,
    pub unpack_p2: &'static ToBitsU_21_Layout,
    pub unpack_p3: &'static ToBitsU_21_Layout,
    pub rotp: &'static RotLeft3impl_22__21__21__1_Layout,
    pub pad_rb: &'static ConcatZeros_21__1_Layout,
    pub pad_rc: &'static ConcatZeros_21__1_Layout,
    pub padded_m1: &'static ConcatZeros_22__2_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor0_4__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor0_4__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        v.visit_component("unpack_p1", self.unpack_p1)?;
        v.visit_component("unpack_p2", self.unpack_p2)?;
        v.visit_component("unpack_p3", self.unpack_p3)?;
        v.visit_component("rotp", self.rotp)?;
        v.visit_component("pad_rb", self.pad_rb)?;
        v.visit_component("pad_rc", self.pad_rc)?;
        v.visit_component("padded_m1", self.padded_m1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_m1: &'static ToBitsU_22_Layout,
    pub unpack_m2: &'static ToBitsU_21_Layout,
    pub padded_m2: &'static ConcatZeros_21__1_Layout,
    pub unpack_m3: &'static ToBitsU_21_Layout,
    pub padded_m3: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_m1", self.unpack_m1)?;
        v.visit_component("unpack_m2", self.unpack_m2)?;
        v.visit_component("padded_m2", self.padded_m2)?;
        v.visit_component("unpack_m3", self.unpack_m3)?;
        v.visit_component("padded_m3", self.padded_m3)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor1_4__4__SuperArm0Layout,
    pub arm1: &'static RhoMinor1_4__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor1_4__4_Layout {
    pub _super: &'static RhoMinor1_4__4__SuperLayout,
    pub unpack_xy1: &'static ToBitsU_22_Layout,
    pub unpack_xy2: &'static ToBitsU_21_Layout,
    pub unpack_xy3: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout3LayoutArray,
    pub withrs_d: &'static Concatenate_21__3_Layout,
    pub padded_xy2: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for RhoMinor1_4__4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor1_4__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_xy1", self.unpack_xy1)?;
        v.visit_component("unpack_xy2", self.unpack_xy2)?;
        v.visit_component("unpack_xy3", self.unpack_xy3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("padded_xy2", self.padded_xy2)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _1: &'static NondetRegLayout3LayoutArray,
    pub _3: &'static NondetRegLayout3LayoutArray,
    pub _4: &'static Concatenate_3__3_Layout,
    pub _6: &'static NondetRegLayout3LayoutArray,
    pub _7: &'static Concatenate_6__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub _10: &'static Concatenate_9__3_Layout,
    pub _12: &'static NondetRegLayout3LayoutArray,
    pub rs1: &'static Concatenate_12__3_Layout,
    pub _14: &'static NondetRegLayout7LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _15: &'static RGZ_22_Layout,
    pub _16: &'static RGZ_22_Layout,
    pub _17: &'static RGZ_24_Layout,
    pub _18: &'static DummyBLayout,
    pub _19: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _17: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout1LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _18: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
    pub _21: &'static RGZ_24_Layout,
    pub _22: &'static DummyBLayout,
    pub _23: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static NondetRegLayout,
    pub _45: &'static NondetRegLayout,
    pub _46: &'static NondetRegLayout,
    pub _47: &'static NondetRegLayout,
    pub _48: &'static NondetRegLayout,
    pub _49: &'static NondetRegLayout,
    pub _50: &'static NondetRegLayout,
    pub _51: &'static NondetRegLayout,
    pub _52: &'static NondetRegLayout,
    pub _53: &'static NondetRegLayout,
    pub _54: &'static NondetRegLayout,
    pub _55: &'static NondetRegLayout,
    pub _56: &'static NondetRegLayout,
    pub _57: &'static NondetRegLayout,
    pub _58: &'static NondetRegLayout,
    pub _59: &'static NondetRegLayout,
    pub _60: &'static NondetRegLayout,
    pub _61: &'static NondetRegLayout,
    pub _62: &'static NondetRegLayout,
    pub _63: &'static NondetRegLayout,
    pub _64: &'static NondetRegLayout,
    pub _65: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _66: &'static NondetRegLayout,
    pub _67: &'static NondetRegLayout,
    pub _68: &'static NondetRegLayout,
    pub _69: &'static NondetRegLayout,
    pub _70: &'static NondetRegLayout,
    pub _71: &'static NondetRegLayout,
    pub _72: &'static NondetRegLayout,
    pub _73: &'static NondetRegLayout,
    pub _74: &'static NondetRegLayout,
    pub _75: &'static NondetRegLayout,
    pub _76: &'static NondetRegLayout,
    pub _77: &'static NondetRegLayout,
    pub _78: &'static NondetRegLayout,
    pub _79: &'static NondetRegLayout,
    pub _80: &'static NondetRegLayout,
    pub _81: &'static NondetRegLayout,
    pub _82: &'static NondetRegLayout,
    pub _83: &'static NondetRegLayout,
    pub _84: &'static NondetRegLayout,
    pub _85: &'static NondetRegLayout,
    pub _86: &'static NondetRegLayout,
    pub _87: &'static NondetRegLayout,
    pub _88: &'static NondetRegLayout,
    pub _89: &'static NondetRegLayout,
    pub d: &'static NondetRegLayout24LayoutArray,
    pub _90: &'static DummyBLayout,
    pub _91: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        v.visit_component("_48", self._48)?;
        v.visit_component("_49", self._49)?;
        v.visit_component("_50", self._50)?;
        v.visit_component("_51", self._51)?;
        v.visit_component("_52", self._52)?;
        v.visit_component("_53", self._53)?;
        v.visit_component("_54", self._54)?;
        v.visit_component("_55", self._55)?;
        v.visit_component("_56", self._56)?;
        v.visit_component("_57", self._57)?;
        v.visit_component("_58", self._58)?;
        v.visit_component("_59", self._59)?;
        v.visit_component("_60", self._60)?;
        v.visit_component("_61", self._61)?;
        v.visit_component("_62", self._62)?;
        v.visit_component("_63", self._63)?;
        v.visit_component("_64", self._64)?;
        v.visit_component("_65", self._65)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_66", self._66)?;
        v.visit_component("_67", self._67)?;
        v.visit_component("_68", self._68)?;
        v.visit_component("_69", self._69)?;
        v.visit_component("_70", self._70)?;
        v.visit_component("_71", self._71)?;
        v.visit_component("_72", self._72)?;
        v.visit_component("_73", self._73)?;
        v.visit_component("_74", self._74)?;
        v.visit_component("_75", self._75)?;
        v.visit_component("_76", self._76)?;
        v.visit_component("_77", self._77)?;
        v.visit_component("_78", self._78)?;
        v.visit_component("_79", self._79)?;
        v.visit_component("_80", self._80)?;
        v.visit_component("_81", self._81)?;
        v.visit_component("_82", self._82)?;
        v.visit_component("_83", self._83)?;
        v.visit_component("_84", self._84)?;
        v.visit_component("_85", self._85)?;
        v.visit_component("_86", self._86)?;
        v.visit_component("_87", self._87)?;
        v.visit_component("_88", self._88)?;
        v.visit_component("_89", self._89)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_90", self._90)?;
        v.visit_component("_91", self._91)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_4__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_4__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMinor3_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor3_4__SuperArm0_SuperLayout,
    pub arm1: &'static RhoMinor3_4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMinor3_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMinor3_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct RhoMajor_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RhoMinor0_0__4_Layout,
    pub arm1: &'static RhoMinor1_0__4_Layout,
    pub arm2: &'static RhoMinor2_0__4_Layout,
    pub arm3: &'static RhoMinor0_2__4_Layout,
    pub arm4: &'static RhoMinor1_2__4_Layout,
    pub arm5: &'static RhoMinor2_2__4_Layout,
    pub arm6: &'static RhoMinor0_4__4_Layout,
    pub arm7: &'static RhoMinor1_4__4_Layout,
    pub arm8: &'static RhoMinor3_4__SuperLayout,
}
impl risc0_zkp::layout::Component for RhoMajor_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct RhoMajor_4_Layout {
    pub _super: &'static RhoMajor_4__SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__7_Layout,
}
impl risc0_zkp::layout::Component for RhoMajor_4_Layout {
    fn ty_name(&self) -> &'static str {
        "RhoMajor_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ChiMinor0_0__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct DummyELayout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for DummyELayout {
    fn ty_name(&self) -> &'static str {
        "DummyELayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_0__0__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_0__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__0_Layout {
    pub _super: &'static ChiMinor1_0__0__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_1__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_1__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_1__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_2__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_2__0__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_2__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__0_Layout {
    pub _super: &'static ChiMinor1_2__0__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_3__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_3__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_3__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_4__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_4__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_4__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__0__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__0__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__0__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_4__0__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_4__0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__0_Layout {
    pub _super: &'static ChiMinor1_4__0__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static RGZ_22_Layout,
    pub _23: &'static RGZ_22_Layout,
    pub _24: &'static RGZ_24_Layout,
    pub _25: &'static DummyBLayout,
    pub _26: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _18: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _16: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _17: &'static RGZ_22_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct OneHotU_24_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_24_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_24_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_9__13_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_9__13_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_9__13_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout6LayoutArray,
    pub _7: &'static NondetRegLayout16LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout9LayoutArray,
    pub unpack_round: &'static OneHotU_24_Layout,
    pub rnd_1: &'static NondetRegLayout2LayoutArray,
    pub rnd_2: &'static NondetRegLayout22LayoutArray,
    pub tmp_d: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub unpack_a002: &'static ToBitsU_22_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("unpack_round", self.unpack_round)?;
        v.visit_component("rnd_1", self.rnd_1)?;
        v.visit_component("rnd_2", self.rnd_2)?;
        v.visit_component("tmp_d", self.tmp_d)?;
        v.visit_component("d", self.d)?;
        v.visit_component("unpack_a002", self.unpack_a002)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_0__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_0__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_0__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_0__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_0__SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_0__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type NondetRegLayout19LayoutArray = [&'static NondetRegLayout; 19];
pub struct Concatenate_3__19_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_3__19_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_3__19_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct IotaXorRcLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a000: &'static ToBitsU_22_Layout,
    pub unpack_a001: &'static ToBitsU_22_Layout,
    pub inv_new_blk: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub a00: &'static NondetRegLayout3LayoutArray,
    pub _9: &'static NondetRegLayout19LayoutArray,
    pub a: &'static Concatenate_3__19_Layout,
    pub d_suffix: &'static NondetRegLayout2LayoutArray,
    pub _11: &'static NondetRegLayout9LayoutArray,
    pub d_tmp: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub _12: &'static RGZ_22_Layout,
    pub _13: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component for IotaXorRcLayout {
    fn ty_name(&self) -> &'static str {
        "IotaXorRcLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a000", self.unpack_a000)?;
        v.visit_component("unpack_a001", self.unpack_a001)?;
        v.visit_component("inv_new_blk", self.inv_new_blk)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("a00", self.a00)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("a", self.a)?;
        v.visit_component("d_suffix", self.d_suffix)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("d_tmp", self.d_tmp)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        Ok(())
    }
}
pub struct ChiMajor_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor0_0__0_Layout,
    pub arm1: &'static ChiMinor1_0__0_Layout,
    pub arm2: &'static ChiMinor0_1__0_Layout,
    pub arm3: &'static ChiMinor0_2__0_Layout,
    pub arm4: &'static ChiMinor1_2__0_Layout,
    pub arm5: &'static ChiMinor0_3__0_Layout,
    pub arm6: &'static ChiMinor0_4__0_Layout,
    pub arm7: &'static ChiMinor1_4__0_Layout,
    pub arm8: &'static ChiMinor2_0__SuperLayout,
    pub arm9: &'static IotaXorRcLayout,
}
impl risc0_zkp::layout::Component for ChiMajor_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub type NondetRegLayout10LayoutArray = [&'static NondetRegLayout; 10];
pub struct OneHotU_10_Layout {
    pub _super: &'static NondetRegLayout10LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_10_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_10_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_10__6_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_10__6_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_10__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ChiMajor_0_Layout {
    pub _super: &'static ChiMajor_0__SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
}
impl risc0_zkp::layout::Component for ChiMajor_0_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ChiMinor0_0__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_0__1__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_0__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__1_Layout {
    pub _super: &'static ChiMinor1_0__1__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_1__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_1__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_1__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_2__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_2__1__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_2__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__1_Layout {
    pub _super: &'static ChiMinor1_2__1__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_3__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_3__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_3__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_4__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_4__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_4__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_4__1__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_4__1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__1_Layout {
    pub _super: &'static ChiMinor1_4__1__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static RGZ_22_Layout,
    pub _23: &'static RGZ_22_Layout,
    pub _24: &'static RGZ_24_Layout,
    pub _25: &'static DummyBLayout,
    pub _26: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _18: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _16: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _17: &'static RGZ_22_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout6LayoutArray,
    pub _7: &'static NondetRegLayout16LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout9LayoutArray,
    pub unpack_round: &'static OneHotU_24_Layout,
    pub rnd_1: &'static NondetRegLayout2LayoutArray,
    pub rnd_2: &'static NondetRegLayout22LayoutArray,
    pub tmp_d: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub unpack_a002: &'static ToBitsU_22_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("unpack_round", self.unpack_round)?;
        v.visit_component("rnd_1", self.rnd_1)?;
        v.visit_component("rnd_2", self.rnd_2)?;
        v.visit_component("tmp_d", self.tmp_d)?;
        v.visit_component("d", self.d)?;
        v.visit_component("unpack_a002", self.unpack_a002)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_1__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_1__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_1__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_1__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_1__SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_1__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMajor_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor0_0__1_Layout,
    pub arm1: &'static ChiMinor1_0__1_Layout,
    pub arm2: &'static ChiMinor0_1__1_Layout,
    pub arm3: &'static ChiMinor0_2__1_Layout,
    pub arm4: &'static ChiMinor1_2__1_Layout,
    pub arm5: &'static ChiMinor0_3__1_Layout,
    pub arm6: &'static ChiMinor0_4__1_Layout,
    pub arm7: &'static ChiMinor1_4__1_Layout,
    pub arm8: &'static ChiMinor2_1__SuperLayout,
    pub arm9: &'static IotaXorRcLayout,
}
impl risc0_zkp::layout::Component for ChiMajor_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub struct ChiMajor_1_Layout {
    pub _super: &'static ChiMajor_1__SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
}
impl risc0_zkp::layout::Component for ChiMajor_1_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ChiMinor0_0__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_0__2__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_0__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__2_Layout {
    pub _super: &'static ChiMinor1_0__2__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_1__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_1__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_1__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_2__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_2__2__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_2__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__2_Layout {
    pub _super: &'static ChiMinor1_2__2__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_3__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_3__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_3__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_4__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_4__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_4__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_4__2__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_4__2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__2_Layout {
    pub _super: &'static ChiMinor1_4__2__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static RGZ_22_Layout,
    pub _23: &'static RGZ_22_Layout,
    pub _24: &'static RGZ_24_Layout,
    pub _25: &'static DummyBLayout,
    pub _26: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _18: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _16: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _17: &'static RGZ_22_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout6LayoutArray,
    pub _7: &'static NondetRegLayout16LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout9LayoutArray,
    pub unpack_round: &'static OneHotU_24_Layout,
    pub rnd_1: &'static NondetRegLayout2LayoutArray,
    pub rnd_2: &'static NondetRegLayout22LayoutArray,
    pub tmp_d: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub unpack_a002: &'static ToBitsU_22_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("unpack_round", self.unpack_round)?;
        v.visit_component("rnd_1", self.rnd_1)?;
        v.visit_component("rnd_2", self.rnd_2)?;
        v.visit_component("tmp_d", self.tmp_d)?;
        v.visit_component("d", self.d)?;
        v.visit_component("unpack_a002", self.unpack_a002)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_2__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_2__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_2__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_2__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_2__SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_2__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMajor_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor0_0__2_Layout,
    pub arm1: &'static ChiMinor1_0__2_Layout,
    pub arm2: &'static ChiMinor0_1__2_Layout,
    pub arm3: &'static ChiMinor0_2__2_Layout,
    pub arm4: &'static ChiMinor1_2__2_Layout,
    pub arm5: &'static ChiMinor0_3__2_Layout,
    pub arm6: &'static ChiMinor0_4__2_Layout,
    pub arm7: &'static ChiMinor1_4__2_Layout,
    pub arm8: &'static ChiMinor2_2__SuperLayout,
    pub arm9: &'static IotaXorRcLayout,
}
impl risc0_zkp::layout::Component for ChiMajor_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub struct ChiMajor_2_Layout {
    pub _super: &'static ChiMajor_2__SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
}
impl risc0_zkp::layout::Component for ChiMajor_2_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ChiMinor0_0__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_0__3__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_0__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__3_Layout {
    pub _super: &'static ChiMinor1_0__3__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_1__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_1__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_1__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_2__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_2__3__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_2__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__3_Layout {
    pub _super: &'static ChiMinor1_2__3__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_3__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_3__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_3__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_4__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_4__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_4__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_4__3__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_4__3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__3_Layout {
    pub _super: &'static ChiMinor1_4__3__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static RGZ_22_Layout,
    pub _23: &'static RGZ_22_Layout,
    pub _24: &'static RGZ_24_Layout,
    pub _25: &'static DummyBLayout,
    pub _26: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _18: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _16: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _17: &'static RGZ_22_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout6LayoutArray,
    pub _7: &'static NondetRegLayout16LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout9LayoutArray,
    pub unpack_round: &'static OneHotU_24_Layout,
    pub rnd_1: &'static NondetRegLayout2LayoutArray,
    pub rnd_2: &'static NondetRegLayout22LayoutArray,
    pub tmp_d: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub unpack_a002: &'static ToBitsU_22_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("unpack_round", self.unpack_round)?;
        v.visit_component("rnd_1", self.rnd_1)?;
        v.visit_component("rnd_2", self.rnd_2)?;
        v.visit_component("tmp_d", self.tmp_d)?;
        v.visit_component("d", self.d)?;
        v.visit_component("unpack_a002", self.unpack_a002)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_3__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_3__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_3__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_3__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_3__SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_3__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMajor_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor0_0__3_Layout,
    pub arm1: &'static ChiMinor1_0__3_Layout,
    pub arm2: &'static ChiMinor0_1__3_Layout,
    pub arm3: &'static ChiMinor0_2__3_Layout,
    pub arm4: &'static ChiMinor1_2__3_Layout,
    pub arm5: &'static ChiMinor0_3__3_Layout,
    pub arm6: &'static ChiMinor0_4__3_Layout,
    pub arm7: &'static ChiMinor1_4__3_Layout,
    pub arm8: &'static ChiMinor2_3__SuperLayout,
    pub arm9: &'static IotaXorRcLayout,
}
impl risc0_zkp::layout::Component for ChiMajor_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub struct ChiMajor_3_Layout {
    pub _super: &'static ChiMajor_3__SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
}
impl risc0_zkp::layout::Component for ChiMajor_3_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ChiMinor0_0__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_0__4__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_0__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_0__4_Layout {
    pub _super: &'static ChiMinor1_0__4__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_1__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_1__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_1__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_2__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_2__4__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_2__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_2__4_Layout {
    pub _super: &'static ChiMinor1_2__4__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor0_3__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_3__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_3__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor0_4__4_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_0xp1: &'static ToBitsU_22_Layout,
    pub unpack_1xp1: &'static ToBitsU_21_Layout,
    pub unpack_0xp2: &'static ToBitsU_22_Layout,
    pub unpack_1xp2: &'static ToBitsU_21_Layout,
    pub unpack_0x: &'static ToBitsU_22_Layout,
    pub unpack_1x: &'static ToBitsU_21_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub result: &'static NondetRegLayout2LayoutArray,
    pub withrs_d: &'static Concatenate_22__2_Layout,
    pub a: &'static ConcatZeros_21__1_Layout,
    pub c: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor0_4__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor0_4__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_0xp1", self.unpack_0xp1)?;
        v.visit_component("unpack_1xp1", self.unpack_1xp1)?;
        v.visit_component("unpack_0xp2", self.unpack_0xp2)?;
        v.visit_component("unpack_1xp2", self.unpack_1xp2)?;
        v.visit_component("unpack_0x", self.unpack_0x)?;
        v.visit_component("unpack_1x", self.unpack_1x)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("result", self.result)?;
        v.visit_component("withrs_d", self.withrs_d)?;
        v.visit_component("a", self.a)?;
        v.visit_component("c", self.c)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyELayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub unpack_2nx: &'static ToBitsU_21_Layout,
    pub unpack_2nxp1: &'static ToBitsU_21_Layout,
    pub unpack_2nxp2: &'static ToBitsU_21_Layout,
    pub res3n: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout3LayoutArray,
    pub withr_d: &'static Concatenate_21__3_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2nx", self.unpack_2nx)?;
        v.visit_component("unpack_2nxp1", self.unpack_2nxp1)?;
        v.visit_component("unpack_2nxp2", self.unpack_2nxp2)?;
        v.visit_component("res3n", self.res3n)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("withr_d", self.withr_d)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor1_4__4__SuperArm0Layout,
    pub arm1: &'static ChiMinor1_4__4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor1_4__4_Layout {
    pub _super: &'static ChiMinor1_4__4__SuperLayout,
    pub unpack_2x: &'static ToBitsU_21_Layout,
    pub unpack_2xp1: &'static ToBitsU_21_Layout,
    pub unpack_2xp2: &'static ToBitsU_21_Layout,
    pub res3: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub withr_a: &'static ConcatElem_21_Layout,
    pub _1: &'static NondetRegLayout,
    pub withr_b: &'static ConcatElem_21_Layout,
    pub withr_c: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for ChiMinor1_4__4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor1_4__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_2x", self.unpack_2x)?;
        v.visit_component("unpack_2xp1", self.unpack_2xp1)?;
        v.visit_component("unpack_2xp2", self.unpack_2xp2)?;
        v.visit_component("res3", self.res3)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("withr_a", self.withr_a)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("withr_b", self.withr_b)?;
        v.visit_component("withr_c", self.withr_c)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static RGZ_22_Layout,
    pub _23: &'static RGZ_22_Layout,
    pub _24: &'static RGZ_24_Layout,
    pub _25: &'static DummyBLayout,
    pub _26: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout7LayoutArray,
    pub _8: &'static NondetRegLayout15LayoutArray,
    pub a: &'static Concatenate_15__7_Layout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout8LayoutArray,
    pub _18: &'static NondetRegLayout14LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout14LayoutArray,
    pub _15: &'static NondetRegLayout8LayoutArray,
    pub b: &'static Concatenate_8__14_Layout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout1LayoutArray,
    pub _18: &'static NondetRegLayout21LayoutArray,
    pub c: &'static Concatenate_1__21_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_24_Layout,
    pub _21: &'static DummyBLayout,
    pub _22: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("_22", self._22)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout15LayoutArray,
    pub c1: &'static Concatenate_1__15_Layout,
    pub _16: &'static NondetRegLayout6LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _17: &'static RGZ_22_Layout,
    pub _18: &'static RGZ_22_Layout,
    pub _19: &'static RGZ_24_Layout,
    pub _20: &'static DummyBLayout,
    pub _21: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("c1", self.c1)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub rs1: &'static NondetRegLayout6LayoutArray,
    pub _7: &'static NondetRegLayout16LayoutArray,
    pub c: &'static Concatenate_16__6_Layout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub rs2: &'static NondetRegLayout9LayoutArray,
    pub unpack_round: &'static OneHotU_24_Layout,
    pub rnd_1: &'static NondetRegLayout2LayoutArray,
    pub rnd_2: &'static NondetRegLayout22LayoutArray,
    pub tmp_d: &'static ConcatZeros_9__13_Layout,
    pub d: &'static Concatenate_22__2_Layout,
    pub unpack_a002: &'static ToBitsU_22_Layout,
    pub _19: &'static RGZ_22_Layout,
    pub _20: &'static RGZ_22_Layout,
}
impl risc0_zkp::layout::Component
    for ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout
{
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("unpack_round", self.unpack_round)?;
        v.visit_component("rnd_1", self.rnd_1)?;
        v.visit_component("rnd_2", self.rnd_2)?;
        v.visit_component("tmp_d", self.tmp_d)?;
        v.visit_component("d", self.d)?;
        v.visit_component("unpack_a002", self.unpack_a002)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_4__SuperArm1_SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm1_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_4__SuperArm1_SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_4__SuperArm1_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMinor2_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor2_4__SuperArm0_SuperLayout,
    pub arm1: &'static ChiMinor2_4__SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ChiMinor2_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMinor2_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ChiMajor_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ChiMinor0_0__4_Layout,
    pub arm1: &'static ChiMinor1_0__4_Layout,
    pub arm2: &'static ChiMinor0_1__4_Layout,
    pub arm3: &'static ChiMinor0_2__4_Layout,
    pub arm4: &'static ChiMinor1_2__4_Layout,
    pub arm5: &'static ChiMinor0_3__4_Layout,
    pub arm6: &'static ChiMinor0_4__4_Layout,
    pub arm7: &'static ChiMinor1_4__4_Layout,
    pub arm8: &'static ChiMinor2_4__SuperLayout,
    pub arm9: &'static IotaXorRcLayout,
}
impl risc0_zkp::layout::Component for ChiMajor_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub struct ChiMajor_4_Layout {
    pub _super: &'static ChiMajor_4__SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
}
impl risc0_zkp::layout::Component for ChiMajor_4_Layout {
    fn ty_name(&self) -> &'static str {
        "ChiMajor_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        Ok(())
    }
}
pub struct ToBitsU_16_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_16_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_16_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout4LayoutArray = [&'static NondetRegLayout; 4];
pub struct ConcatElem_11_Layout {
    pub _super: &'static NondetRegLayout12LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatElem_11_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatElem_11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_12__4_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_12__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_12__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct UseOnceMemoryElementLayout {
    pub count: &'static NondetRegLayout,
    pub index: &'static NondetRegLayout,
    pub digest0: &'static NondetRegLayout,
    pub digest1: &'static NondetRegLayout,
    pub digest2: &'static NondetRegLayout,
    pub digest3: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for UseOnceMemoryElementLayout {
    fn ty_name(&self) -> &'static str {
        "UseOnceMemoryElementLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("count", self.count)?;
        v.visit_component("index", self.index)?;
        v.visit_component("digest0", self.digest0)?;
        v.visit_component("digest1", self.digest1)?;
        v.visit_component("digest2", self.digest2)?;
        v.visit_component("digest3", self.digest3)?;
        Ok(())
    }
}
pub struct WriteMemoryLayout {
    pub elem: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for WriteMemoryLayout {
    fn ty_name(&self) -> &'static str {
        "WriteMemoryLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("elem", self.elem)?;
        Ok(())
    }
}
pub struct AbsorbWord_0__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_1__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_1__0_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_1__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_2__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_3__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_3__0_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_3__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_4__0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_4__0_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_4__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_0__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_1__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_1__1_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_1__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_2__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_3__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_3__1_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_3__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct ConcatZeros_11__5_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_11__5_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_11__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct AbsorbCopy1Layout {
    pub _super: &'static RetTupleLayout,
    pub auxr: &'static ConcatZeros_11__5_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub a: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static RGZ_22_Layout,
    pub _45: &'static RGZ_24_Layout,
    pub _46: &'static DummyBLayout,
    pub _47: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for AbsorbCopy1Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbCopy1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("a", self.a)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        Ok(())
    }
}
pub struct AbsorbMajor1_SuperArm9Layout {
    pub _super: &'static AbsorbCopy1Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for AbsorbMajor1_SuperArm9Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor1_SuperArm9Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct AbsorbMajor1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static AbsorbWord_0__0_Layout,
    pub arm1: &'static AbsorbWord_1__0_Layout,
    pub arm2: &'static AbsorbWord_2__0_Layout,
    pub arm3: &'static AbsorbWord_3__0_Layout,
    pub arm4: &'static AbsorbWord_4__0_Layout,
    pub arm5: &'static AbsorbWord_0__1_Layout,
    pub arm6: &'static AbsorbWord_1__1_Layout,
    pub arm7: &'static AbsorbWord_2__1_Layout,
    pub arm8: &'static AbsorbWord_3__1_Layout,
    pub arm9: &'static AbsorbMajor1_SuperArm9Layout,
}
impl risc0_zkp::layout::Component for AbsorbMajor1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub type NondetRegLayout11LayoutArray = [&'static NondetRegLayout; 11];
pub struct ConcatZeros_10__1_Layout {
    pub _super: &'static NondetRegLayout11LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_10__1_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_10__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type UseOnceMemoryElementLayout1LayoutArray = [&'static UseOnceMemoryElementLayout; 1];
pub struct _ArgumentsAbsorbMajor1_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsAbsorbMajor1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsAbsorbMajor1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct AbsorbMajor1Layout {
    pub _super: &'static AbsorbMajor1_SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__1_Layout,
    pub _arguments_absorb_major1__super: &'static _ArgumentsAbsorbMajor1_SuperLayout,
}
impl risc0_zkp::layout::Component for AbsorbMajor1Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        v.visit_component(
            "_arguments_absorb_major1__super",
            self._arguments_absorb_major1__super,
        )?;
        Ok(())
    }
}
pub struct AbsorbWord_4__1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_4__1_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_4__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_0__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_1__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_1__2_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_1__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_2__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_3__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_3__2_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_3__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_4__2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_4__2_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_4__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_0__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbWord_1__3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub padded23: &'static ConcatElem_21_Layout,
    pub _12: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub results: &'static NondetRegLayout3LayoutArray,
    pub withr34: &'static Concatenate_21__3_Layout,
    pub padded_e: &'static ConcatZeros_21__1_Layout,
    pub padded_f: &'static ConcatZeros_21__1_Layout,
}
impl risc0_zkp::layout::Component for AbsorbWord_1__3_Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbWord_1__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("padded23", self.padded23)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("results", self.results)?;
        v.visit_component("withr34", self.withr34)?;
        v.visit_component("padded_e", self.padded_e)?;
        v.visit_component("padded_f", self.padded_f)?;
        Ok(())
    }
}
pub struct AbsorbCopy2Layout {
    pub _super: &'static RetTupleLayout,
    pub auxr: &'static ConcatZeros_11__5_Layout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub _8: &'static NondetRegLayout,
    pub _9: &'static NondetRegLayout,
    pub _10: &'static NondetRegLayout,
    pub _11: &'static NondetRegLayout,
    pub _12: &'static NondetRegLayout,
    pub _13: &'static NondetRegLayout,
    pub _14: &'static NondetRegLayout,
    pub _15: &'static NondetRegLayout,
    pub _16: &'static NondetRegLayout,
    pub _17: &'static NondetRegLayout,
    pub _18: &'static NondetRegLayout,
    pub _19: &'static NondetRegLayout,
    pub _20: &'static NondetRegLayout,
    pub _21: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout22LayoutArray,
    pub _22: &'static NondetRegLayout,
    pub _23: &'static NondetRegLayout,
    pub _24: &'static NondetRegLayout,
    pub _25: &'static NondetRegLayout,
    pub _26: &'static NondetRegLayout,
    pub _27: &'static NondetRegLayout,
    pub _28: &'static NondetRegLayout,
    pub _29: &'static NondetRegLayout,
    pub _30: &'static NondetRegLayout,
    pub _31: &'static NondetRegLayout,
    pub _32: &'static NondetRegLayout,
    pub _33: &'static NondetRegLayout,
    pub _34: &'static NondetRegLayout,
    pub _35: &'static NondetRegLayout,
    pub _36: &'static NondetRegLayout,
    pub _37: &'static NondetRegLayout,
    pub _38: &'static NondetRegLayout,
    pub _39: &'static NondetRegLayout,
    pub _40: &'static NondetRegLayout,
    pub _41: &'static NondetRegLayout,
    pub _42: &'static NondetRegLayout,
    pub _43: &'static NondetRegLayout,
    pub c: &'static NondetRegLayout22LayoutArray,
    pub _44: &'static RGZ_22_Layout,
    pub _45: &'static RGZ_24_Layout,
    pub _46: &'static DummyBLayout,
    pub _47: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for AbsorbCopy2Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbCopy2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("_10", self._10)?;
        v.visit_component("_11", self._11)?;
        v.visit_component("_12", self._12)?;
        v.visit_component("_13", self._13)?;
        v.visit_component("_14", self._14)?;
        v.visit_component("_15", self._15)?;
        v.visit_component("_16", self._16)?;
        v.visit_component("_17", self._17)?;
        v.visit_component("_18", self._18)?;
        v.visit_component("_19", self._19)?;
        v.visit_component("_20", self._20)?;
        v.visit_component("_21", self._21)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_22", self._22)?;
        v.visit_component("_23", self._23)?;
        v.visit_component("_24", self._24)?;
        v.visit_component("_25", self._25)?;
        v.visit_component("_26", self._26)?;
        v.visit_component("_27", self._27)?;
        v.visit_component("_28", self._28)?;
        v.visit_component("_29", self._29)?;
        v.visit_component("_30", self._30)?;
        v.visit_component("_31", self._31)?;
        v.visit_component("_32", self._32)?;
        v.visit_component("_33", self._33)?;
        v.visit_component("_34", self._34)?;
        v.visit_component("_35", self._35)?;
        v.visit_component("_36", self._36)?;
        v.visit_component("_37", self._37)?;
        v.visit_component("_38", self._38)?;
        v.visit_component("_39", self._39)?;
        v.visit_component("_40", self._40)?;
        v.visit_component("_41", self._41)?;
        v.visit_component("_42", self._42)?;
        v.visit_component("_43", self._43)?;
        v.visit_component("c", self.c)?;
        v.visit_component("_44", self._44)?;
        v.visit_component("_45", self._45)?;
        v.visit_component("_46", self._46)?;
        v.visit_component("_47", self._47)?;
        Ok(())
    }
}
pub struct AbsorbMajor2_SuperArm8Layout {
    pub _super: &'static AbsorbCopy2Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for AbsorbMajor2_SuperArm8Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor2_SuperArm8Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct AbsorbMajor2_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static AbsorbWord_4__1_Layout,
    pub arm1: &'static AbsorbWord_0__2_Layout,
    pub arm2: &'static AbsorbWord_1__2_Layout,
    pub arm3: &'static AbsorbWord_2__2_Layout,
    pub arm4: &'static AbsorbWord_3__2_Layout,
    pub arm5: &'static AbsorbWord_4__2_Layout,
    pub arm6: &'static AbsorbWord_0__3_Layout,
    pub arm7: &'static AbsorbWord_1__3_Layout,
    pub arm8: &'static AbsorbMajor2_SuperArm8Layout,
}
impl risc0_zkp::layout::Component for AbsorbMajor2_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor2_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        Ok(())
    }
}
pub struct ConcatZeros_9__2_Layout {
    pub _super: &'static NondetRegLayout11LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_9__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_9__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct _ArgumentsAbsorbMajor2_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsAbsorbMajor2_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsAbsorbMajor2_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct AbsorbMajor2Layout {
    pub _super: &'static AbsorbMajor2_SuperLayout,
    pub minor_onehot: &'static OneHotU_9_Layout,
    pub aux_regs: &'static ConcatZeros_9__2_Layout,
    pub _arguments_absorb_major2__super: &'static _ArgumentsAbsorbMajor2_SuperLayout,
}
impl risc0_zkp::layout::Component for AbsorbMajor2Layout {
    fn ty_name(&self) -> &'static str {
        "AbsorbMajor2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        v.visit_component(
            "_arguments_absorb_major2__super",
            self._arguments_absorb_major2__super,
        )?;
        Ok(())
    }
}
pub struct MemPaddingSha256_SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256_SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256_SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct MemPaddingSha256_SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256_SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256_SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct MemPaddingSha256_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static MemPaddingSha256_SuperArm0Layout,
    pub arm1: &'static MemPaddingSha256_SuperArm1Layout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type NondetRegLayout30LayoutArray = [&'static NondetRegLayout; 30];
pub struct ToBitsU_30_Layout {
    pub _super: &'static NondetRegLayout30LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_30_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_30_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout13LayoutArray = [&'static NondetRegLayout; 13];
pub struct Concatenate_9__4_Layout {
    pub _super: &'static NondetRegLayout13LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_9__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_9__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_13__6_Layout {
    pub _super: &'static NondetRegLayout19LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_13__6_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_13__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_19__3_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_19__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_19__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_3__21_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_3__21_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_3__21_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct _ArgumentsMemPaddingSha256__10Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsMemPaddingSha256__10Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsMemPaddingSha256__10Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__10Arm0_SuperLayout {
    pub _0: &'static WriteMemoryLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__10Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__10Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__10Arm1Layout {
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__10Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__10Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__10Layout {
    pub arm0: &'static MemPaddingSha256__10Arm0_SuperLayout,
    pub arm1: &'static MemPaddingSha256__10Arm1Layout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__10Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__10Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsMemPaddingSha256__13Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsMemPaddingSha256__13Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsMemPaddingSha256__13Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__13Arm0Layout {
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__13Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__13Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__13Arm1_SuperLayout {
    pub _0: &'static WriteMemoryLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__13Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__13Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct MemPaddingSha256__13Layout {
    pub arm0: &'static MemPaddingSha256__13Arm0Layout,
    pub arm1: &'static MemPaddingSha256__13Arm1_SuperLayout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256__13Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256__13Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct MemPaddingSha256Layout {
    pub _super: &'static MemPaddingSha256_SuperLayout,
    pub unpack_bitlen: &'static ToBitsU_30_Layout,
    pub b12: &'static NondetRegLayout,
    pub b34: &'static NondetRegLayout,
    pub memidx: &'static NondetRegLayout,
    pub invn: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub a_tmp: &'static NondetRegLayout21LayoutArray,
    pub a: &'static ConcatElem_21_Layout,
    pub b_tmp: &'static NondetRegLayout9LayoutArray,
    pub zdigest: &'static NondetRegLayout4LayoutArray,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub vdigest: &'static NondetRegLayout4LayoutArray,
    pub b_tmp2: &'static Concatenate_9__4_Layout,
    pub b_tmp3: &'static ConcatZeros_13__6_Layout,
    pub b: &'static Concatenate_19__3_Layout,
    pub _9: &'static NondetRegLayout3LayoutArray,
    pub d: &'static ConcatZeros_3__21_Layout,
    pub _arguments_mem_padding_sha256__10: &'static _ArgumentsMemPaddingSha256__10Layout,
    pub _12: &'static MemPaddingSha256__10Layout,
    pub _arguments_mem_padding_sha256__13: &'static _ArgumentsMemPaddingSha256__13Layout,
    pub _15: &'static MemPaddingSha256__13Layout,
}
impl risc0_zkp::layout::Component for MemPaddingSha256Layout {
    fn ty_name(&self) -> &'static str {
        "MemPaddingSha256Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_bitlen", self.unpack_bitlen)?;
        v.visit_component("b12", self.b12)?;
        v.visit_component("b34", self.b34)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invn", self.invn)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("a_tmp", self.a_tmp)?;
        v.visit_component("a", self.a)?;
        v.visit_component("b_tmp", self.b_tmp)?;
        v.visit_component("zdigest", self.zdigest)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("vdigest", self.vdigest)?;
        v.visit_component("b_tmp2", self.b_tmp2)?;
        v.visit_component("b_tmp3", self.b_tmp3)?;
        v.visit_component("b", self.b)?;
        v.visit_component("_9", self._9)?;
        v.visit_component("d", self.d)?;
        v.visit_component(
            "_arguments_mem_padding_sha256__10",
            self._arguments_mem_padding_sha256__10,
        )?;
        v.visit_component("_12", self._12)?;
        v.visit_component(
            "_arguments_mem_padding_sha256__13",
            self._arguments_mem_padding_sha256__13,
        )?;
        v.visit_component("_15", self._15)?;
        Ok(())
    }
}
pub struct Sha256init_1__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_1__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_1__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_1__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_1__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_1__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256init_1__SuperArm0Layout,
    pub arm1: &'static Sha256init_1__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Sha256init_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type NondetRegLayout32LayoutArray = [&'static NondetRegLayout; 32];
pub struct Concatenate_16__16_Layout {
    pub _super: &'static NondetRegLayout32LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_16__16_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_16__16_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_10__12_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_10__12_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_10__12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout20LayoutArray = [&'static NondetRegLayout; 20];
pub struct Ae2sLayout {
    pub e: &'static NondetRegLayout22LayoutArray,
    pub f1: &'static NondetRegLayout10LayoutArray,
    pub f2: &'static NondetRegLayout12LayoutArray,
    pub f: &'static Concatenate_10__12_Layout,
    pub c: &'static NondetRegLayout20LayoutArray,
}
impl risc0_zkp::layout::Component for Ae2sLayout {
    fn ty_name(&self) -> &'static str {
        "Ae2sLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("e", self.e)?;
        v.visit_component("f1", self.f1)?;
        v.visit_component("f2", self.f2)?;
        v.visit_component("f", self.f)?;
        v.visit_component("c", self.c)?;
        Ok(())
    }
}
pub struct ConcatZeros_20__2_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_20__2_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_20__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Sha256init_1_Layout {
    pub _super: &'static Sha256init_1__SuperLayout,
    pub low_a: &'static ToBitsU_16_Layout,
    pub high_a: &'static ToBitsU_16_Layout,
    pub unpacked_a: &'static Concatenate_16__16_Layout,
    pub low_e: &'static ToBitsU_16_Layout,
    pub high_e: &'static ToBitsU_16_Layout,
    pub unpacked_e: &'static Concatenate_16__16_Layout,
    pub stae: &'static Ae2sLayout,
    pub padc: &'static ConcatZeros_20__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256init_1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("low_a", self.low_a)?;
        v.visit_component("high_a", self.high_a)?;
        v.visit_component("unpacked_a", self.unpacked_a)?;
        v.visit_component("low_e", self.low_e)?;
        v.visit_component("high_e", self.high_e)?;
        v.visit_component("unpacked_e", self.unpacked_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("padc", self.padc)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm1Layout {
    pub _super: &'static Sha256init_1_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256init_2__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_2__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_2__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_2__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_2__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_2__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256init_2__SuperArm0Layout,
    pub arm1: &'static Sha256init_2__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Sha256init_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256init_2_Layout {
    pub _super: &'static Sha256init_2__SuperLayout,
    pub low_a: &'static ToBitsU_16_Layout,
    pub high_a: &'static ToBitsU_16_Layout,
    pub unpacked_a: &'static Concatenate_16__16_Layout,
    pub low_e: &'static ToBitsU_16_Layout,
    pub high_e: &'static ToBitsU_16_Layout,
    pub unpacked_e: &'static Concatenate_16__16_Layout,
    pub stae: &'static Ae2sLayout,
    pub padc: &'static ConcatZeros_20__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256init_2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("low_a", self.low_a)?;
        v.visit_component("high_a", self.high_a)?;
        v.visit_component("unpacked_a", self.unpacked_a)?;
        v.visit_component("low_e", self.low_e)?;
        v.visit_component("high_e", self.high_e)?;
        v.visit_component("unpacked_e", self.unpacked_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("padc", self.padc)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm2Layout {
    pub _super: &'static Sha256init_2_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm2Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256init_3__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_3__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_3__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_3__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_3__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_3__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256init_3__SuperArm0Layout,
    pub arm1: &'static Sha256init_3__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Sha256init_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256init_3_Layout {
    pub _super: &'static Sha256init_3__SuperLayout,
    pub low_a: &'static ToBitsU_16_Layout,
    pub high_a: &'static ToBitsU_16_Layout,
    pub unpacked_a: &'static Concatenate_16__16_Layout,
    pub low_e: &'static ToBitsU_16_Layout,
    pub high_e: &'static ToBitsU_16_Layout,
    pub unpacked_e: &'static Concatenate_16__16_Layout,
    pub stae: &'static Ae2sLayout,
    pub padc: &'static ConcatZeros_20__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256init_3_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("low_a", self.low_a)?;
        v.visit_component("high_a", self.high_a)?;
        v.visit_component("unpacked_a", self.unpacked_a)?;
        v.visit_component("low_e", self.low_e)?;
        v.visit_component("high_e", self.high_e)?;
        v.visit_component("unpacked_e", self.unpacked_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("padc", self.padc)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm3Layout {
    pub _super: &'static Sha256init_3_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm3Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256init_4__SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_4__SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_4__SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_4__SuperArm1Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyELayout,
}
impl risc0_zkp::layout::Component for Sha256init_4__SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_4__SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        Ok(())
    }
}
pub struct Sha256init_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256init_4__SuperArm0Layout,
    pub arm1: &'static Sha256init_4__SuperArm1Layout,
}
impl risc0_zkp::layout::Component for Sha256init_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256init_4_Layout {
    pub _super: &'static Sha256init_4__SuperLayout,
    pub low_a: &'static ToBitsU_16_Layout,
    pub high_a: &'static ToBitsU_16_Layout,
    pub unpacked_a: &'static Concatenate_16__16_Layout,
    pub low_e: &'static ToBitsU_16_Layout,
    pub high_e: &'static ToBitsU_16_Layout,
    pub unpacked_e: &'static Concatenate_16__16_Layout,
    pub stae: &'static Ae2sLayout,
    pub padc: &'static ConcatZeros_20__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256init_4_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256init_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("low_a", self.low_a)?;
        v.visit_component("high_a", self.high_a)?;
        v.visit_component("unpacked_a", self.unpacked_a)?;
        v.visit_component("low_e", self.low_e)?;
        v.visit_component("high_e", self.high_e)?;
        v.visit_component("unpacked_e", self.unpacked_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("padc", self.padc)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm4Layout {
    pub _super: &'static Sha256init_4_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm4Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256fini_5__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_5__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_5__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_5__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256fini_5__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_5__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_5__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ToBitsU_19_Layout {
    pub _super: &'static NondetRegLayout19LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_19_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_19_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ToBitsU_20_Layout {
    pub _super: &'static NondetRegLayout20LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_20_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_20_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_3__4_Layout {
    pub _super: &'static NondetRegLayout7LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_3__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_3__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct UnpackWithCarryLayout {
    pub _super: &'static Concatenate_16__16_Layout,
    pub low_bits: &'static ToBitsU_19_Layout,
    pub high_bits: &'static ToBitsU_20_Layout,
    pub carrybits: &'static Concatenate_3__4_Layout,
}
impl risc0_zkp::layout::Component for UnpackWithCarryLayout {
    fn ty_name(&self) -> &'static str {
        "UnpackWithCarryLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("low_bits", self.low_bits)?;
        v.visit_component("high_bits", self.high_bits)?;
        v.visit_component("carrybits", self.carrybits)?;
        Ok(())
    }
}
pub struct Concatenate_7__7_Layout {
    pub _super: &'static NondetRegLayout14LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_7__7_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_7__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_14__8_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_14__8_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_14__8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct DummySLayout {
    pub _super: &'static NondetRegLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for DummySLayout {
    fn ty_name(&self) -> &'static str {
        "DummySLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Sha256fini_5_Layout {
    pub _super: &'static Sha256fini_5__SuperLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub tmp14: &'static Concatenate_7__7_Layout,
    pub pada: &'static ConcatZeros_14__8_Layout,
    pub padc: &'static ConcatZeros_20__2_Layout,
    pub inv_more_input: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub d_suffix: &'static NondetRegLayout3LayoutArray,
    pub _2: &'static DummySLayout,
    pub d: &'static Concatenate_21__3_Layout,
    pub _3: &'static DummySLayout,
    pub zb: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for Sha256fini_5_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("tmp14", self.tmp14)?;
        v.visit_component("pada", self.pada)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("inv_more_input", self.inv_more_input)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("d_suffix", self.d_suffix)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("zb", self.zb)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm5Layout {
    pub _super: &'static Sha256fini_5_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm5Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256fini_6__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_6__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_6__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_6__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256fini_6__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_6__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_6__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_6_Layout {
    pub _super: &'static Sha256fini_6__SuperLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub tmp14: &'static Concatenate_7__7_Layout,
    pub pada: &'static ConcatZeros_14__8_Layout,
    pub padc: &'static ConcatZeros_20__2_Layout,
    pub inv_more_input: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub d_suffix: &'static NondetRegLayout3LayoutArray,
    pub _2: &'static DummySLayout,
    pub d: &'static Concatenate_21__3_Layout,
    pub _3: &'static DummySLayout,
    pub zb: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for Sha256fini_6_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("tmp14", self.tmp14)?;
        v.visit_component("pada", self.pada)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("inv_more_input", self.inv_more_input)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("d_suffix", self.d_suffix)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("zb", self.zb)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm6Layout {
    pub _super: &'static Sha256fini_6_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm6Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256fini_7__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_7__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_7__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_7__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256fini_7__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_7__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_7__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_7_Layout {
    pub _super: &'static Sha256fini_7__SuperLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub tmp14: &'static Concatenate_7__7_Layout,
    pub pada: &'static ConcatZeros_14__8_Layout,
    pub padc: &'static ConcatZeros_20__2_Layout,
    pub inv_more_input: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub d_suffix: &'static NondetRegLayout3LayoutArray,
    pub _2: &'static DummySLayout,
    pub d: &'static Concatenate_21__3_Layout,
    pub _3: &'static DummySLayout,
    pub zb: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for Sha256fini_7_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("tmp14", self.tmp14)?;
        v.visit_component("pada", self.pada)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("inv_more_input", self.inv_more_input)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("d_suffix", self.d_suffix)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("zb", self.zb)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm7Layout {
    pub _super: &'static Sha256fini_7_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm7Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256fini_8__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_8__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_8__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_8__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256fini_8__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256fini_8__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_8__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256fini_8_Layout {
    pub _super: &'static Sha256fini_8__SuperLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub tmp14: &'static Concatenate_7__7_Layout,
    pub pada: &'static ConcatZeros_14__8_Layout,
    pub padc: &'static ConcatZeros_20__2_Layout,
    pub inv_more_input: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub d_suffix: &'static NondetRegLayout3LayoutArray,
    pub _2: &'static DummySLayout,
    pub d: &'static Concatenate_21__3_Layout,
    pub _3: &'static DummySLayout,
    pub zb: &'static ConcatElem_21_Layout,
}
impl risc0_zkp::layout::Component for Sha256fini_8_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256fini_8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("tmp14", self.tmp14)?;
        v.visit_component("pada", self.pada)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("inv_more_input", self.inv_more_input)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("d_suffix", self.d_suffix)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("d", self.d)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("zb", self.zb)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm8Layout {
    pub _super: &'static Sha256fini_8_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm8Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm8Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperArm9Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
    pub _3: &'static DummyELayout,
    pub _4: &'static DummyBLayout,
    pub _5: &'static DummyBLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
    pub _extra1: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperArm9Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperArm9Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Sha256setupMajor_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static MemPaddingSha256Layout,
    pub arm1: &'static Sha256setupMajor_SuperArm1Layout,
    pub arm2: &'static Sha256setupMajor_SuperArm2Layout,
    pub arm3: &'static Sha256setupMajor_SuperArm3Layout,
    pub arm4: &'static Sha256setupMajor_SuperArm4Layout,
    pub arm5: &'static Sha256setupMajor_SuperArm5Layout,
    pub arm6: &'static Sha256setupMajor_SuperArm6Layout,
    pub arm7: &'static Sha256setupMajor_SuperArm7Layout,
    pub arm8: &'static Sha256setupMajor_SuperArm8Layout,
    pub arm9: &'static Sha256setupMajor_SuperArm9Layout,
}
impl risc0_zkp::layout::Component for Sha256setupMajor_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajor_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        Ok(())
    }
}
pub type UseOnceMemoryElementLayout2LayoutArray = [&'static UseOnceMemoryElementLayout; 2];
pub struct _ArgumentsSha256setupMajor_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout2LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256setupMajor_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256setupMajor_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256setupMajorLayout {
    pub _super: &'static Sha256setupMajor_SuperLayout,
    pub minor_onehot: &'static OneHotU_10_Layout,
    pub aux_regs: &'static ConcatZeros_10__6_Layout,
    pub _arguments_sha256setup_major__super: &'static _ArgumentsSha256setupMajor_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256setupMajorLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256setupMajorLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("aux_regs", self.aux_regs)?;
        v.visit_component(
            "_arguments_sha256setup_major__super",
            self._arguments_sha256setup_major__super,
        )?;
        Ok(())
    }
}
pub struct EqualWord_0_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub c: &'static ConcatElem_21_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
    pub _14: &'static ConcatZeros_21__3_Layout,
}
impl risc0_zkp::layout::Component for EqualWord_0_Layout {
    fn ty_name(&self) -> &'static str {
        "EqualWord_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("c", self.c)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        v.visit_component("_14", self._14)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm0Layout {
    pub _super: &'static EqualWord_0_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct EqualWord_1_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub c: &'static ConcatElem_21_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
    pub _14: &'static ConcatZeros_21__3_Layout,
}
impl risc0_zkp::layout::Component for EqualWord_1_Layout {
    fn ty_name(&self) -> &'static str {
        "EqualWord_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("c", self.c)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        v.visit_component("_14", self._14)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm1Layout {
    pub _super: &'static EqualWord_1_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm1Layout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct EqualWord_2_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub c: &'static ConcatElem_21_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
    pub _14: &'static ConcatZeros_21__3_Layout,
}
impl risc0_zkp::layout::Component for EqualWord_2_Layout {
    fn ty_name(&self) -> &'static str {
        "EqualWord_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("c", self.c)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        v.visit_component("_14", self._14)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm2Layout {
    pub _super: &'static EqualWord_2_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm2Layout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct EqualWord_3_Layout {
    pub _super: &'static RetTupleLayout,
    pub unpack_a: &'static ToBitsU_22_Layout,
    pub unpack_e: &'static ToBitsU_21_Layout,
    pub unpack_f: &'static ToBitsU_21_Layout,
    pub unpack_1: &'static ToBitsU_16_Layout,
    pub unpack_2: &'static ToBitsU_16_Layout,
    pub unpack_3: &'static ToBitsU_16_Layout,
    pub unpack_4: &'static ToBitsU_16_Layout,
    pub memidx: &'static NondetRegLayout,
    pub invmidx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _4: &'static WriteMemoryLayout,
    pub arr12: &'static NondetRegLayout22LayoutArray,
    pub arr23: &'static NondetRegLayout21LayoutArray,
    pub arr34: &'static NondetRegLayout21LayoutArray,
    pub c: &'static ConcatElem_21_Layout,
    pub e: &'static ConcatZeros_21__1_Layout,
    pub f: &'static ConcatZeros_21__1_Layout,
    pub _14: &'static ConcatZeros_21__3_Layout,
}
impl risc0_zkp::layout::Component for EqualWord_3_Layout {
    fn ty_name(&self) -> &'static str {
        "EqualWord_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("unpack_a", self.unpack_a)?;
        v.visit_component("unpack_e", self.unpack_e)?;
        v.visit_component("unpack_f", self.unpack_f)?;
        v.visit_component("unpack_1", self.unpack_1)?;
        v.visit_component("unpack_2", self.unpack_2)?;
        v.visit_component("unpack_3", self.unpack_3)?;
        v.visit_component("unpack_4", self.unpack_4)?;
        v.visit_component("memidx", self.memidx)?;
        v.visit_component("invmidx", self.invmidx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("arr12", self.arr12)?;
        v.visit_component("arr23", self.arr23)?;
        v.visit_component("arr34", self.arr34)?;
        v.visit_component("c", self.c)?;
        v.visit_component("e", self.e)?;
        v.visit_component("f", self.f)?;
        v.visit_component("_14", self._14)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm3Layout {
    pub _super: &'static EqualWord_3_Layout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm3Layout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct ParseNextBlock_SuperArm0Layout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static DummyBLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
    pub _3: &'static DummyBLayout,
    pub _4: &'static DummyBLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ParseNextBlock_SuperArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ParseNextBlock_SuperArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct ParseNextBlock_SuperArm1_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub _0: &'static WriteMemoryLayout,
    pub _1: &'static DummyBLayout,
    pub _2: &'static DummyBLayout,
    pub _3: &'static DummyBLayout,
    pub _4: &'static DummyBLayout,
    pub _5: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ParseNextBlock_SuperArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ParseNextBlock_SuperArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        Ok(())
    }
}
pub struct ParseNextBlock_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ParseNextBlock_SuperArm0Layout,
    pub arm1: &'static ParseNextBlock_SuperArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ParseNextBlock_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ParseNextBlock_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ToBitsU_3_Layout {
    pub _super: &'static NondetRegLayout3LayoutArray,
}
impl risc0_zkp::layout::Component for ToBitsU_3_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBitsU_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_11__3_Layout {
    pub _super: &'static NondetRegLayout14LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_11__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_11__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_14__6_Layout {
    pub _super: &'static NondetRegLayout20LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_14__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_14__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_20__4_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_20__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_20__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct _ArgumentsParseNextBlock_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsParseNextBlock_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsParseNextBlock_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct ParseNextBlockLayout {
    pub _super: &'static ParseNextBlock_SuperLayout,
    pub memidx_nxt: &'static NondetRegLayout,
    pub invmidx_nxt: &'static NondetRegLayout,
    pub invmidx_nn: &'static NondetRegLayout,
    pub blk: &'static NondetRegLayout,
    pub inv_blk: &'static NondetRegLayout,
    pub prod: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _3: &'static WriteMemoryLayout,
    pub idx2: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
    pub _5: &'static NondetRegLayout,
    pub _6: &'static NondetRegLayout,
    pub _7: &'static NondetRegLayout,
    pub digest2: &'static NondetRegLayout4LayoutArray,
    pub nskip: &'static NondetRegLayout,
    pub unpack_nskip: &'static ToBitsU_3_Layout,
    pub dz: &'static NondetRegLayout11LayoutArray,
    pub d1: &'static Concatenate_11__3_Layout,
    pub _8: &'static NondetRegLayout6LayoutArray,
    pub d2: &'static Concatenate_14__6_Layout,
    pub d: &'static Concatenate_20__4_Layout,
    pub _arguments_parse_next_block__super: &'static _ArgumentsParseNextBlock_SuperLayout,
}
impl risc0_zkp::layout::Component for ParseNextBlockLayout {
    fn ty_name(&self) -> &'static str {
        "ParseNextBlockLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("memidx_nxt", self.memidx_nxt)?;
        v.visit_component("invmidx_nxt", self.invmidx_nxt)?;
        v.visit_component("invmidx_nn", self.invmidx_nn)?;
        v.visit_component("blk", self.blk)?;
        v.visit_component("inv_blk", self.inv_blk)?;
        v.visit_component("prod", self.prod)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("idx2", self.idx2)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("digest2", self.digest2)?;
        v.visit_component("nskip", self.nskip)?;
        v.visit_component("unpack_nskip", self.unpack_nskip)?;
        v.visit_component("dz", self.dz)?;
        v.visit_component("d1", self.d1)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("d2", self.d2)?;
        v.visit_component("d", self.d)?;
        v.visit_component(
            "_arguments_parse_next_block__super",
            self._arguments_parse_next_block__super,
        )?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm5_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub blk: &'static NondetRegLayout,
    pub idx: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub digest: &'static NondetRegLayout4LayoutArray,
    pub auxtmp1: &'static ConcatElem_11_Layout,
    pub auxr: &'static Concatenate_12__4_Layout,
    pub _3: &'static WriteMemoryLayout,
    pub _4: &'static DummyBLayout,
    pub _5: &'static DummyBLayout,
    pub _6: &'static DummyBLayout,
    pub _7: &'static DummyELayout,
    pub _8: &'static DummyBLayout,
    pub _9: &'static DummyBLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm5_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm5_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("blk", self.blk)?;
        v.visit_component("idx", self.idx)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("digest", self.digest)?;
        v.visit_component("auxtmp1", self.auxtmp1)?;
        v.visit_component("auxr", self.auxr)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("_5", self._5)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("_7", self._7)?;
        v.visit_component("_8", self._8)?;
        v.visit_component("_9", self._9)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperArm5Layout {
    pub _super: &'static ExtraMajor_SuperArm5_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperArm5Layout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperArm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct ExtraMajor_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static ExtraMajor_SuperArm0Layout,
    pub arm1: &'static ExtraMajor_SuperArm1Layout,
    pub arm2: &'static ExtraMajor_SuperArm2Layout,
    pub arm3: &'static ExtraMajor_SuperArm3Layout,
    pub arm4: &'static ParseNextBlockLayout,
    pub arm5: &'static ExtraMajor_SuperArm5Layout,
}
impl risc0_zkp::layout::Component for ExtraMajor_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajor_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        Ok(())
    }
}
pub struct OneHotU_6_Layout {
    pub _super: &'static NondetRegLayout6LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_6_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct ConcatZeros_6__5_Layout {
    pub _super: &'static NondetRegLayout11LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_6__5_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_6__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct _ArgumentsExtraMajor_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout2LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsExtraMajor_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsExtraMajor_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct ExtraMajorLayout {
    pub _super: &'static ExtraMajor_SuperLayout,
    pub minor_onehot: &'static OneHotU_6_Layout,
    pub auxregs: &'static ConcatZeros_6__5_Layout,
    pub _arguments_extra_major__super: &'static _ArgumentsExtraMajor_SuperLayout,
}
impl risc0_zkp::layout::Component for ExtraMajorLayout {
    fn ty_name(&self) -> &'static str {
        "ExtraMajorLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("minor_onehot", self.minor_onehot)?;
        v.visit_component("auxregs", self.auxregs)?;
        v.visit_component(
            "_arguments_extra_major__super",
            self._arguments_extra_major__super,
        )?;
        Ok(())
    }
}
pub struct Sha256load_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_0___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_0___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_0___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct ReadMemoryLayout {
    pub elem: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for ReadMemoryLayout {
    fn ty_name(&self) -> &'static str {
        "ReadMemoryLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("elem", self.elem)?;
        Ok(())
    }
}
pub struct Sha256load_0___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_0___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_0___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_0___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_0___0Arm1Layout {
    pub _super: &'static Sha256load_0___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_0___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_0___0Layout {
    pub arm0: &'static Sha256load_0___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_0___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_0___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct W2sLayout {
    pub a: &'static NondetRegLayout22LayoutArray,
    pub b: &'static NondetRegLayout10LayoutArray,
}
impl risc0_zkp::layout::Component for W2sLayout {
    fn ty_name(&self) -> &'static str {
        "W2sLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("a", self.a)?;
        v.visit_component("b", self.b)?;
        Ok(())
    }
}
pub struct Concatenate_7__5_Layout {
    pub _super: &'static NondetRegLayout12LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_7__5_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_7__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_20__2_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_20__2_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_20__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Concatenate_18__6_Layout {
    pub _super: &'static NondetRegLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for Concatenate_18__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Concatenate_18__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Sha256load_0_Layout {
    pub _super: &'static Sha256load_0__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_0___0: &'static _ArgumentsSha256load_0___0Layout,
    pub _2: &'static Sha256load_0___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_0_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_0___0",
            self._arguments_sha256load_0___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_1___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_1___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_1___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_1___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_1___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_1___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_1___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_1___0Arm1Layout {
    pub _super: &'static Sha256load_1___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_1___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_1___0Layout {
    pub arm0: &'static Sha256load_1___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_1___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_1___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_1_Layout {
    pub _super: &'static Sha256load_1__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_1___0: &'static _ArgumentsSha256load_1___0Layout,
    pub _2: &'static Sha256load_1___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_1___0",
            self._arguments_sha256load_1___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_2___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_2___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_2___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_2___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_2___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_2___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_2___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_2___0Arm1Layout {
    pub _super: &'static Sha256load_2___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_2___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_2___0Layout {
    pub arm0: &'static Sha256load_2___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_2___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_2___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_2_Layout {
    pub _super: &'static Sha256load_2__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_2___0: &'static _ArgumentsSha256load_2___0Layout,
    pub _2: &'static Sha256load_2___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_2___0",
            self._arguments_sha256load_2___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_3___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_3___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_3___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_3___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_3___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_3___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_3___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_3___0Arm1Layout {
    pub _super: &'static Sha256load_3___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_3___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_3___0Layout {
    pub arm0: &'static Sha256load_3___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_3___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_3___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_3_Layout {
    pub _super: &'static Sha256load_3__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_3___0: &'static _ArgumentsSha256load_3___0Layout,
    pub _2: &'static Sha256load_3___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_3_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_3___0",
            self._arguments_sha256load_3___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_4___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_4___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_4___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_4___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_4___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_4___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_4___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_4___0Arm1Layout {
    pub _super: &'static Sha256load_4___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_4___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_4___0Layout {
    pub arm0: &'static Sha256load_4___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_4___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_4___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_4_Layout {
    pub _super: &'static Sha256load_4__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_4___0: &'static _ArgumentsSha256load_4___0Layout,
    pub _2: &'static Sha256load_4___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_4_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_4___0",
            self._arguments_sha256load_4___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_5__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_5__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_5___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_5___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_5___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_5___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_5___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_5___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_5___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_5___0Arm1Layout {
    pub _super: &'static Sha256load_5___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_5___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_5___0Layout {
    pub arm0: &'static Sha256load_5___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_5___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_5___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_5_Layout {
    pub _super: &'static Sha256load_5__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_5___0: &'static _ArgumentsSha256load_5___0Layout,
    pub _2: &'static Sha256load_5___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_5_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_5___0",
            self._arguments_sha256load_5___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_6__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_6__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_6___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_6___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_6___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_6___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_6___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_6___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_6___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_6___0Arm1Layout {
    pub _super: &'static Sha256load_6___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_6___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_6___0Layout {
    pub arm0: &'static Sha256load_6___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_6___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_6___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_6_Layout {
    pub _super: &'static Sha256load_6__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_6___0: &'static _ArgumentsSha256load_6___0Layout,
    pub _2: &'static Sha256load_6___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_6_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_6___0",
            self._arguments_sha256load_6___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_7__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_7__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_7___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_7___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_7___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_7___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_7___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_7___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_7___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_7___0Arm1Layout {
    pub _super: &'static Sha256load_7___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_7___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_7___0Layout {
    pub arm0: &'static Sha256load_7___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_7___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_7___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_7_Layout {
    pub _super: &'static Sha256load_7__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_7___0: &'static _ArgumentsSha256load_7___0Layout,
    pub _2: &'static Sha256load_7___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_7_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_7___0",
            self._arguments_sha256load_7___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_8__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_8__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_8___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_8___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_8___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_8___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_8___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_8___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_8___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_8___0Arm1Layout {
    pub _super: &'static Sha256load_8___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_8___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_8___0Layout {
    pub arm0: &'static Sha256load_8___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_8___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_8___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_8_Layout {
    pub _super: &'static Sha256load_8__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_8___0: &'static _ArgumentsSha256load_8___0Layout,
    pub _2: &'static Sha256load_8___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_8_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_8___0",
            self._arguments_sha256load_8___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_9__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_9__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_9___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_9___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_9___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_9___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_9___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_9___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_9___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_9___0Arm1Layout {
    pub _super: &'static Sha256load_9___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_9___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_9___0Layout {
    pub arm0: &'static Sha256load_9___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_9___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_9___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_9_Layout {
    pub _super: &'static Sha256load_9__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_9___0: &'static _ArgumentsSha256load_9___0Layout,
    pub _2: &'static Sha256load_9___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_9_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_9___0",
            self._arguments_sha256load_9___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_10__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_10__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_10___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_10___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_10___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_10___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_10___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_10___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_10___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_10___0Arm1Layout {
    pub _super: &'static Sha256load_10___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_10___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_10___0Layout {
    pub arm0: &'static Sha256load_10___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_10___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_10___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_10_Layout {
    pub _super: &'static Sha256load_10__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_10___0: &'static _ArgumentsSha256load_10___0Layout,
    pub _2: &'static Sha256load_10___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_10_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_10_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_10___0",
            self._arguments_sha256load_10___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_11__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_11__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_11___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_11___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_11___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_11___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_11___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_11___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_11___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_11___0Arm1Layout {
    pub _super: &'static Sha256load_11___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_11___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_11___0Layout {
    pub arm0: &'static Sha256load_11___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_11___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_11___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_11_Layout {
    pub _super: &'static Sha256load_11__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_11___0: &'static _ArgumentsSha256load_11___0Layout,
    pub _2: &'static Sha256load_11___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_11_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_11___0",
            self._arguments_sha256load_11___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_12__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_12__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_12___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_12___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_12___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_12___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_12___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_12___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_12___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_12___0Arm1Layout {
    pub _super: &'static Sha256load_12___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_12___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_12___0Layout {
    pub arm0: &'static Sha256load_12___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_12___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_12___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_12_Layout {
    pub _super: &'static Sha256load_12__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_12___0: &'static _ArgumentsSha256load_12___0Layout,
    pub _2: &'static Sha256load_12___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_12_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_12___0",
            self._arguments_sha256load_12___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_13__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_13__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_13___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_13___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_13___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_13___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_13___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_13___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_13___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_13___0Arm1Layout {
    pub _super: &'static Sha256load_13___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_13___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_13___0Layout {
    pub arm0: &'static Sha256load_13___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_13___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_13___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_13_Layout {
    pub _super: &'static Sha256load_13__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_13___0: &'static _ArgumentsSha256load_13___0Layout,
    pub _2: &'static Sha256load_13___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_13_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_13_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_13___0",
            self._arguments_sha256load_13___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_14__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_14__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_14___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_14___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_14___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_14___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_14___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_14___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_14___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_14___0Arm1Layout {
    pub _super: &'static Sha256load_14___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_14___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_14___0Layout {
    pub arm0: &'static Sha256load_14___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_14___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_14___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_14_Layout {
    pub _super: &'static Sha256load_14__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_14___0: &'static _ArgumentsSha256load_14___0Layout,
    pub _2: &'static Sha256load_14___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_14_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_14_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_14___0",
            self._arguments_sha256load_14___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256load_15__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256load_15__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256load_15___0Layout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256load_15___0Layout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256load_15___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256load_15___0Arm0_SuperLayout {
    pub _0: &'static ReadMemoryLayout,
}
impl risc0_zkp::layout::Component for Sha256load_15___0Arm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15___0Arm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Sha256load_15___0Arm1_SuperLayout {
    pub _0: &'static NondetRegLayout,
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Sha256load_15___0Arm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15___0Arm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        Ok(())
    }
}
pub struct Sha256load_15___0Arm1Layout {
    pub _super: &'static Sha256load_15___0Arm1_SuperLayout,
    pub _extra0: &'static UseOnceMemoryElementLayout,
}
impl risc0_zkp::layout::Component for Sha256load_15___0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15___0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Sha256load_15___0Layout {
    pub arm0: &'static Sha256load_15___0Arm0_SuperLayout,
    pub arm1: &'static Sha256load_15___0Arm1Layout,
}
impl risc0_zkp::layout::Component for Sha256load_15___0Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15___0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256load_15_Layout {
    pub _super: &'static Sha256load_15__SuperLayout,
    pub inv_within: &'static NondetRegLayout,
    pub rs: &'static NondetRegLayout5LayoutArray,
    pub _arguments_sha256load_15___0: &'static _ArgumentsSha256load_15___0Layout,
    pub _2: &'static Sha256load_15___0Layout,
    pub high_w: &'static ToBitsU_16_Layout,
    pub low_w: &'static ToBitsU_16_Layout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub _6: &'static NondetRegLayout18LayoutArray,
    pub d: &'static Concatenate_18__6_Layout,
}
impl risc0_zkp::layout::Component for Sha256load_15_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256load_15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv_within", self.inv_within)?;
        v.visit_component("rs", self.rs)?;
        v.visit_component(
            "_arguments_sha256load_15___0",
            self._arguments_sha256load_15___0,
        )?;
        v.visit_component("_2", self._2)?;
        v.visit_component("high_w", self.high_w)?;
        v.visit_component("low_w", self.low_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("_6", self._6)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256loadAll_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256load_0_Layout,
    pub arm1: &'static Sha256load_1_Layout,
    pub arm2: &'static Sha256load_2_Layout,
    pub arm3: &'static Sha256load_3_Layout,
    pub arm4: &'static Sha256load_4_Layout,
    pub arm5: &'static Sha256load_5_Layout,
    pub arm6: &'static Sha256load_6_Layout,
    pub arm7: &'static Sha256load_7_Layout,
    pub arm8: &'static Sha256load_8_Layout,
    pub arm9: &'static Sha256load_9_Layout,
    pub arm10: &'static Sha256load_10_Layout,
    pub arm11: &'static Sha256load_11_Layout,
    pub arm12: &'static Sha256load_12_Layout,
    pub arm13: &'static Sha256load_13_Layout,
    pub arm14: &'static Sha256load_14_Layout,
    pub arm15: &'static Sha256load_15_Layout,
}
impl risc0_zkp::layout::Component for Sha256loadAll_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256loadAll_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        v.visit_component("arm10", self.arm10)?;
        v.visit_component("arm11", self.arm11)?;
        v.visit_component("arm12", self.arm12)?;
        v.visit_component("arm13", self.arm13)?;
        v.visit_component("arm14", self.arm14)?;
        v.visit_component("arm15", self.arm15)?;
        Ok(())
    }
}
pub struct OneHotU_16_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotU_16_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHotU_16_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct _ArgumentsSha256loadAll_SuperLayout {
    pub use_once_memory_element: &'static UseOnceMemoryElementLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _ArgumentsSha256loadAll_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_ArgumentsSha256loadAll_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        Ok(())
    }
}
pub struct Sha256loadAllLayout {
    pub _super: &'static Sha256loadAll_SuperLayout,
    pub m: &'static OneHotU_16_Layout,
    pub _arguments_sha256load_all__super: &'static _ArgumentsSha256loadAll_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256loadAllLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256loadAllLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("m", self.m)?;
        v.visit_component(
            "_arguments_sha256load_all__super",
            self._arguments_sha256load_all__super,
        )?;
        Ok(())
    }
}
pub struct Sha256mix_0__0__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__0__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__0__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__0__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct ConcatZeros_7__15_Layout {
    pub _super: &'static NondetRegLayout22LayoutArray,
}
impl risc0_zkp::layout::Component for ConcatZeros_7__15_Layout {
    fn ty_name(&self) -> &'static str {
        "ConcatZeros_7__15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct Sha256mix_0__0_Layout {
    pub _super: &'static Sha256mix_0__0__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__0_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__1__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__1__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__1__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__1__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__1_Layout {
    pub _super: &'static Sha256mix_0__1__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__2__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__2__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__2__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__2__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__2_Layout {
    pub _super: &'static Sha256mix_0__2__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__3__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__3__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__3__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__3__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__3_Layout {
    pub _super: &'static Sha256mix_0__3__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__4__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__4__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__4__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__4__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__4_Layout {
    pub _super: &'static Sha256mix_0__4__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__5__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__5__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__5__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__5__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__5__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__5__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__5__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__5_Layout {
    pub _super: &'static Sha256mix_0__5__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__5_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__6__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__6__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__6__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__6__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__6__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__6__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__6__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__6_Layout {
    pub _super: &'static Sha256mix_0__6__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__7__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__7__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__7__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__7__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__7__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__7__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__7__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__7_Layout {
    pub _super: &'static Sha256mix_0__7__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__7_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__8__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__8__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__8__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__8__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__8__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__8__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__8__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__8_Layout {
    pub _super: &'static Sha256mix_0__8__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__8_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__9__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__9__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__9__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__9__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__9__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__9__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__9__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__9_Layout {
    pub _super: &'static Sha256mix_0__9__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__9_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__10__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__10__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__10__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__10__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__10__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__10__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__10__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__10_Layout {
    pub _super: &'static Sha256mix_0__10__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__10_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__10_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__11__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__11__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__11__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__11__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__11__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__11__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__11__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__11_Layout {
    pub _super: &'static Sha256mix_0__11__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__11_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__12__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__12__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__12__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__12__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__12__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__12__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__12__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__12_Layout {
    pub _super: &'static Sha256mix_0__12__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__12_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__13__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__13__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__13__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__13__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__13__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__13__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__13__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__13_Layout {
    pub _super: &'static Sha256mix_0__13__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__13_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__13_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__14__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__14__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__14__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__14__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__14__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__14__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__14__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__14_Layout {
    pub _super: &'static Sha256mix_0__14__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__14_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__14_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_0__15__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__15__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__15__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__15__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_0__15__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__15__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__15__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_0__15_Layout {
    pub _super: &'static Sha256mix_0__15__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_0__15_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_0__15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mixAll_0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256mix_0__0_Layout,
    pub arm1: &'static Sha256mix_0__1_Layout,
    pub arm2: &'static Sha256mix_0__2_Layout,
    pub arm3: &'static Sha256mix_0__3_Layout,
    pub arm4: &'static Sha256mix_0__4_Layout,
    pub arm5: &'static Sha256mix_0__5_Layout,
    pub arm6: &'static Sha256mix_0__6_Layout,
    pub arm7: &'static Sha256mix_0__7_Layout,
    pub arm8: &'static Sha256mix_0__8_Layout,
    pub arm9: &'static Sha256mix_0__9_Layout,
    pub arm10: &'static Sha256mix_0__10_Layout,
    pub arm11: &'static Sha256mix_0__11_Layout,
    pub arm12: &'static Sha256mix_0__12_Layout,
    pub arm13: &'static Sha256mix_0__13_Layout,
    pub arm14: &'static Sha256mix_0__14_Layout,
    pub arm15: &'static Sha256mix_0__15_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        v.visit_component("arm10", self.arm10)?;
        v.visit_component("arm11", self.arm11)?;
        v.visit_component("arm12", self.arm12)?;
        v.visit_component("arm13", self.arm13)?;
        v.visit_component("arm14", self.arm14)?;
        v.visit_component("arm15", self.arm15)?;
        Ok(())
    }
}
pub struct Sha256mixAll_0_Layout {
    pub _super: &'static Sha256mixAll_0__SuperLayout,
    pub m: &'static OneHotU_16_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_0_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("m", self.m)?;
        Ok(())
    }
}
pub struct Sha256mix_1__0__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__0__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__0__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__0__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__0_Layout {
    pub _super: &'static Sha256mix_1__0__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__0_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__1__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__1__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__1__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__1__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__1_Layout {
    pub _super: &'static Sha256mix_1__1__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__2__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__2__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__2__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__2__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__2_Layout {
    pub _super: &'static Sha256mix_1__2__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__3__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__3__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__3__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__3__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__3_Layout {
    pub _super: &'static Sha256mix_1__3__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__4__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__4__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__4__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__4__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__4_Layout {
    pub _super: &'static Sha256mix_1__4__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__5__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__5__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__5__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__5__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__5__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__5__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__5__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__5_Layout {
    pub _super: &'static Sha256mix_1__5__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__5_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__6__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__6__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__6__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__6__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__6__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__6__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__6__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__6_Layout {
    pub _super: &'static Sha256mix_1__6__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__7__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__7__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__7__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__7__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__7__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__7__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__7__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__7_Layout {
    pub _super: &'static Sha256mix_1__7__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__7_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__8__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__8__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__8__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__8__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__8__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__8__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__8__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__8_Layout {
    pub _super: &'static Sha256mix_1__8__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__8_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__9__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__9__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__9__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__9__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__9__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__9__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__9__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__9_Layout {
    pub _super: &'static Sha256mix_1__9__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__9_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__10__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__10__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__10__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__10__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__10__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__10__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__10__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__10_Layout {
    pub _super: &'static Sha256mix_1__10__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__10_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__10_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__11__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__11__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__11__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__11__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__11__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__11__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__11__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__11_Layout {
    pub _super: &'static Sha256mix_1__11__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__11_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__12__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__12__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__12__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__12__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__12__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__12__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__12__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__12_Layout {
    pub _super: &'static Sha256mix_1__12__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__12_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__13__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__13__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__13__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__13__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__13__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__13__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__13__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__13_Layout {
    pub _super: &'static Sha256mix_1__13__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__13_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__13_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__14__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__14__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__14__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__14__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__14__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__14__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__14__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__14_Layout {
    pub _super: &'static Sha256mix_1__14__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__14_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__14_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_1__15__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__15__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__15__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__15__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_1__15__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__15__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__15__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_1__15_Layout {
    pub _super: &'static Sha256mix_1__15__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_1__15_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_1__15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mixAll_1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256mix_1__0_Layout,
    pub arm1: &'static Sha256mix_1__1_Layout,
    pub arm2: &'static Sha256mix_1__2_Layout,
    pub arm3: &'static Sha256mix_1__3_Layout,
    pub arm4: &'static Sha256mix_1__4_Layout,
    pub arm5: &'static Sha256mix_1__5_Layout,
    pub arm6: &'static Sha256mix_1__6_Layout,
    pub arm7: &'static Sha256mix_1__7_Layout,
    pub arm8: &'static Sha256mix_1__8_Layout,
    pub arm9: &'static Sha256mix_1__9_Layout,
    pub arm10: &'static Sha256mix_1__10_Layout,
    pub arm11: &'static Sha256mix_1__11_Layout,
    pub arm12: &'static Sha256mix_1__12_Layout,
    pub arm13: &'static Sha256mix_1__13_Layout,
    pub arm14: &'static Sha256mix_1__14_Layout,
    pub arm15: &'static Sha256mix_1__15_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        v.visit_component("arm10", self.arm10)?;
        v.visit_component("arm11", self.arm11)?;
        v.visit_component("arm12", self.arm12)?;
        v.visit_component("arm13", self.arm13)?;
        v.visit_component("arm14", self.arm14)?;
        v.visit_component("arm15", self.arm15)?;
        Ok(())
    }
}
pub struct Sha256mixAll_1_Layout {
    pub _super: &'static Sha256mixAll_1__SuperLayout,
    pub m: &'static OneHotU_16_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("m", self.m)?;
        Ok(())
    }
}
pub struct Sha256mix_2__0__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__0__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__0__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__0__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__0__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__0__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__0__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__0_Layout {
    pub _super: &'static Sha256mix_2__0__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__0_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__0_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__1__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__1__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__1__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__1__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__1__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__1__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__1__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__1_Layout {
    pub _super: &'static Sha256mix_2__1__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__1_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__1_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__2__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__2__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__2__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__2__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__2_Layout {
    pub _super: &'static Sha256mix_2__2__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__3__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__3__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__3__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__3__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__3__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__3__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__3__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__3_Layout {
    pub _super: &'static Sha256mix_2__3__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__3_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__4__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__4__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__4__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__4__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__4__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__4__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__4__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__4_Layout {
    pub _super: &'static Sha256mix_2__4__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__4_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__5__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__5__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__5__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__5__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__5__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__5__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__5__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__5_Layout {
    pub _super: &'static Sha256mix_2__5__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__5_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__6__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__6__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__6__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__6__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__6__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__6__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__6__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__6_Layout {
    pub _super: &'static Sha256mix_2__6__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__6_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__7__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__7__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__7__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__7__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__7__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__7__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__7__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__7_Layout {
    pub _super: &'static Sha256mix_2__7__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__7_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__7_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__8__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__8__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__8__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__8__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__8__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__8__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__8__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__8_Layout {
    pub _super: &'static Sha256mix_2__8__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__8_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__9__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__9__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__9__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__9__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__9__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__9__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__9__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__9_Layout {
    pub _super: &'static Sha256mix_2__9__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__9_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__9_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__10__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__10__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__10__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__10__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__10__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__10__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__10__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__10_Layout {
    pub _super: &'static Sha256mix_2__10__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__10_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__10_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__11__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__11__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__11__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__11__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__11__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__11__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__11__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__11_Layout {
    pub _super: &'static Sha256mix_2__11__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__11_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__12__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__12__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__12__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__12__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__12__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__12__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__12__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__12_Layout {
    pub _super: &'static Sha256mix_2__12__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__12_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__12_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__13__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__13__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__13__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__13__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__13__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__13__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__13__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__13_Layout {
    pub _super: &'static Sha256mix_2__13__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__13_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__13_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__14__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__14__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__14__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__14__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__14__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__14__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__14__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__14_Layout {
    pub _super: &'static Sha256mix_2__14__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__14_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__14_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mix_2__15__SuperArm1_Super_SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static RetTupleLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__15__SuperArm1_Super_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__15__SuperArm1_Super_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__15__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static RetTupleLayout,
    pub arm1: &'static Sha256mix_2__15__SuperArm1_Super_SuperLayout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__15__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__15__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct Sha256mix_2__15_Layout {
    pub _super: &'static Sha256mix_2__15__SuperLayout,
    pub new_w: &'static UnpackWithCarryLayout,
    pub new_a: &'static UnpackWithCarryLayout,
    pub new_e: &'static UnpackWithCarryLayout,
    pub stae: &'static Ae2sLayout,
    pub stw: &'static W2sLayout,
    pub tmpb: &'static Concatenate_7__5_Layout,
    pub padb: &'static Concatenate_10__12_Layout,
    pub padc: &'static Concatenate_20__2_Layout,
    pub tmpd: &'static ConcatZeros_7__15_Layout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static NondetRegLayout,
    pub _4: &'static NondetRegLayout2LayoutArray,
    pub d: &'static Concatenate_22__2_Layout,
}
impl risc0_zkp::layout::Component for Sha256mix_2__15_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mix_2__15_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("new_w", self.new_w)?;
        v.visit_component("new_a", self.new_a)?;
        v.visit_component("new_e", self.new_e)?;
        v.visit_component("stae", self.stae)?;
        v.visit_component("stw", self.stw)?;
        v.visit_component("tmpb", self.tmpb)?;
        v.visit_component("padb", self.padb)?;
        v.visit_component("padc", self.padc)?;
        v.visit_component("tmpd", self.tmpd)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("_4", self._4)?;
        v.visit_component("d", self.d)?;
        Ok(())
    }
}
pub struct Sha256mixAll_2__SuperLayout {
    pub _super: &'static RetTupleLayout,
    pub arm0: &'static Sha256mix_2__0_Layout,
    pub arm1: &'static Sha256mix_2__1_Layout,
    pub arm2: &'static Sha256mix_2__2_Layout,
    pub arm3: &'static Sha256mix_2__3_Layout,
    pub arm4: &'static Sha256mix_2__4_Layout,
    pub arm5: &'static Sha256mix_2__5_Layout,
    pub arm6: &'static Sha256mix_2__6_Layout,
    pub arm7: &'static Sha256mix_2__7_Layout,
    pub arm8: &'static Sha256mix_2__8_Layout,
    pub arm9: &'static Sha256mix_2__9_Layout,
    pub arm10: &'static Sha256mix_2__10_Layout,
    pub arm11: &'static Sha256mix_2__11_Layout,
    pub arm12: &'static Sha256mix_2__12_Layout,
    pub arm13: &'static Sha256mix_2__13_Layout,
    pub arm14: &'static Sha256mix_2__14_Layout,
    pub arm15: &'static Sha256mix_2__15_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_2__SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_2__SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        v.visit_component("arm10", self.arm10)?;
        v.visit_component("arm11", self.arm11)?;
        v.visit_component("arm12", self.arm12)?;
        v.visit_component("arm13", self.arm13)?;
        v.visit_component("arm14", self.arm14)?;
        v.visit_component("arm15", self.arm15)?;
        Ok(())
    }
}
pub struct Sha256mixAll_2_Layout {
    pub _super: &'static Sha256mixAll_2__SuperLayout,
    pub m: &'static OneHotU_16_Layout,
}
impl risc0_zkp::layout::Component for Sha256mixAll_2_Layout {
    fn ty_name(&self) -> &'static str {
        "Sha256mixAll_2_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("m", self.m)?;
        Ok(())
    }
}
pub struct TopRetvalsLayout {
    pub _super: &'static RetTupleLayout,
    pub _selector: &'static NondetRegLayout22LayoutArray,
    pub arm0: &'static Xor5wordsMinor0Layout,
    pub arm1: &'static Xor5wordsMinor1Layout,
    pub arm2: &'static Xor5wordsMinor2Layout,
    pub arm3: &'static Xor5wordsMinor3Layout,
    pub arm4: &'static RhoMajor_0_Layout,
    pub arm5: &'static RhoMajor_1_Layout,
    pub arm6: &'static RhoMajor_2_Layout,
    pub arm7: &'static RhoMajor_3_Layout,
    pub arm8: &'static RhoMajor_4_Layout,
    pub arm9: &'static ChiMajor_0_Layout,
    pub arm10: &'static ChiMajor_1_Layout,
    pub arm11: &'static ChiMajor_2_Layout,
    pub arm12: &'static ChiMajor_3_Layout,
    pub arm13: &'static ChiMajor_4_Layout,
    pub arm14: &'static AbsorbMajor1Layout,
    pub arm15: &'static AbsorbMajor2Layout,
    pub arm16: &'static Sha256setupMajorLayout,
    pub arm17: &'static ExtraMajorLayout,
    pub arm18: &'static Sha256loadAllLayout,
    pub arm19: &'static Sha256mixAll_0_Layout,
    pub arm20: &'static Sha256mixAll_1_Layout,
    pub arm21: &'static Sha256mixAll_2_Layout,
}
impl risc0_zkp::layout::Component for TopRetvalsLayout {
    fn ty_name(&self) -> &'static str {
        "TopRetvalsLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_selector", self._selector)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        v.visit_component("arm8", self.arm8)?;
        v.visit_component("arm9", self.arm9)?;
        v.visit_component("arm10", self.arm10)?;
        v.visit_component("arm11", self.arm11)?;
        v.visit_component("arm12", self.arm12)?;
        v.visit_component("arm13", self.arm13)?;
        v.visit_component("arm14", self.arm14)?;
        v.visit_component("arm15", self.arm15)?;
        v.visit_component("arm16", self.arm16)?;
        v.visit_component("arm17", self.arm17)?;
        v.visit_component("arm18", self.arm18)?;
        v.visit_component("arm19", self.arm19)?;
        v.visit_component("arm20", self.arm20)?;
        v.visit_component("arm21", self.arm21)?;
        Ok(())
    }
}
pub struct ShaOutputLayout {
    pub vals: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ShaOutputLayout {
    fn ty_name(&self) -> &'static str {
        "ShaOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("vals", self.vals)?;
        Ok(())
    }
}
pub type Reg1LayoutArray = [&'static Reg; 1];
pub struct LayoutAccumLayout {
    pub columns: &'static Reg1LayoutArray,
}
impl risc0_zkp::layout::Component for LayoutAccumLayout {
    fn ty_name(&self) -> &'static str {
        "LayoutAccumLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("columns", self.columns)?;
        Ok(())
    }
}
pub struct TopLayout {
    pub retvals: &'static RetTupleLayout,
    pub pc: &'static NondetRegLayout,
    pub inv_pc: &'static NondetRegLayout,
    pub major_onehot: &'static OneHot_22_Layout,
    pub retvals_redef: &'static TopRetvalsLayout,
}
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("retvals", self.retvals)?;
        v.visit_component("pc", self.pc)?;
        v.visit_component("inv_pc", self.inv_pc)?;
        v.visit_component("major_onehot", self.major_onehot)?;
        v.visit_component("retvals_redef", self.retvals_redef)?;
        Ok(())
    }
}
pub struct _globalLayout {
    pub output: &'static ShaOutputLayout,
}
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("output", self.output)?;
        Ok(())
    }
}
pub struct Arg_UseOnceMemoryElementLayout {
    pub index: &'static Reg,
    pub digest0: &'static Reg,
    pub digest1: &'static Reg,
    pub digest2: &'static Reg,
    pub digest3: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_UseOnceMemoryElementLayout {
    fn ty_name(&self) -> &'static str {
        "Arg_UseOnceMemoryElementLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("index", self.index)?;
        v.visit_component("digest0", self.digest0)?;
        v.visit_component("digest1", self.digest1)?;
        v.visit_component("digest2", self.digest2)?;
        v.visit_component("digest3", self.digest3)?;
        Ok(())
    }
}
pub struct _accumLayout {
    pub use_once_memory_element: &'static Arg_UseOnceMemoryElementLayout,
    pub _offset: &'static Reg,
}
impl risc0_zkp::layout::Component for _accumLayout {
    fn ty_name(&self) -> &'static str {
        "_accumLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("use_once_memory_element", self.use_once_memory_element)?;
        v.visit_component("_offset", self._offset)?;
        Ok(())
    }
}
pub struct _mixLayout {
    pub randomness: &'static _accumLayout,
}
impl risc0_zkp::layout::Component for _mixLayout {
    fn ty_name(&self) -> &'static str {
        "_mixLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("randomness", self.randomness)?;
        Ok(())
    }
}
#[derive(Copy, Clone, Debug)]
pub struct NondetRegStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct RegStruct {
    pub _super: NondetRegStruct,
}
pub type Val22Array = [Val; 22];
pub type NondetRegStruct22Array = [NondetRegStruct; 22];
pub type Val24Array = [Val; 24];
pub type NondetRegStruct24Array = [NondetRegStruct; 24];
pub type Val16Array = [Val; 16];
pub type NondetRegStruct16Array = [NondetRegStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct RetTupleStruct<'a> {
    pub arr_a: NondetRegStruct22Array,
    pub arr_b: NondetRegStruct22Array,
    pub arr_c: NondetRegStruct22Array,
    pub arr_d: NondetRegStruct24Array,
    pub arr_e: NondetRegStruct22Array,
    pub arr_f: NondetRegStruct22Array,
    pub minor: RegStruct,
    pub major: RegStruct,
    pub rnd: RegStruct,
    pub blk: RegStruct,
    pub midx: RegStruct,
    pub auxr: NondetRegStruct16Array,
    pub _layout: BoundLayout<'a, RetTupleLayout, Val>,
}
pub type NondetRegStruct32Array = [NondetRegStruct; 32];
#[derive(Copy, Clone, Debug)]
pub struct W2sAStruct {
    pub _super: NondetRegStruct,
}
pub type W2sAStruct22Array = [W2sAStruct; 22];
pub type Val10Array = [Val; 10];
#[derive(Copy, Clone, Debug)]
pub struct W2sBStruct {
    pub _super: NondetRegStruct,
}
pub type W2sBStruct10Array = [W2sBStruct; 10];
#[derive(Copy, Clone, Debug)]
pub struct W2sStruct {
    pub a: W2sAStruct22Array,
    pub b: W2sBStruct10Array,
}
pub type NondetRegStruct10Array = [NondetRegStruct; 10];
pub type NondetRegStruct12Array = [NondetRegStruct; 12];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_10__12__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_10__12__SuperStruct22Array = [Concatenate_10__12__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_10__12_Struct {
    pub _super: Concatenate_10__12__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Ae2sEStruct {
    pub _super: NondetRegStruct,
}
pub type Ae2sEStruct22Array = [Ae2sEStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Ae2sF1Struct {
    pub _super: NondetRegStruct,
}
pub type Ae2sF1Struct10Array = [Ae2sF1Struct; 10];
pub type Val12Array = [Val; 12];
#[derive(Copy, Clone, Debug)]
pub struct Ae2sF2Struct {
    pub _super: NondetRegStruct,
}
pub type Ae2sF2Struct12Array = [Ae2sF2Struct; 12];
pub type Val20Array = [Val; 20];
#[derive(Copy, Clone, Debug)]
pub struct Ae2sCStruct {
    pub _super: NondetRegStruct,
}
pub type Ae2sCStruct20Array = [Ae2sCStruct; 20];
#[derive(Copy, Clone, Debug)]
pub struct Ae2sStruct {
    pub e: Ae2sEStruct22Array,
    pub f: Concatenate_10__12_Struct,
    pub c: Ae2sCStruct20Array,
}
pub type Val19Array = [Val; 19];
pub type Val31Array = [Val; 31];
pub type NondetRegStruct19Array = [NondetRegStruct; 19];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_19_Struct {
    pub _super: NondetRegStruct19Array,
}
pub type NondetRegStruct20Array = [NondetRegStruct; 20];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_20_Struct {
    pub _super: NondetRegStruct20Array,
}
pub type Val7Array = [Val; 7];
pub type NondetRegStruct3Array = [NondetRegStruct; 3];
pub type NondetRegStruct4Array = [NondetRegStruct; 4];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__4__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_3__4__SuperStruct7Array = [Concatenate_3__4__SuperStruct; 7];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__4_Struct {
    pub _super: Concatenate_3__4__SuperStruct7Array,
}
pub type Val32Array = [Val; 32];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_16__16__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_16__16__SuperStruct32Array = [Concatenate_16__16__SuperStruct; 32];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_16__16_Struct {
    pub _super: Concatenate_16__16__SuperStruct32Array,
}
#[derive(Copy, Clone, Debug)]
pub struct UnpackWithCarryStruct {
    pub _super: Concatenate_16__16_Struct,
    pub carrybits: Concatenate_3__4_Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_16_Struct {
    pub _super: NondetRegStruct16Array,
}
pub type Val4Array = [Val; 4];
#[derive(Copy, Clone, Debug)]
pub struct UseOnceMemoryElementStruct {
    pub count: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct ReadMemoryStruct {}
pub type NondetRegStruct5Array = [NondetRegStruct; 5];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_16__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ToBitsU_16__SuperStruct16Array = [ToBitsU_16__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_16_Struct {
    pub _super: ToBitsU_16__SuperStruct16Array,
}
pub type NondetRegStruct7Array = [NondetRegStruct; 7];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_7__5__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_7__5__SuperStruct12Array = [Concatenate_7__5__SuperStruct; 12];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_7__5_Struct {
    pub _super: Concatenate_7__5__SuperStruct12Array,
}
pub type NondetRegStruct2Array = [NondetRegStruct; 2];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_20__2__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_20__2__SuperStruct22Array = [Concatenate_20__2__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_20__2_Struct {
    pub _super: Concatenate_20__2__SuperStruct22Array,
}
pub type NondetRegStruct18Array = [NondetRegStruct; 18];
pub type NondetRegStruct6Array = [NondetRegStruct; 6];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_18__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_18__6__SuperStruct24Array = [Concatenate_18__6__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_18__6_Struct {
    pub _super: Concatenate_18__6__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}
pub type ComponentStruct5Array = [ComponentStruct; 5];
pub type Val2Array = [Val; 2];
pub type Val18Array = [Val; 18];
#[derive(Copy, Clone, Debug)]
pub struct ShaOutputValsStruct {}
pub type ShaOutputValsStruct16Array = [ShaOutputValsStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ShaOutputStruct {}
#[derive(Copy, Clone, Debug)]
pub struct WriteMemoryStruct {}
#[derive(Copy, Clone, Debug)]
pub struct DummyBStruct {
    pub _super: NondetRegStruct22Array,
}
pub type Val21Array = [Val; 21];
pub type NondetRegStruct21Array = [NondetRegStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct DummySStruct {
    pub _super: NondetRegStruct21Array,
}
#[derive(Copy, Clone, Debug)]
pub struct DummyEStruct {
    pub _super: NondetRegStruct24Array,
}
pub type Val5Array = [Val; 5];
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_5_Struct {
    pub _super: NondetRegStruct5Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ToBits_22_Struct {
    pub _super: NondetRegStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_22_Struct {
    pub _super: NondetRegStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct FromBits_22_Struct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct Xor5wordsResult_0_Struct {
    pub _super: FromBits_22_Struct,
    pub minor_onehot: OneHotU_5_Struct,
    pub unpack_a: ToBits_22_Struct,
    pub unpack_b: ToBits_22_Struct,
    pub unpack_c: ToBits_22_Struct,
    pub unpack_d: ToBits_22_Struct,
    pub unpack_e: ToBitsU_22_Struct,
    pub resbits: NondetRegStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_22__2__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_22__2__SuperStruct24Array = [Concatenate_22__2__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_22__2_Struct {
    pub _super: Concatenate_22__2__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_5__11__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_5__11__SuperStruct16Array = [ConcatZeros_5__11__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_5__11_Struct {
    pub _super: ConcatZeros_5__11__SuperStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ToBits_21_Struct {
    pub _super: NondetRegStruct21Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_21_Struct {
    pub _super: NondetRegStruct21Array,
}
#[derive(Copy, Clone, Debug)]
pub struct FromBits_21_Struct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct Xor5wordsResult_1_Struct {
    pub _super: FromBits_21_Struct,
    pub minor_onehot: OneHotU_5_Struct,
    pub unpack_a: ToBits_21_Struct,
    pub unpack_b: ToBits_21_Struct,
    pub unpack_c: ToBits_21_Struct,
    pub unpack_d: ToBits_21_Struct,
    pub unpack_e: ToBitsU_21_Struct,
    pub resbits: NondetRegStruct21Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatElem_21__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatElem_21__SuperStruct22Array = [ConcatElem_21__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatElem_21_Struct {
    pub _super: ConcatElem_21__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_21__1__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_21__1__SuperStruct22Array = [ConcatZeros_21__1__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_21__1_Struct {
    pub _super: ConcatZeros_21__1__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_21__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_21__3__SuperStruct24Array = [ConcatZeros_21__3__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_21__3_Struct {
    pub _super: ConcatZeros_21__3__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Xor5wordsResult_2_Struct {
    pub _super: FromBits_21_Struct,
    pub minor_onehot: OneHotU_5_Struct,
    pub unpack_a: ToBits_21_Struct,
    pub unpack_b: ToBits_21_Struct,
    pub unpack_c: ToBits_21_Struct,
    pub unpack_d: ToBits_21_Struct,
    pub unpack_e: ToBitsU_21_Struct,
    pub resbits: NondetRegStruct21Array,
}
pub type RegStruct22Array = [RegStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct RGZ_22_Struct {
    pub _super: RegStruct22Array,
}
pub type NondetRegStruct9Array = [NondetRegStruct; 9];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_9__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_9__3__SuperStruct12Array = [Concatenate_9__3__SuperStruct; 12];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_9__3_Struct {
    pub _super: Concatenate_9__3__SuperStruct12Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_12__12__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_12__12__SuperStruct24Array = [ConcatZeros_12__12__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_12__12_Struct {
    pub _super: ConcatZeros_12__12__SuperStruct24Array,
}
pub type Val9Array = [Val; 9];
pub type RegStruct9Array = [RegStruct; 9];
pub type Val15Array = [Val; 15];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_12__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_12__3__SuperStruct15Array = [Concatenate_12__3__SuperStruct; 15];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_12__3_Struct {
    pub _super: Concatenate_12__3__SuperStruct15Array,
}
pub type NondetRegStruct15Array = [NondetRegStruct; 15];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_15__9__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_15__9__SuperStruct24Array = [ConcatZeros_15__9__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_15__9_Struct {
    pub _super: ConcatZeros_15__9__SuperStruct24Array,
}
pub type RegStruct12Array = [RegStruct; 12];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_15__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_15__3__SuperStruct18Array = [Concatenate_15__3__SuperStruct; 18];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_15__3_Struct {
    pub _super: Concatenate_15__3__SuperStruct18Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_18__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_18__6__SuperStruct24Array = [ConcatZeros_18__6__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_18__6_Struct {
    pub _super: ConcatZeros_18__6__SuperStruct24Array,
}
pub type RegStruct15Array = [RegStruct; 15];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_18__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_18__3__SuperStruct21Array = [Concatenate_18__3__SuperStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_18__3_Struct {
    pub _super: Concatenate_18__3__SuperStruct21Array,
}
pub type RegStruct18Array = [RegStruct; 18];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_21__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_21__3__SuperStruct24Array = [Concatenate_21__3__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_21__3_Struct {
    pub _super: Concatenate_21__3__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_24__0__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_24__0__SuperStruct24Array = [ConcatZeros_24__0__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_24__0_Struct {
    pub _super: ConcatZeros_24__0__SuperStruct24Array,
}
pub type RegStruct21Array = [RegStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__19__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_3__19__SuperStruct22Array = [Concatenate_3__19__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__19_Struct {
    pub _super: Concatenate_3__19__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__13__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_9__13__SuperStruct22Array = [ConcatZeros_9__13__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__13_Struct {
    pub _super: ConcatZeros_9__13__SuperStruct22Array,
}
pub type Val22Array3Array = [Val22Array; 3];
#[derive(Copy, Clone, Debug)]
pub struct RctableStruct {
    pub _super: Val22Array3Array,
}
pub type Val3Array = [Val; 3];
pub type RegStruct19Array = [RegStruct; 19];
pub type NondetRegStruct11Array = [NondetRegStruct; 11];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_11__5__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_11__5__SuperStruct16Array = [ConcatZeros_11__5__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_11__5_Struct {
    pub _super: ConcatZeros_11__5__SuperStruct16Array,
}
pub type RegStruct24Array = [RegStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct RGZ_24_Struct {
    pub _super: RegStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_10_Struct {
    pub _super: NondetRegStruct10Array,
}
pub type Val11Array = [Val; 11];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_10__1__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_10__1__SuperStruct11Array = [ConcatZeros_10__1__SuperStruct; 11];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_10__1_Struct {
    pub _super: ConcatZeros_10__1__SuperStruct11Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatElem_11__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatElem_11__SuperStruct12Array = [ConcatElem_11__SuperStruct; 12];
#[derive(Copy, Clone, Debug)]
pub struct ConcatElem_11_Struct {
    pub _super: ConcatElem_11__SuperStruct12Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_12__4__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_12__4__SuperStruct16Array = [Concatenate_12__4__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_12__4_Struct {
    pub _super: Concatenate_12__4__SuperStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__0_Arr12Struct22Array = [AbsorbWord_0__0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__0_Arr23Struct21Array = [AbsorbWord_0__0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__0_Arr34Struct21Array = [AbsorbWord_0__0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__0_Arr12Struct22Array = [AbsorbWord_1__0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__0_Arr23Struct21Array = [AbsorbWord_1__0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__0_Arr34Struct21Array = [AbsorbWord_1__0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__0_Arr12Struct22Array = [AbsorbWord_2__0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__0_Arr23Struct21Array = [AbsorbWord_2__0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__0_Arr34Struct21Array = [AbsorbWord_2__0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__0_Arr12Struct22Array = [AbsorbWord_3__0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__0_Arr23Struct21Array = [AbsorbWord_3__0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__0_Arr34Struct21Array = [AbsorbWord_3__0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__0_Arr12Struct22Array = [AbsorbWord_4__0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__0_Arr23Struct21Array = [AbsorbWord_4__0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__0_Arr34Struct21Array = [AbsorbWord_4__0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__1_Arr12Struct22Array = [AbsorbWord_0__1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__1_Arr23Struct21Array = [AbsorbWord_0__1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__1_Arr34Struct21Array = [AbsorbWord_0__1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__1_Arr12Struct22Array = [AbsorbWord_1__1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__1_Arr23Struct21Array = [AbsorbWord_1__1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__1_Arr34Struct21Array = [AbsorbWord_1__1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__1_Arr12Struct22Array = [AbsorbWord_2__1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__1_Arr23Struct21Array = [AbsorbWord_2__1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__1_Arr34Struct21Array = [AbsorbWord_2__1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__1_Arr12Struct22Array = [AbsorbWord_3__1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__1_Arr23Struct21Array = [AbsorbWord_3__1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__1_Arr34Struct21Array = [AbsorbWord_3__1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_9_Struct {
    pub _super: NondetRegStruct9Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__2__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_9__2__SuperStruct11Array = [ConcatZeros_9__2__SuperStruct; 11];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__2_Struct {
    pub _super: ConcatZeros_9__2__SuperStruct11Array,
}
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__1_Arr12Struct22Array = [AbsorbWord_4__1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__1_Arr23Struct21Array = [AbsorbWord_4__1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__1_Arr34Struct21Array = [AbsorbWord_4__1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__2_Arr12Struct22Array = [AbsorbWord_0__2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__2_Arr23Struct21Array = [AbsorbWord_0__2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__2_Arr34Struct21Array = [AbsorbWord_0__2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__2_Arr12Struct22Array = [AbsorbWord_1__2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__2_Arr23Struct21Array = [AbsorbWord_1__2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__2_Arr34Struct21Array = [AbsorbWord_1__2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__2_Arr12Struct22Array = [AbsorbWord_2__2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__2_Arr23Struct21Array = [AbsorbWord_2__2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_2__2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_2__2_Arr34Struct21Array = [AbsorbWord_2__2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__2_Arr12Struct22Array = [AbsorbWord_3__2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__2_Arr23Struct21Array = [AbsorbWord_3__2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_3__2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_3__2_Arr34Struct21Array = [AbsorbWord_3__2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__2_Arr12Struct22Array = [AbsorbWord_4__2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__2_Arr23Struct21Array = [AbsorbWord_4__2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_4__2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_4__2_Arr34Struct21Array = [AbsorbWord_4__2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__3_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__3_Arr12Struct22Array = [AbsorbWord_0__3_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__3_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__3_Arr23Struct21Array = [AbsorbWord_0__3_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_0__3_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_0__3_Arr34Struct21Array = [AbsorbWord_0__3_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__3_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__3_Arr12Struct22Array = [AbsorbWord_1__3_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__3_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__3_Arr23Struct21Array = [AbsorbWord_1__3_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct AbsorbWord_1__3_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type AbsorbWord_1__3_Arr34Struct21Array = [AbsorbWord_1__3_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_3_Struct {
    pub _super: NondetRegStruct3Array,
}
pub type Val14Array = [Val; 14];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_11__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_11__3__SuperStruct14Array = [Concatenate_11__3__SuperStruct; 14];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_11__3_Struct {
    pub _super: Concatenate_11__3__SuperStruct14Array,
}
pub type NondetRegStruct14Array = [NondetRegStruct; 14];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_14__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_14__6__SuperStruct20Array = [Concatenate_14__6__SuperStruct; 20];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_14__6_Struct {
    pub _super: Concatenate_14__6__SuperStruct20Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_20__4__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_20__4__SuperStruct24Array = [Concatenate_20__4__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_20__4_Struct {
    pub _super: Concatenate_20__4__SuperStruct24Array,
}
pub type Val30Array = [Val; 30];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_30__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ToBitsU_30__SuperStruct30Array = [ToBitsU_30__SuperStruct; 30];
#[derive(Copy, Clone, Debug)]
pub struct ToBitsU_30_Struct {
    pub _super: ToBitsU_30__SuperStruct30Array,
}
pub type Val13Array = [Val; 13];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_9__4__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_9__4__SuperStruct13Array = [Concatenate_9__4__SuperStruct; 13];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_9__4_Struct {
    pub _super: Concatenate_9__4__SuperStruct13Array,
}
pub type NondetRegStruct13Array = [NondetRegStruct; 13];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_13__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_13__6__SuperStruct19Array = [ConcatZeros_13__6__SuperStruct; 19];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_13__6_Struct {
    pub _super: ConcatZeros_13__6__SuperStruct19Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_19__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_19__3__SuperStruct22Array = [Concatenate_19__3__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_19__3_Struct {
    pub _super: Concatenate_19__3__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_3__21__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_3__21__SuperStruct24Array = [ConcatZeros_3__21__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_3__21_Struct {
    pub _super: ConcatZeros_3__21__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct MemPaddingSha256ATmpStruct {
    pub _super: NondetRegStruct,
}
pub type MemPaddingSha256ATmpStruct21Array = [MemPaddingSha256ATmpStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct MemPaddingSha256BTmpStruct {
    pub _super: NondetRegStruct,
}
pub type MemPaddingSha256BTmpStruct9Array = [MemPaddingSha256BTmpStruct; 9];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_10__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_10__6__SuperStruct16Array = [ConcatZeros_10__6__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_10__6_Struct {
    pub _super: ConcatZeros_10__6__SuperStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_20__2__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_20__2__SuperStruct22Array = [ConcatZeros_20__2__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_20__2_Struct {
    pub _super: ConcatZeros_20__2__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_7__7__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_7__7__SuperStruct14Array = [Concatenate_7__7__SuperStruct; 14];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_7__7_Struct {
    pub _super: Concatenate_7__7__SuperStruct14Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_14__8__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_14__8__SuperStruct22Array = [ConcatZeros_14__8__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_14__8_Struct {
    pub _super: ConcatZeros_14__8__SuperStruct22Array,
}
pub type Val6Array = [Val; 6];
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_6_Struct {
    pub _super: NondetRegStruct6Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_6__5__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_6__5__SuperStruct11Array = [ConcatZeros_6__5__SuperStruct; 11];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_6__5_Struct {
    pub _super: ConcatZeros_6__5__SuperStruct11Array,
}
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_0_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_0_Arr12Struct22Array = [EqualWord_0_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_0_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_0_Arr23Struct21Array = [EqualWord_0_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_0_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_0_Arr34Struct21Array = [EqualWord_0_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_1_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_1_Arr12Struct22Array = [EqualWord_1_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_1_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_1_Arr23Struct21Array = [EqualWord_1_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_1_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_1_Arr34Struct21Array = [EqualWord_1_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_2_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_2_Arr12Struct22Array = [EqualWord_2_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_2_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_2_Arr23Struct21Array = [EqualWord_2_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_2_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_2_Arr34Struct21Array = [EqualWord_2_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_3_Arr12Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_3_Arr12Struct22Array = [EqualWord_3_Arr12Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_3_Arr23Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_3_Arr23Struct21Array = [EqualWord_3_Arr23Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct EqualWord_3_Arr34Struct {
    pub _super: NondetRegStruct,
}
pub type EqualWord_3_Arr34Struct21Array = [EqualWord_3_Arr34Struct; 21];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_22_Struct {
    pub _super: NondetRegStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__7__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_9__7__SuperStruct16Array = [ConcatZeros_9__7__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_9__7_Struct {
    pub _super: ConcatZeros_9__7__SuperStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct RotLeft3impl_22__21__21__1_RaStruct {
    pub _super: NondetRegStruct,
}
pub type RotLeft3impl_22__21__21__1_RaStruct22Array = [RotLeft3impl_22__21__21__1_RaStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct RotLeft3impl_22__21__21__1_RbStruct {
    pub _super: NondetRegStruct,
}
pub type RotLeft3impl_22__21__21__1_RbStruct21Array = [RotLeft3impl_22__21__21__1_RbStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct RotLeft3impl_22__21__21__1_RcStruct {
    pub _super: NondetRegStruct,
}
pub type RotLeft3impl_22__21__21__1_RcStruct21Array = [RotLeft3impl_22__21__21__1_RcStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct RotLeft3impl_22__21__21__1_Struct {
    pub ra: RotLeft3impl_22__21__21__1_RaStruct22Array,
    pub rb: RotLeft3impl_22__21__21__1_RbStruct21Array,
    pub rc: RotLeft3impl_22__21__21__1_RcStruct21Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_22__2__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_22__2__SuperStruct24Array = [ConcatZeros_22__2__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_22__2_Struct {
    pub _super: ConcatZeros_22__2__SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_6__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_6__3__SuperStruct9Array = [Concatenate_6__3__SuperStruct; 9];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_6__3_Struct {
    pub _super: Concatenate_6__3__SuperStruct9Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__3__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_3__3__SuperStruct6Array = [Concatenate_3__3__SuperStruct; 6];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_3__3_Struct {
    pub _super: Concatenate_3__3__SuperStruct6Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_15__7__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_15__7__SuperStruct22Array = [Concatenate_15__7__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_15__7_Struct {
    pub _super: Concatenate_15__7__SuperStruct22Array,
}
pub type NondetRegStruct8Array = [NondetRegStruct; 8];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_8__14__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_8__14__SuperStruct22Array = [Concatenate_8__14__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_8__14_Struct {
    pub _super: Concatenate_8__14__SuperStruct22Array,
}
pub type NondetRegStruct1Array = [NondetRegStruct; 1];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_1__21__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_1__21__SuperStruct22Array = [Concatenate_1__21__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_1__21_Struct {
    pub _super: Concatenate_1__21__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_1__15__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_1__15__SuperStruct16Array = [Concatenate_1__15__SuperStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_1__15_Struct {
    pub _super: Concatenate_1__15__SuperStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_16__6__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type Concatenate_16__6__SuperStruct22Array = [Concatenate_16__6__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct Concatenate_16__6_Struct {
    pub _super: Concatenate_16__6__SuperStruct22Array,
}
pub type RegStruct3Array = [RegStruct; 3];
pub type RegStruct7Array = [RegStruct; 7];
pub type RegStruct14Array = [RegStruct; 14];
pub type Val8Array = [Val; 8];
pub type RegStruct8Array = [RegStruct; 8];
pub type Val1Array = [Val; 1];
pub type RegStruct1Array = [RegStruct; 1];
pub type RegStruct6Array = [RegStruct; 6];
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_24__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type OneHotU_24__SuperStruct24Array = [OneHotU_24__SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct OneHotU_24_Struct {
    pub _super: OneHotU_24__SuperStruct24Array,
}
pub type RegStruct16Array = [RegStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_1Struct {
    pub _super: NondetRegStruct,
}
pub type ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_1Struct2Array =
    [ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_1Struct; 2];
#[derive(Copy, Clone, Debug)]
pub struct ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_2Struct {
    pub _super: NondetRegStruct,
}
pub type ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_2Struct22Array =
    [ChiMinor2_4__SuperArm1_SuperArm1_SuperArm1_SuperArm1_SuperRnd_2Struct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_7__15__SuperStruct {
    pub _super: NondetRegStruct,
}
pub type ConcatZeros_7__15__SuperStruct22Array = [ConcatZeros_7__15__SuperStruct; 22];
#[derive(Copy, Clone, Debug)]
pub struct ConcatZeros_7__15_Struct {
    pub _super: ConcatZeros_7__15__SuperStruct22Array,
}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {}
