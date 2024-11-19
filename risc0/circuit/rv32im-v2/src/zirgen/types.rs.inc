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
pub type NondetRegLayout8LayoutArray = [&'static NondetRegLayout; 8];
pub struct OneHot_8_Layout {
    pub _super: &'static NondetRegLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_8_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_8_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct InstInputLayout {
    pub minor_onehot: &'static OneHot_8_Layout,
}
impl risc0_zkp::layout::Component for InstInputLayout {
    fn ty_name(&self) -> &'static str {
        "InstInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("minor_onehot", self.minor_onehot)?;
        Ok(())
    }
}
pub type NondetRegLayout11LayoutArray = [&'static NondetRegLayout; 11];
pub struct OneHot_11_Layout {
    pub _super: &'static NondetRegLayout11LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_11_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_11_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct MemoryArgLayout {
    pub count: &'static NondetRegLayout,
    pub addr: &'static NondetRegLayout,
    pub cycle: &'static NondetRegLayout,
    pub data_low: &'static NondetRegLayout,
    pub data_high: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for MemoryArgLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryArgLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("count", self.count)?;
        v.visit_component("addr", self.addr)?;
        v.visit_component("cycle", self.cycle)?;
        v.visit_component("data_low", self.data_low)?;
        v.visit_component("data_high", self.data_high)?;
        Ok(())
    }
}
pub struct MemoryIOLayout {
    pub old_txn: &'static MemoryArgLayout,
    pub new_txn: &'static MemoryArgLayout,
}
impl risc0_zkp::layout::Component for MemoryIOLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryIOLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("old_txn", self.old_txn)?;
        v.visit_component("new_txn", self.new_txn)?;
        Ok(())
    }
}
pub struct CycleArgLayout {
    pub count: &'static NondetRegLayout,
    pub cycle: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for CycleArgLayout {
    fn ty_name(&self) -> &'static str {
        "CycleArgLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("count", self.count)?;
        v.visit_component("cycle", self.cycle)?;
        Ok(())
    }
}
pub struct IsCycleLayout {
    pub arg: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for IsCycleLayout {
    fn ty_name(&self) -> &'static str {
        "IsCycleLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub struct IsForwardLayout {
    pub _0: &'static IsCycleLayout,
}
impl risc0_zkp::layout::Component for IsForwardLayout {
    fn ty_name(&self) -> &'static str {
        "IsForwardLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct MemoryWriteLayout {
    pub io: &'static MemoryIOLayout,
    pub _0: &'static IsForwardLayout,
}
impl risc0_zkp::layout::Component for MemoryWriteLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryWriteLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("io", self.io)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct IsZeroLayout {
    pub _super: &'static NondetRegLayout,
    pub inv: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for IsZeroLayout {
    fn ty_name(&self) -> &'static str {
        "IsZeroLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv", self.inv)?;
        Ok(())
    }
}
pub struct WriteRdLayout {
    pub _0: &'static MemoryWriteLayout,
    pub is_rd0: &'static IsZeroLayout,
    pub write_addr: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for WriteRdLayout {
    fn ty_name(&self) -> &'static str {
        "WriteRdLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("is_rd0", self.is_rd0)?;
        v.visit_component("write_addr", self.write_addr)?;
        Ok(())
    }
}
pub struct ArgU16Layout {
    pub count: &'static NondetRegLayout,
    pub val: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ArgU16Layout {
    fn ty_name(&self) -> &'static str {
        "ArgU16Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("count", self.count)?;
        v.visit_component("val", self.val)?;
        Ok(())
    }
}
pub struct NondetU16RegLayout {
    pub arg: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for NondetU16RegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetU16RegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub struct NormalizeU32Layout {
    pub low16: &'static NondetU16RegLayout,
    pub high16: &'static NondetU16RegLayout,
    pub low_carry: &'static NondetRegLayout,
    pub high_carry: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for NormalizeU32Layout {
    fn ty_name(&self) -> &'static str {
        "NormalizeU32Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low16", self.low16)?;
        v.visit_component("high16", self.high16)?;
        v.visit_component("low_carry", self.low_carry)?;
        v.visit_component("high_carry", self.high_carry)?;
        Ok(())
    }
}
pub struct FinalizeMiscLayout {
    pub _0: &'static WriteRdLayout,
    pub write_data: &'static NormalizeU32Layout,
    pub pc_norm: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for FinalizeMiscLayout {
    fn ty_name(&self) -> &'static str {
        "FinalizeMiscLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("write_data", self.write_data)?;
        v.visit_component("pc_norm", self.pc_norm)?;
        Ok(())
    }
}
pub struct Misc0Arm0Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc0Arm1Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub type NondetRegLayout16LayoutArray = [&'static NondetRegLayout; 16];
pub struct ToBits_16_Layout {
    pub _super: &'static NondetRegLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ToBits_16_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBits_16_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct BitwiseAndU16Layout {
    pub bits_x: &'static ToBits_16_Layout,
    pub bits_y: &'static ToBits_16_Layout,
}
impl risc0_zkp::layout::Component for BitwiseAndU16Layout {
    fn ty_name(&self) -> &'static str {
        "BitwiseAndU16Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("bits_x", self.bits_x)?;
        v.visit_component("bits_y", self.bits_y)?;
        Ok(())
    }
}
pub struct BitwiseAndLayout {
    pub _0: &'static BitwiseAndU16Layout,
    pub _1: &'static BitwiseAndU16Layout,
}
impl risc0_zkp::layout::Component for BitwiseAndLayout {
    fn ty_name(&self) -> &'static str {
        "BitwiseAndLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct BitwiseXorLayout {
    pub and_xy: &'static BitwiseAndLayout,
}
impl risc0_zkp::layout::Component for BitwiseXorLayout {
    fn ty_name(&self) -> &'static str {
        "BitwiseXorLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("and_xy", self.and_xy)?;
        Ok(())
    }
}
pub struct OpXORLayout {
    pub _0: &'static BitwiseXorLayout,
}
impl risc0_zkp::layout::Component for OpXORLayout {
    fn ty_name(&self) -> &'static str {
        "OpXORLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc0Arm2Layout {
    pub _super: &'static OpXORLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct BitwiseOrLayout {
    pub and_xy: &'static BitwiseAndLayout,
}
impl risc0_zkp::layout::Component for BitwiseOrLayout {
    fn ty_name(&self) -> &'static str {
        "BitwiseOrLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("and_xy", self.and_xy)?;
        Ok(())
    }
}
pub struct OpORLayout {
    pub _0: &'static BitwiseOrLayout,
}
impl risc0_zkp::layout::Component for OpORLayout {
    fn ty_name(&self) -> &'static str {
        "OpORLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc0Arm3Layout {
    pub _super: &'static OpORLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpANDLayout {
    pub _0: &'static BitwiseAndLayout,
}
impl risc0_zkp::layout::Component for OpANDLayout {
    fn ty_name(&self) -> &'static str {
        "OpANDLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc0Arm4Layout {
    pub _super: &'static OpANDLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct GetSignU32Layout {
    pub _super: &'static NondetRegLayout,
    pub rest_times_two: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for GetSignU32Layout {
    fn ty_name(&self) -> &'static str {
        "GetSignU32Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("rest_times_two", self.rest_times_two)?;
        Ok(())
    }
}
pub struct CmpLessThanLayout {
    pub diff: &'static NormalizeU32Layout,
    pub s1: &'static GetSignU32Layout,
    pub s2: &'static GetSignU32Layout,
    pub s3: &'static GetSignU32Layout,
    pub overflow: &'static NondetRegLayout,
    pub is_less_than: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for CmpLessThanLayout {
    fn ty_name(&self) -> &'static str {
        "CmpLessThanLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("diff", self.diff)?;
        v.visit_component("s1", self.s1)?;
        v.visit_component("s2", self.s2)?;
        v.visit_component("s3", self.s3)?;
        v.visit_component("overflow", self.overflow)?;
        v.visit_component("is_less_than", self.is_less_than)?;
        Ok(())
    }
}
pub struct OpSLTLayout {
    pub cmp: &'static CmpLessThanLayout,
}
impl risc0_zkp::layout::Component for OpSLTLayout {
    fn ty_name(&self) -> &'static str {
        "OpSLTLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct CmpLessThanUnsignedLayout {
    pub diff: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for CmpLessThanUnsignedLayout {
    fn ty_name(&self) -> &'static str {
        "CmpLessThanUnsignedLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("diff", self.diff)?;
        Ok(())
    }
}
pub struct OpSLTULayout {
    pub cmp: &'static CmpLessThanUnsignedLayout,
}
impl risc0_zkp::layout::Component for OpSLTULayout {
    fn ty_name(&self) -> &'static str {
        "OpSLTULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc0Arm6Layout {
    pub _super: &'static OpSLTULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Misc0Arm7Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc0MiscOutputLayout {
    pub arm0: &'static Misc0Arm0Layout,
    pub arm1: &'static Misc0Arm1Layout,
    pub arm2: &'static Misc0Arm2Layout,
    pub arm3: &'static Misc0Arm3Layout,
    pub arm4: &'static Misc0Arm4Layout,
    pub arm5: &'static OpSLTLayout,
    pub arm6: &'static Misc0Arm6Layout,
    pub arm7: &'static Misc0Arm7Layout,
}
impl risc0_zkp::layout::Component for Misc0MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "Misc0MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub type ArgU16Layout5LayoutArray = [&'static ArgU16Layout; 5];
pub struct _Arguments_Misc0MiscOutputLayout {
    pub arg_u16: &'static ArgU16Layout5LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Misc0MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Misc0MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct DecoderLayout {
    pub _f7_6: &'static NondetRegLayout,
    pub _f7_45: &'static NondetRegLayout,
    pub _f7_23: &'static NondetRegLayout,
    pub _f7_01: &'static NondetRegLayout,
    pub _rs2_34: &'static NondetRegLayout,
    pub _rs2_12: &'static NondetRegLayout,
    pub _rs2_0: &'static NondetRegLayout,
    pub _rs1_34: &'static NondetRegLayout,
    pub _rs1_12: &'static NondetRegLayout,
    pub _rs1_0: &'static NondetRegLayout,
    pub _f3_2: &'static NondetRegLayout,
    pub _f3_01: &'static NondetRegLayout,
    pub _rd_34: &'static NondetRegLayout,
    pub _rd_12: &'static NondetRegLayout,
    pub _rd_0: &'static NondetRegLayout,
    pub opcode: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for DecoderLayout {
    fn ty_name(&self) -> &'static str {
        "DecoderLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_f7_6", self._f7_6)?;
        v.visit_component("_f7_45", self._f7_45)?;
        v.visit_component("_f7_23", self._f7_23)?;
        v.visit_component("_f7_01", self._f7_01)?;
        v.visit_component("_rs2_34", self._rs2_34)?;
        v.visit_component("_rs2_12", self._rs2_12)?;
        v.visit_component("_rs2_0", self._rs2_0)?;
        v.visit_component("_rs1_34", self._rs1_34)?;
        v.visit_component("_rs1_12", self._rs1_12)?;
        v.visit_component("_rs1_0", self._rs1_0)?;
        v.visit_component("_f3_2", self._f3_2)?;
        v.visit_component("_f3_01", self._f3_01)?;
        v.visit_component("_rd_34", self._rd_34)?;
        v.visit_component("_rd_12", self._rd_12)?;
        v.visit_component("_rd_0", self._rd_0)?;
        v.visit_component("opcode", self.opcode)?;
        Ok(())
    }
}
pub struct U16RegLayout {
    pub ret: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for U16RegLayout {
    fn ty_name(&self) -> &'static str {
        "U16RegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("ret", self.ret)?;
        Ok(())
    }
}
pub struct AddrDecomposeLayout {
    pub low2: &'static NondetRegLayout,
    pub upper_diff: &'static U16RegLayout,
    pub _0: &'static IsZeroLayout,
    pub med14: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for AddrDecomposeLayout {
    fn ty_name(&self) -> &'static str {
        "AddrDecomposeLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low2", self.low2)?;
        v.visit_component("upper_diff", self.upper_diff)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("med14", self.med14)?;
        Ok(())
    }
}
pub struct MemoryReadLayout {
    pub io: &'static MemoryIOLayout,
    pub _0: &'static IsForwardLayout,
}
impl risc0_zkp::layout::Component for MemoryReadLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryReadLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("io", self.io)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct DecodeInstLayout {
    pub _super: &'static DecoderLayout,
    pub arg: &'static CycleArgLayout,
    pub pc_addr: &'static AddrDecomposeLayout,
    pub load_inst: &'static MemoryReadLayout,
}
impl risc0_zkp::layout::Component for DecodeInstLayout {
    fn ty_name(&self) -> &'static str {
        "DecodeInstLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arg", self.arg)?;
        v.visit_component("pc_addr", self.pc_addr)?;
        v.visit_component("load_inst", self.load_inst)?;
        Ok(())
    }
}
pub struct ReadRegLayout {
    pub _super: &'static MemoryReadLayout,
    pub addr: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ReadRegLayout {
    fn ty_name(&self) -> &'static str {
        "ReadRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("addr", self.addr)?;
        Ok(())
    }
}
pub struct MiscInputLayout {
    pub decoded: &'static DecodeInstLayout,
    pub rs1: &'static ReadRegLayout,
    pub rs2: &'static ReadRegLayout,
}
impl risc0_zkp::layout::Component for MiscInputLayout {
    fn ty_name(&self) -> &'static str {
        "MiscInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("decoded", self.decoded)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("rs2", self.rs2)?;
        Ok(())
    }
}
pub struct Misc0Layout {
    pub _super: &'static FinalizeMiscLayout,
    pub misc_output: &'static Misc0MiscOutputLayout,
    pub _arguments_misc_output: &'static _Arguments_Misc0MiscOutputLayout,
    pub input: &'static MiscInputLayout,
}
impl risc0_zkp::layout::Component for Misc0Layout {
    fn ty_name(&self) -> &'static str {
        "Misc0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("misc_output", self.misc_output)?;
        v.visit_component("_arguments_misc_output", self._arguments_misc_output)?;
        v.visit_component("input", self.input)?;
        Ok(())
    }
}
pub struct OpXORILayout {
    pub _0: &'static BitwiseXorLayout,
}
impl risc0_zkp::layout::Component for OpXORILayout {
    fn ty_name(&self) -> &'static str {
        "OpXORILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc1Arm0Layout {
    pub _super: &'static OpXORILayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpORILayout {
    pub _0: &'static BitwiseOrLayout,
}
impl risc0_zkp::layout::Component for OpORILayout {
    fn ty_name(&self) -> &'static str {
        "OpORILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc1Arm1Layout {
    pub _super: &'static OpORILayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpANDILayout {
    pub _0: &'static BitwiseAndLayout,
}
impl risc0_zkp::layout::Component for OpANDILayout {
    fn ty_name(&self) -> &'static str {
        "OpANDILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Misc1Arm2Layout {
    pub _super: &'static OpANDILayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpSLTILayout {
    pub cmp: &'static CmpLessThanLayout,
}
impl risc0_zkp::layout::Component for OpSLTILayout {
    fn ty_name(&self) -> &'static str {
        "OpSLTILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct OpSLTIULayout {
    pub cmp: &'static CmpLessThanUnsignedLayout,
}
impl risc0_zkp::layout::Component for OpSLTIULayout {
    fn ty_name(&self) -> &'static str {
        "OpSLTIULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc1Arm4Layout {
    pub _super: &'static OpSLTIULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct CmpEqualLayout {
    pub low_same: &'static IsZeroLayout,
    pub high_same: &'static IsZeroLayout,
    pub is_equal: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for CmpEqualLayout {
    fn ty_name(&self) -> &'static str {
        "CmpEqualLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low_same", self.low_same)?;
        v.visit_component("high_same", self.high_same)?;
        v.visit_component("is_equal", self.is_equal)?;
        Ok(())
    }
}
pub struct OpBEQLayout {
    pub cmp: &'static CmpEqualLayout,
}
impl risc0_zkp::layout::Component for OpBEQLayout {
    fn ty_name(&self) -> &'static str {
        "OpBEQLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc1Arm5Layout {
    pub _super: &'static OpBEQLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpBNELayout {
    pub cmp: &'static CmpEqualLayout,
}
impl risc0_zkp::layout::Component for OpBNELayout {
    fn ty_name(&self) -> &'static str {
        "OpBNELayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc1Arm6Layout {
    pub _super: &'static OpBNELayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc1Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct OpBLTLayout {
    pub cmp: &'static CmpLessThanLayout,
}
impl risc0_zkp::layout::Component for OpBLTLayout {
    fn ty_name(&self) -> &'static str {
        "OpBLTLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc1MiscOutputLayout {
    pub arm0: &'static Misc1Arm0Layout,
    pub arm1: &'static Misc1Arm1Layout,
    pub arm2: &'static Misc1Arm2Layout,
    pub arm3: &'static OpSLTILayout,
    pub arm4: &'static Misc1Arm4Layout,
    pub arm5: &'static Misc1Arm5Layout,
    pub arm6: &'static Misc1Arm6Layout,
    pub arm7: &'static OpBLTLayout,
}
impl risc0_zkp::layout::Component for Misc1MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "Misc1MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub struct _Arguments_Misc1MiscOutputLayout {
    pub arg_u16: &'static ArgU16Layout5LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Misc1MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Misc1MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct Misc1Layout {
    pub _super: &'static FinalizeMiscLayout,
    pub misc_output: &'static Misc1MiscOutputLayout,
    pub _arguments_misc_output: &'static _Arguments_Misc1MiscOutputLayout,
    pub input: &'static MiscInputLayout,
}
impl risc0_zkp::layout::Component for Misc1Layout {
    fn ty_name(&self) -> &'static str {
        "Misc1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("misc_output", self.misc_output)?;
        v.visit_component("_arguments_misc_output", self._arguments_misc_output)?;
        v.visit_component("input", self.input)?;
        Ok(())
    }
}
pub struct OpBGELayout {
    pub cmp: &'static CmpLessThanLayout,
}
impl risc0_zkp::layout::Component for OpBGELayout {
    fn ty_name(&self) -> &'static str {
        "OpBGELayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct OpBLTULayout {
    pub cmp: &'static CmpLessThanUnsignedLayout,
}
impl risc0_zkp::layout::Component for OpBLTULayout {
    fn ty_name(&self) -> &'static str {
        "OpBLTULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc2Arm1Layout {
    pub _super: &'static OpBLTULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct OpBGEULayout {
    pub cmp: &'static CmpLessThanUnsignedLayout,
}
impl risc0_zkp::layout::Component for OpBGEULayout {
    fn ty_name(&self) -> &'static str {
        "OpBGEULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cmp", self.cmp)?;
        Ok(())
    }
}
pub struct Misc2Arm2Layout {
    pub _super: &'static OpBGEULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Misc2Arm3Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc2Arm4Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc2Arm5Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc2Arm6Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc2Arm7Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Misc2Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        Ok(())
    }
}
pub struct Misc2MiscOutputLayout {
    pub arm0: &'static OpBGELayout,
    pub arm1: &'static Misc2Arm1Layout,
    pub arm2: &'static Misc2Arm2Layout,
    pub arm3: &'static Misc2Arm3Layout,
    pub arm4: &'static Misc2Arm4Layout,
    pub arm5: &'static Misc2Arm5Layout,
    pub arm6: &'static Misc2Arm6Layout,
    pub arm7: &'static Misc2Arm7Layout,
}
impl risc0_zkp::layout::Component for Misc2MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "Misc2MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub struct _Arguments_Misc2MiscOutputLayout {
    pub arg_u16: &'static ArgU16Layout5LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Misc2MiscOutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Misc2MiscOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct Misc2Layout {
    pub _super: &'static FinalizeMiscLayout,
    pub misc_output: &'static Misc2MiscOutputLayout,
    pub _arguments_misc_output: &'static _Arguments_Misc2MiscOutputLayout,
    pub input: &'static MiscInputLayout,
}
impl risc0_zkp::layout::Component for Misc2Layout {
    fn ty_name(&self) -> &'static str {
        "Misc2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("misc_output", self.misc_output)?;
        v.visit_component("_arguments_misc_output", self._arguments_misc_output)?;
        v.visit_component("input", self.input)?;
        Ok(())
    }
}
pub struct MulInputLayout {
    pub decoded: &'static DecodeInstLayout,
    pub rs1: &'static ReadRegLayout,
    pub rs2: &'static ReadRegLayout,
}
impl risc0_zkp::layout::Component for MulInputLayout {
    fn ty_name(&self) -> &'static str {
        "MulInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("decoded", self.decoded)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("rs2", self.rs2)?;
        Ok(())
    }
}
pub struct ArgU8Layout {
    pub count: &'static NondetRegLayout,
    pub val: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ArgU8Layout {
    fn ty_name(&self) -> &'static str {
        "ArgU8Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("count", self.count)?;
        v.visit_component("val", self.val)?;
        Ok(())
    }
}
pub struct NondetU8RegLayout {
    pub arg: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for NondetU8RegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetU8RegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub struct ExpandU32Layout {
    pub b0: &'static NondetU8RegLayout,
    pub b1: &'static NondetU8RegLayout,
    pub b2: &'static NondetU8RegLayout,
    pub b3: &'static NondetU8RegLayout,
    pub b3_top7times2: &'static NondetU8RegLayout,
    pub top_bit: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ExpandU32Layout {
    fn ty_name(&self) -> &'static str {
        "ExpandU32Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("b0", self.b0)?;
        v.visit_component("b1", self.b1)?;
        v.visit_component("b2", self.b2)?;
        v.visit_component("b3", self.b3)?;
        v.visit_component("b3_top7times2", self.b3_top7times2)?;
        v.visit_component("top_bit", self.top_bit)?;
        Ok(())
    }
}
pub struct NondetFakeTwitRegLayout {
    pub reg0: &'static NondetRegLayout,
    pub reg1: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for NondetFakeTwitRegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetFakeTwitRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("reg0", self.reg0)?;
        v.visit_component("reg1", self.reg1)?;
        Ok(())
    }
}
pub struct SplitTotalLayout {
    pub out: &'static NondetU16RegLayout,
    pub carry_byte: &'static NondetU8RegLayout,
    pub carry_extra: &'static NondetFakeTwitRegLayout,
}
impl risc0_zkp::layout::Component for SplitTotalLayout {
    fn ty_name(&self) -> &'static str {
        "SplitTotalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("out", self.out)?;
        v.visit_component("carry_byte", self.carry_byte)?;
        v.visit_component("carry_extra", self.carry_extra)?;
        Ok(())
    }
}
pub struct MultiplyAccumulateLayout {
    pub ax: &'static ExpandU32Layout,
    pub bx: &'static ExpandU32Layout,
    pub c_sign: &'static NondetRegLayout,
    pub c_rest_times2: &'static NondetU16RegLayout,
    pub s0: &'static SplitTotalLayout,
    pub s1: &'static SplitTotalLayout,
    pub s2: &'static SplitTotalLayout,
    pub s3_out: &'static NondetU16RegLayout,
    pub s3_carry: &'static NondetFakeTwitRegLayout,
}
impl risc0_zkp::layout::Component for MultiplyAccumulateLayout {
    fn ty_name(&self) -> &'static str {
        "MultiplyAccumulateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("ax", self.ax)?;
        v.visit_component("bx", self.bx)?;
        v.visit_component("c_sign", self.c_sign)?;
        v.visit_component("c_rest_times2", self.c_rest_times2)?;
        v.visit_component("s0", self.s0)?;
        v.visit_component("s1", self.s1)?;
        v.visit_component("s2", self.s2)?;
        v.visit_component("s3_out", self.s3_out)?;
        v.visit_component("s3_carry", self.s3_carry)?;
        Ok(())
    }
}
pub struct DoMulLayout {
    pub mul: &'static MultiplyAccumulateLayout,
}
impl risc0_zkp::layout::Component for DoMulLayout {
    fn ty_name(&self) -> &'static str {
        "DoMulLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("mul", self.mul)?;
        Ok(())
    }
}
pub type NondetRegLayout5LayoutArray = [&'static NondetRegLayout; 5];
pub struct ToBits_5_Layout {
    pub _super: &'static NondetRegLayout5LayoutArray,
}
impl risc0_zkp::layout::Component for ToBits_5_Layout {
    fn ty_name(&self) -> &'static str {
        "ToBits_5_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct DynPo2Layout {
    pub low5: &'static ToBits_5_Layout,
    pub check_u16: &'static NondetU16RegLayout,
    pub b3: &'static NondetRegLayout,
    pub low: &'static NondetRegLayout,
    pub high: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for DynPo2Layout {
    fn ty_name(&self) -> &'static str {
        "DynPo2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low5", self.low5)?;
        v.visit_component("check_u16", self.check_u16)?;
        v.visit_component("b3", self.b3)?;
        v.visit_component("low", self.low)?;
        v.visit_component("high", self.high)?;
        Ok(())
    }
}
pub struct OpSLLLayout {
    pub _0: &'static DoMulLayout,
    pub shift_mul: &'static DynPo2Layout,
}
impl risc0_zkp::layout::Component for OpSLLLayout {
    fn ty_name(&self) -> &'static str {
        "OpSLLLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        Ok(())
    }
}
pub struct OpSLLILayout {
    pub _0: &'static DoMulLayout,
    pub shift_mul: &'static DynPo2Layout,
}
impl risc0_zkp::layout::Component for OpSLLILayout {
    fn ty_name(&self) -> &'static str {
        "OpSLLILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        Ok(())
    }
}
pub struct OpMULLayout {
    pub _0: &'static DoMulLayout,
}
impl risc0_zkp::layout::Component for OpMULLayout {
    fn ty_name(&self) -> &'static str {
        "OpMULLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Mul0Arm2Layout {
    pub _super: &'static OpMULLayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct OpMULHLayout {
    pub _0: &'static DoMulLayout,
}
impl risc0_zkp::layout::Component for OpMULHLayout {
    fn ty_name(&self) -> &'static str {
        "OpMULHLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Mul0Arm3Layout {
    pub _super: &'static OpMULHLayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct OpMULHSULayout {
    pub _0: &'static DoMulLayout,
}
impl risc0_zkp::layout::Component for OpMULHSULayout {
    fn ty_name(&self) -> &'static str {
        "OpMULHSULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Mul0Arm4Layout {
    pub _super: &'static OpMULHSULayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct OpMULHULayout {
    pub _0: &'static DoMulLayout,
}
impl risc0_zkp::layout::Component for OpMULHULayout {
    fn ty_name(&self) -> &'static str {
        "OpMULHULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Mul0Arm5Layout {
    pub _super: &'static OpMULHULayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Mul0Arm6Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra6: &'static ArgU8Layout,
    pub _extra7: &'static ArgU8Layout,
    pub _extra8: &'static ArgU8Layout,
    pub _extra9: &'static ArgU8Layout,
    pub _extra10: &'static ArgU8Layout,
    pub _extra11: &'static ArgU8Layout,
    pub _extra12: &'static ArgU8Layout,
    pub _extra13: &'static ArgU8Layout,
    pub _extra14: &'static ArgU8Layout,
    pub _extra15: &'static ArgU8Layout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra18: &'static ArgU8Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        Ok(())
    }
}
pub struct Mul0Arm7Layout {
    pub _extra0: &'static ArgU16Layout,
    pub _extra6: &'static ArgU8Layout,
    pub _extra7: &'static ArgU8Layout,
    pub _extra8: &'static ArgU8Layout,
    pub _extra9: &'static ArgU8Layout,
    pub _extra10: &'static ArgU8Layout,
    pub _extra11: &'static ArgU8Layout,
    pub _extra12: &'static ArgU8Layout,
    pub _extra13: &'static ArgU8Layout,
    pub _extra14: &'static ArgU8Layout,
    pub _extra15: &'static ArgU8Layout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra18: &'static ArgU8Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Mul0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        Ok(())
    }
}
pub struct Mul0MulOutputLayout {
    pub arm0: &'static OpSLLLayout,
    pub arm1: &'static OpSLLILayout,
    pub arm2: &'static Mul0Arm2Layout,
    pub arm3: &'static Mul0Arm3Layout,
    pub arm4: &'static Mul0Arm4Layout,
    pub arm5: &'static Mul0Arm5Layout,
    pub arm6: &'static Mul0Arm6Layout,
    pub arm7: &'static Mul0Arm7Layout,
}
impl risc0_zkp::layout::Component for Mul0MulOutputLayout {
    fn ty_name(&self) -> &'static str {
        "Mul0MulOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub type ArgU16Layout6LayoutArray = [&'static ArgU16Layout; 6];
pub type ArgU8Layout13LayoutArray = [&'static ArgU8Layout; 13];
pub struct _Arguments_Mul0MulOutputLayout {
    pub arg_u16: &'static ArgU16Layout6LayoutArray,
    pub arg_u8: &'static ArgU8Layout13LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Mul0MulOutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Mul0MulOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct Mul0Layout {
    pub input: &'static MulInputLayout,
    pub mul_output: &'static Mul0MulOutputLayout,
    pub _0: &'static WriteRdLayout,
    pub _arguments_mul_output: &'static _Arguments_Mul0MulOutputLayout,
    pub pc_add: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for Mul0Layout {
    fn ty_name(&self) -> &'static str {
        "Mul0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("input", self.input)?;
        v.visit_component("mul_output", self.mul_output)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_arguments_mul_output", self._arguments_mul_output)?;
        v.visit_component("pc_add", self.pc_add)?;
        Ok(())
    }
}
pub struct DoDivLayout {
    pub quot_low: &'static NondetRegLayout,
    pub quot_high: &'static NondetRegLayout,
    pub rem_low: &'static NondetU16RegLayout,
    pub rem_high: &'static NondetU16RegLayout,
    pub mul: &'static MultiplyAccumulateLayout,
    pub top_bit_type: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for DoDivLayout {
    fn ty_name(&self) -> &'static str {
        "DoDivLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("quot_low", self.quot_low)?;
        v.visit_component("quot_high", self.quot_high)?;
        v.visit_component("rem_low", self.rem_low)?;
        v.visit_component("rem_high", self.rem_high)?;
        v.visit_component("mul", self.mul)?;
        v.visit_component("top_bit_type", self.top_bit_type)?;
        Ok(())
    }
}
pub struct OpSRLLayout {
    pub _0: &'static DoDivLayout,
    pub shift_mul: &'static DynPo2Layout,
}
impl risc0_zkp::layout::Component for OpSRLLayout {
    fn ty_name(&self) -> &'static str {
        "OpSRLLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        Ok(())
    }
}
pub struct Div0Arm0Layout {
    pub _super: &'static OpSRLLayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct TopBitLayout {
    pub _super: &'static NondetRegLayout,
    pub rest: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for TopBitLayout {
    fn ty_name(&self) -> &'static str {
        "TopBitLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("rest", self.rest)?;
        Ok(())
    }
}
pub struct OpSRALayout {
    pub _0: &'static DoDivLayout,
    pub shift_mul: &'static DynPo2Layout,
    pub flip: &'static TopBitLayout,
}
impl risc0_zkp::layout::Component for OpSRALayout {
    fn ty_name(&self) -> &'static str {
        "OpSRALayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        v.visit_component("flip", self.flip)?;
        Ok(())
    }
}
pub struct OpSRLILayout {
    pub _0: &'static DoDivLayout,
    pub shift_mul: &'static DynPo2Layout,
}
impl risc0_zkp::layout::Component for OpSRLILayout {
    fn ty_name(&self) -> &'static str {
        "OpSRLILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        Ok(())
    }
}
pub struct Div0Arm2Layout {
    pub _super: &'static OpSRLILayout,
    pub _extra0: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct OpSRAILayout {
    pub _0: &'static DoDivLayout,
    pub shift_mul: &'static DynPo2Layout,
    pub flip: &'static TopBitLayout,
}
impl risc0_zkp::layout::Component for OpSRAILayout {
    fn ty_name(&self) -> &'static str {
        "OpSRAILayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("shift_mul", self.shift_mul)?;
        v.visit_component("flip", self.flip)?;
        Ok(())
    }
}
pub struct OpDIVLayout {
    pub _0: &'static DoDivLayout,
}
impl risc0_zkp::layout::Component for OpDIVLayout {
    fn ty_name(&self) -> &'static str {
        "OpDIVLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Div0Arm4Layout {
    pub _super: &'static OpDIVLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct OpDIVULayout {
    pub _0: &'static DoDivLayout,
}
impl risc0_zkp::layout::Component for OpDIVULayout {
    fn ty_name(&self) -> &'static str {
        "OpDIVULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Div0Arm5Layout {
    pub _super: &'static OpDIVULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct OpREMLayout {
    pub _0: &'static DoDivLayout,
}
impl risc0_zkp::layout::Component for OpREMLayout {
    fn ty_name(&self) -> &'static str {
        "OpREMLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Div0Arm6Layout {
    pub _super: &'static OpREMLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct OpREMULayout {
    pub _0: &'static DoDivLayout,
}
impl risc0_zkp::layout::Component for OpREMULayout {
    fn ty_name(&self) -> &'static str {
        "OpREMULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Div0Arm7Layout {
    pub _super: &'static OpREMULayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Div0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Div0MulOutputLayout {
    pub arm0: &'static Div0Arm0Layout,
    pub arm1: &'static OpSRALayout,
    pub arm2: &'static Div0Arm2Layout,
    pub arm3: &'static OpSRAILayout,
    pub arm4: &'static Div0Arm4Layout,
    pub arm5: &'static Div0Arm5Layout,
    pub arm6: &'static Div0Arm6Layout,
    pub arm7: &'static Div0Arm7Layout,
}
impl risc0_zkp::layout::Component for Div0MulOutputLayout {
    fn ty_name(&self) -> &'static str {
        "Div0MulOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub struct DivInputLayout {
    pub decoded: &'static DecodeInstLayout,
    pub rs1: &'static ReadRegLayout,
    pub rs2: &'static ReadRegLayout,
}
impl risc0_zkp::layout::Component for DivInputLayout {
    fn ty_name(&self) -> &'static str {
        "DivInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("decoded", self.decoded)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("rs2", self.rs2)?;
        Ok(())
    }
}
pub type ArgU16Layout9LayoutArray = [&'static ArgU16Layout; 9];
pub struct _Arguments_Div0MulOutputLayout {
    pub arg_u8: &'static ArgU8Layout13LayoutArray,
    pub arg_u16: &'static ArgU16Layout9LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Div0MulOutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Div0MulOutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u8", self.arg_u8)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct Div0Layout {
    pub mul_output: &'static Div0MulOutputLayout,
    pub input: &'static DivInputLayout,
    pub _0: &'static WriteRdLayout,
    pub _arguments_mul_output: &'static _Arguments_Div0MulOutputLayout,
    pub pc_add: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for Div0Layout {
    fn ty_name(&self) -> &'static str {
        "Div0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("mul_output", self.mul_output)?;
        v.visit_component("input", self.input)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_arguments_mul_output", self._arguments_mul_output)?;
        v.visit_component("pc_add", self.pc_add)?;
        Ok(())
    }
}
pub struct AddrDecomposeBitsLayout {
    pub low0: &'static NondetRegLayout,
    pub low1: &'static NondetRegLayout,
    pub upper_diff: &'static U16RegLayout,
    pub _0: &'static IsZeroLayout,
    pub med14: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for AddrDecomposeBitsLayout {
    fn ty_name(&self) -> &'static str {
        "AddrDecomposeBitsLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low0", self.low0)?;
        v.visit_component("low1", self.low1)?;
        v.visit_component("upper_diff", self.upper_diff)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("med14", self.med14)?;
        Ok(())
    }
}
pub struct MemLoadInputLayout {
    pub decoded: &'static DecodeInstLayout,
    pub rs1: &'static ReadRegLayout,
    pub data_0: &'static MemoryReadLayout,
    pub addr: &'static AddrDecomposeBitsLayout,
    pub addr_u32: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for MemLoadInputLayout {
    fn ty_name(&self) -> &'static str {
        "MemLoadInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("decoded", self.decoded)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("data_0", self.data_0)?;
        v.visit_component("addr", self.addr)?;
        v.visit_component("addr_u32", self.addr_u32)?;
        Ok(())
    }
}
pub struct SplitWordLayout {
    pub byte0: &'static NondetU8RegLayout,
    pub byte1: &'static NondetU8RegLayout,
}
impl risc0_zkp::layout::Component for SplitWordLayout {
    fn ty_name(&self) -> &'static str {
        "SplitWordLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("byte0", self.byte0)?;
        v.visit_component("byte1", self.byte1)?;
        Ok(())
    }
}
pub struct OpLBLayout {
    pub bytes: &'static SplitWordLayout,
    pub low7x2: &'static NondetU8RegLayout,
    pub high_bit: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for OpLBLayout {
    fn ty_name(&self) -> &'static str {
        "OpLBLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("bytes", self.bytes)?;
        v.visit_component("low7x2", self.low7x2)?;
        v.visit_component("high_bit", self.high_bit)?;
        Ok(())
    }
}
pub struct OpLHLayout {
    pub low15x2: &'static NondetU8RegLayout,
    pub high_bit: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for OpLHLayout {
    fn ty_name(&self) -> &'static str {
        "OpLHLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low15x2", self.low15x2)?;
        v.visit_component("high_bit", self.high_bit)?;
        Ok(())
    }
}
pub struct Mem0Arm1Layout {
    pub _super: &'static OpLHLayout,
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Mem0Arm2Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct OpLBULayout {
    pub bytes: &'static SplitWordLayout,
}
impl risc0_zkp::layout::Component for OpLBULayout {
    fn ty_name(&self) -> &'static str {
        "OpLBULayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("bytes", self.bytes)?;
        Ok(())
    }
}
pub struct Mem0Arm3Layout {
    pub _super: &'static OpLBULayout,
    pub _extra0: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct Mem0Arm4Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Mem0Arm5Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Mem0Arm6Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Mem0Arm7Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        Ok(())
    }
}
pub struct Mem0OutputLayout {
    pub arm0: &'static OpLBLayout,
    pub arm1: &'static Mem0Arm1Layout,
    pub arm2: &'static Mem0Arm2Layout,
    pub arm3: &'static Mem0Arm3Layout,
    pub arm4: &'static Mem0Arm4Layout,
    pub arm5: &'static Mem0Arm5Layout,
    pub arm6: &'static Mem0Arm6Layout,
    pub arm7: &'static Mem0Arm7Layout,
}
impl risc0_zkp::layout::Component for Mem0OutputLayout {
    fn ty_name(&self) -> &'static str {
        "Mem0OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub type ArgU8Layout3LayoutArray = [&'static ArgU8Layout; 3];
pub struct _Arguments_Mem0OutputLayout {
    pub arg_u8: &'static ArgU8Layout3LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Mem0OutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Mem0OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct Mem0Layout {
    pub input: &'static MemLoadInputLayout,
    pub _0: &'static WriteRdLayout,
    pub output: &'static Mem0OutputLayout,
    pub _arguments_output: &'static _Arguments_Mem0OutputLayout,
    pub pc_add: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for Mem0Layout {
    fn ty_name(&self) -> &'static str {
        "Mem0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("input", self.input)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("output", self.output)?;
        v.visit_component("_arguments_output", self._arguments_output)?;
        v.visit_component("pc_add", self.pc_add)?;
        Ok(())
    }
}
pub struct MemStoreInputLayout {
    pub decoded: &'static DecodeInstLayout,
    pub rs1: &'static ReadRegLayout,
    pub rs2: &'static ReadRegLayout,
    pub data_0: &'static MemoryReadLayout,
    pub addr: &'static AddrDecomposeBitsLayout,
    pub addr_u32: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for MemStoreInputLayout {
    fn ty_name(&self) -> &'static str {
        "MemStoreInputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("decoded", self.decoded)?;
        v.visit_component("rs1", self.rs1)?;
        v.visit_component("rs2", self.rs2)?;
        v.visit_component("data_0", self.data_0)?;
        v.visit_component("addr", self.addr)?;
        v.visit_component("addr_u32", self.addr_u32)?;
        Ok(())
    }
}
pub struct MemStoreFinalizeLayout {
    pub _0: &'static MemoryWriteLayout,
}
impl risc0_zkp::layout::Component for MemStoreFinalizeLayout {
    fn ty_name(&self) -> &'static str {
        "MemStoreFinalizeLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct OpSBLayout {
    pub orig_bytes: &'static SplitWordLayout,
    pub new_bytes: &'static SplitWordLayout,
}
impl risc0_zkp::layout::Component for OpSBLayout {
    fn ty_name(&self) -> &'static str {
        "OpSBLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("orig_bytes", self.orig_bytes)?;
        v.visit_component("new_bytes", self.new_bytes)?;
        Ok(())
    }
}
pub struct Mem1Arm1Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm2Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm3Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm4Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm5Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm6Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1Arm7Layout {
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Mem1Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        Ok(())
    }
}
pub struct Mem1OutputLayout {
    pub arm0: &'static OpSBLayout,
    pub arm1: &'static Mem1Arm1Layout,
    pub arm2: &'static Mem1Arm2Layout,
    pub arm3: &'static Mem1Arm3Layout,
    pub arm4: &'static Mem1Arm4Layout,
    pub arm5: &'static Mem1Arm5Layout,
    pub arm6: &'static Mem1Arm6Layout,
    pub arm7: &'static Mem1Arm7Layout,
}
impl risc0_zkp::layout::Component for Mem1OutputLayout {
    fn ty_name(&self) -> &'static str {
        "Mem1OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub type ArgU8Layout4LayoutArray = [&'static ArgU8Layout; 4];
pub struct _Arguments_Mem1OutputLayout {
    pub arg_u8: &'static ArgU8Layout4LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Mem1OutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Mem1OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct Mem1Layout {
    pub input: &'static MemStoreInputLayout,
    pub _0: &'static MemStoreFinalizeLayout,
    pub output: &'static Mem1OutputLayout,
    pub _arguments_output: &'static _Arguments_Mem1OutputLayout,
    pub pc_add: &'static NormalizeU32Layout,
}
impl risc0_zkp::layout::Component for Mem1Layout {
    fn ty_name(&self) -> &'static str {
        "Mem1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("input", self.input)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("output", self.output)?;
        v.visit_component("_arguments_output", self._arguments_output)?;
        v.visit_component("pc_add", self.pc_add)?;
        Ok(())
    }
}
pub struct MemoryPageInLayout {
    pub io: &'static MemoryIOLayout,
}
impl risc0_zkp::layout::Component for MemoryPageInLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryPageInLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("io", self.io)?;
        Ok(())
    }
}
pub struct ControlLoadRoot__0_SuperLayout {
    pub mem: &'static MemoryPageInLayout,
}
impl risc0_zkp::layout::Component for ControlLoadRoot__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlLoadRoot__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("mem", self.mem)?;
        Ok(())
    }
}
pub type ControlLoadRoot__0_SuperLayout8LayoutArray = [&'static ControlLoadRoot__0_SuperLayout; 8];
pub struct ControlLoadRootLayout {
    pub _0: &'static ControlLoadRoot__0_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for ControlLoadRootLayout {
    fn ty_name(&self) -> &'static str {
        "ControlLoadRootLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Control0Arm0Layout {
    pub _super: &'static ControlLoadRootLayout,
    pub _extra24: &'static ArgU8Layout,
    pub _extra25: &'static ArgU8Layout,
    pub _extra26: &'static ArgU8Layout,
    pub _extra27: &'static ArgU8Layout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra32: &'static ArgU8Layout,
    pub _extra33: &'static ArgU8Layout,
    pub _extra34: &'static ArgU8Layout,
    pub _extra35: &'static ArgU8Layout,
    pub _extra36: &'static ArgU8Layout,
    pub _extra37: &'static ArgU8Layout,
    pub _extra38: &'static ArgU8Layout,
    pub _extra39: &'static ArgU8Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
    pub _extra16: &'static ArgU16Layout,
    pub _extra17: &'static ArgU16Layout,
    pub _extra18: &'static ArgU16Layout,
    pub _extra19: &'static ArgU16Layout,
    pub _extra20: &'static ArgU16Layout,
    pub _extra0: &'static CycleArgLayout,
    pub _extra1: &'static CycleArgLayout,
    pub _extra2: &'static CycleArgLayout,
    pub _extra3: &'static CycleArgLayout,
    pub _extra21: &'static ArgU16Layout,
    pub _extra22: &'static ArgU16Layout,
    pub _extra23: &'static ArgU16Layout,
    pub _extra4: &'static CycleArgLayout,
    pub _extra5: &'static CycleArgLayout,
    pub _extra6: &'static CycleArgLayout,
    pub _extra7: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Control0Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        Ok(())
    }
}
pub struct ControlResumeArm0_SuperLayout {
    pub pc: &'static MemoryReadLayout,
    pub mode: &'static MemoryReadLayout,
}
impl risc0_zkp::layout::Component for ControlResumeArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlResumeArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("pc", self.pc)?;
        v.visit_component("mode", self.mode)?;
        Ok(())
    }
}
pub struct ControlResumeArm0Layout {
    pub _super: &'static ControlResumeArm0_SuperLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra12: &'static CycleArgLayout,
    pub _extra13: &'static CycleArgLayout,
    pub _extra14: &'static CycleArgLayout,
    pub _extra15: &'static CycleArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ControlResumeArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ControlResumeArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        Ok(())
    }
}
pub struct ControlResumeArm1_Super__0_SuperLayout {
    pub _0: &'static MemoryWriteLayout,
}
impl risc0_zkp::layout::Component for ControlResumeArm1_Super__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlResumeArm1_Super__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub type ControlResumeArm1_Super__0_SuperLayout8LayoutArray =
    [&'static ControlResumeArm1_Super__0_SuperLayout; 8];
pub struct ControlResumeArm1_SuperLayout {
    pub _0: &'static ControlResumeArm1_Super__0_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for ControlResumeArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlResumeArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ControlResume_SuperLayout {
    pub arm0: &'static ControlResumeArm0Layout,
    pub arm1: &'static ControlResumeArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for ControlResume_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlResume_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type MemoryArgLayout16LayoutArray = [&'static MemoryArgLayout; 16];
pub type CycleArgLayout8LayoutArray = [&'static CycleArgLayout; 8];
pub struct _Arguments_ControlResume_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_ControlResume_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_ControlResume_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        Ok(())
    }
}
pub struct ControlResumeLayout {
    pub _super: &'static ControlResume_SuperLayout,
    pub pc_zero: &'static IsZeroLayout,
    pub _arguments__super: &'static _Arguments_ControlResume_SuperLayout,
}
impl risc0_zkp::layout::Component for ControlResumeLayout {
    fn ty_name(&self) -> &'static str {
        "ControlResumeLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("pc_zero", self.pc_zero)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub struct Control0Arm1Layout {
    pub _super: &'static ControlResumeLayout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra18: &'static ArgU8Layout,
    pub _extra19: &'static ArgU8Layout,
    pub _extra20: &'static ArgU8Layout,
    pub _extra21: &'static ArgU8Layout,
    pub _extra22: &'static ArgU8Layout,
    pub _extra23: &'static ArgU8Layout,
    pub _extra24: &'static ArgU8Layout,
    pub _extra25: &'static ArgU8Layout,
    pub _extra26: &'static ArgU8Layout,
    pub _extra27: &'static ArgU8Layout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Control0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct ControlUserECALLLayout {
    pub _1: &'static MemoryWriteLayout,
    pub load_inst: &'static MemoryReadLayout,
    pub _0: &'static U16RegLayout,
    pub safe_mode: &'static NondetRegLayout,
    pub dispatch_idx: &'static MemoryReadLayout,
    pub new_pc_addr: &'static MemoryReadLayout,
    pub pc_addr: &'static AddrDecomposeBitsLayout,
}
impl risc0_zkp::layout::Component for ControlUserECALLLayout {
    fn ty_name(&self) -> &'static str {
        "ControlUserECALLLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_1", self._1)?;
        v.visit_component("load_inst", self.load_inst)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("safe_mode", self.safe_mode)?;
        v.visit_component("dispatch_idx", self.dispatch_idx)?;
        v.visit_component("new_pc_addr", self.new_pc_addr)?;
        v.visit_component("pc_addr", self.pc_addr)?;
        Ok(())
    }
}
pub struct Control0Arm2Layout {
    pub _super: &'static ControlUserECALLLayout,
    pub _extra25: &'static ArgU8Layout,
    pub _extra26: &'static ArgU8Layout,
    pub _extra27: &'static ArgU8Layout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra32: &'static ArgU8Layout,
    pub _extra33: &'static ArgU8Layout,
    pub _extra34: &'static ArgU8Layout,
    pub _extra35: &'static ArgU8Layout,
    pub _extra36: &'static ArgU8Layout,
    pub _extra37: &'static ArgU8Layout,
    pub _extra38: &'static ArgU8Layout,
    pub _extra39: &'static ArgU8Layout,
    pub _extra40: &'static ArgU8Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
    pub _extra16: &'static ArgU16Layout,
    pub _extra17: &'static ArgU16Layout,
    pub _extra18: &'static ArgU16Layout,
    pub _extra19: &'static ArgU16Layout,
    pub _extra20: &'static ArgU16Layout,
    pub _extra21: &'static ArgU16Layout,
    pub _extra22: &'static ArgU16Layout,
    pub _extra23: &'static ArgU16Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
}
impl risc0_zkp::layout::Component for Control0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra40", self._extra40)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        Ok(())
    }
}
pub struct ControlMRETLayout {
    pub pc_add: &'static NormalizeU32Layout,
    pub safe_mode: &'static NondetRegLayout,
    pub load_inst: &'static MemoryReadLayout,
    pub pc: &'static MemoryReadLayout,
    pub pc_addr: &'static AddrDecomposeBitsLayout,
}
impl risc0_zkp::layout::Component for ControlMRETLayout {
    fn ty_name(&self) -> &'static str {
        "ControlMRETLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("pc_add", self.pc_add)?;
        v.visit_component("safe_mode", self.safe_mode)?;
        v.visit_component("load_inst", self.load_inst)?;
        v.visit_component("pc", self.pc)?;
        v.visit_component("pc_addr", self.pc_addr)?;
        Ok(())
    }
}
pub struct Control0Arm3Layout {
    pub _super: &'static ControlMRETLayout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra32: &'static ArgU8Layout,
    pub _extra33: &'static ArgU8Layout,
    pub _extra34: &'static ArgU8Layout,
    pub _extra35: &'static ArgU8Layout,
    pub _extra36: &'static ArgU8Layout,
    pub _extra37: &'static ArgU8Layout,
    pub _extra38: &'static ArgU8Layout,
    pub _extra39: &'static ArgU8Layout,
    pub _extra40: &'static ArgU8Layout,
    pub _extra41: &'static ArgU8Layout,
    pub _extra42: &'static ArgU8Layout,
    pub _extra43: &'static ArgU8Layout,
    pub _extra44: &'static ArgU8Layout,
    pub _extra45: &'static ArgU8Layout,
    pub _extra18: &'static ArgU16Layout,
    pub _extra19: &'static ArgU16Layout,
    pub _extra20: &'static ArgU16Layout,
    pub _extra21: &'static ArgU16Layout,
    pub _extra22: &'static ArgU16Layout,
    pub _extra23: &'static ArgU16Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra28: &'static ArgU16Layout,
    pub _extra29: &'static ArgU16Layout,
    pub _extra12: &'static CycleArgLayout,
    pub _extra13: &'static CycleArgLayout,
    pub _extra14: &'static CycleArgLayout,
    pub _extra15: &'static CycleArgLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
}
impl risc0_zkp::layout::Component for Control0Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra40", self._extra40)?;
        v.visit_component("_extra41", self._extra41)?;
        v.visit_component("_extra42", self._extra42)?;
        v.visit_component("_extra43", self._extra43)?;
        v.visit_component("_extra44", self._extra44)?;
        v.visit_component("_extra45", self._extra45)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub type MemoryReadLayout8LayoutArray = [&'static MemoryReadLayout; 8];
pub struct ControlSuspendArm0_SuperLayout {
    pub _0: &'static MemoryReadLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for ControlSuspendArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlSuspendArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ControlSuspendArm1_SuperLayout {
    pub _0: &'static MemoryWriteLayout,
    pub _1: &'static MemoryWriteLayout,
    pub state: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ControlSuspendArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlSuspendArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        v.visit_component("state", self.state)?;
        Ok(())
    }
}
pub struct ControlSuspendArm1Layout {
    pub _super: &'static ControlSuspendArm1_SuperLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra12: &'static CycleArgLayout,
    pub _extra13: &'static CycleArgLayout,
    pub _extra14: &'static CycleArgLayout,
    pub _extra15: &'static CycleArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ControlSuspendArm1Layout {
    fn ty_name(&self) -> &'static str {
        "ControlSuspendArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        Ok(())
    }
}
pub struct ControlSuspend_SuperLayout {
    pub arm0: &'static ControlSuspendArm0_SuperLayout,
    pub arm1: &'static ControlSuspendArm1Layout,
}
impl risc0_zkp::layout::Component for ControlSuspend_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlSuspend_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _Arguments_ControlSuspend_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_ControlSuspend_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_ControlSuspend_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        Ok(())
    }
}
pub struct ControlSuspendLayout {
    pub _super: &'static ControlSuspend_SuperLayout,
    pub pc_zero: &'static IsZeroLayout,
    pub _arguments__super: &'static _Arguments_ControlSuspend_SuperLayout,
}
impl risc0_zkp::layout::Component for ControlSuspendLayout {
    fn ty_name(&self) -> &'static str {
        "ControlSuspendLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("pc_zero", self.pc_zero)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub struct Control0Arm4Layout {
    pub _super: &'static ControlSuspendLayout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra18: &'static ArgU8Layout,
    pub _extra19: &'static ArgU8Layout,
    pub _extra20: &'static ArgU8Layout,
    pub _extra21: &'static ArgU8Layout,
    pub _extra22: &'static ArgU8Layout,
    pub _extra23: &'static ArgU8Layout,
    pub _extra24: &'static ArgU8Layout,
    pub _extra25: &'static ArgU8Layout,
    pub _extra26: &'static ArgU8Layout,
    pub _extra27: &'static ArgU8Layout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Control0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct MemoryPageOutLayout {
    pub io: &'static MemoryIOLayout,
    pub _0: &'static IsForwardLayout,
}
impl risc0_zkp::layout::Component for MemoryPageOutLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryPageOutLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("io", self.io)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub type MemoryPageOutLayout8LayoutArray = [&'static MemoryPageOutLayout; 8];
pub struct ControlStoreRootLayout {
    pub _0: &'static MemoryPageOutLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for ControlStoreRootLayout {
    fn ty_name(&self) -> &'static str {
        "ControlStoreRootLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Control0Arm5Layout {
    pub _super: &'static ControlStoreRootLayout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra18: &'static ArgU8Layout,
    pub _extra19: &'static ArgU8Layout,
    pub _extra20: &'static ArgU8Layout,
    pub _extra21: &'static ArgU8Layout,
    pub _extra22: &'static ArgU8Layout,
    pub _extra23: &'static ArgU8Layout,
    pub _extra24: &'static ArgU8Layout,
    pub _extra25: &'static ArgU8Layout,
    pub _extra26: &'static ArgU8Layout,
    pub _extra27: &'static ArgU8Layout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra30: &'static ArgU8Layout,
    pub _extra31: &'static ArgU8Layout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Control0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct ControlTableArm0_Super__0_SuperLayout {
    pub arg: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for ControlTableArm0_Super__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm0_Super__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub type ControlTableArm0_Super__0_SuperLayout16LayoutArray =
    [&'static ControlTableArm0_Super__0_SuperLayout; 16];
pub struct ControlTableArm0_SuperLayout {
    pub done: &'static IsZeroLayout,
    pub _0: &'static ControlTableArm0_Super__0_SuperLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ControlTableArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("done", self.done)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ControlTableArm0Layout {
    pub _super: &'static ControlTableArm0_SuperLayout,
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
    pub _extra2: &'static ArgU8Layout,
    pub _extra3: &'static ArgU8Layout,
    pub _extra4: &'static ArgU8Layout,
    pub _extra5: &'static ArgU8Layout,
    pub _extra6: &'static ArgU8Layout,
    pub _extra7: &'static ArgU8Layout,
    pub _extra8: &'static ArgU8Layout,
    pub _extra9: &'static ArgU8Layout,
    pub _extra10: &'static ArgU8Layout,
    pub _extra11: &'static ArgU8Layout,
    pub _extra12: &'static ArgU8Layout,
    pub _extra13: &'static ArgU8Layout,
    pub _extra14: &'static ArgU8Layout,
    pub _extra15: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for ControlTableArm0Layout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct ControlTableArm1_Super__0_SuperLayout {
    pub arg: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for ControlTableArm1_Super__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm1_Super__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub type ControlTableArm1_Super__0_SuperLayout16LayoutArray =
    [&'static ControlTableArm1_Super__0_SuperLayout; 16];
pub struct ControlTableArm1_SuperLayout {
    pub done: &'static IsZeroLayout,
    pub _0: &'static ControlTableArm1_Super__0_SuperLayout16LayoutArray,
}
impl risc0_zkp::layout::Component for ControlTableArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("done", self.done)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ControlTableArm1Layout {
    pub _super: &'static ControlTableArm1_SuperLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for ControlTableArm1Layout {
    fn ty_name(&self) -> &'static str {
        "ControlTableArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct ControlTable_SuperLayout {
    pub arm0: &'static ControlTableArm0Layout,
    pub arm1: &'static ControlTableArm1Layout,
}
impl risc0_zkp::layout::Component for ControlTable_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTable_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type ArgU16Layout16LayoutArray = [&'static ArgU16Layout; 16];
pub type ArgU8Layout16LayoutArray = [&'static ArgU8Layout; 16];
pub struct _Arguments_ControlTable_SuperLayout {
    pub arg_u16: &'static ArgU16Layout16LayoutArray,
    pub arg_u8: &'static ArgU8Layout16LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_ControlTable_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_ControlTable_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct ControlTableLayout {
    pub _super: &'static ControlTable_SuperLayout,
    pub _arguments__super: &'static _Arguments_ControlTable_SuperLayout,
    pub entry: &'static NondetRegLayout,
    pub mode: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ControlTableLayout {
    fn ty_name(&self) -> &'static str {
        "ControlTableLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        v.visit_component("entry", self.entry)?;
        v.visit_component("mode", self.mode)?;
        Ok(())
    }
}
pub struct Control0Arm6Layout {
    pub _super: &'static ControlTableLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra18: &'static CycleArgLayout,
    pub _extra19: &'static CycleArgLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra20: &'static CycleArgLayout,
    pub _extra21: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra22: &'static CycleArgLayout,
    pub _extra23: &'static CycleArgLayout,
    pub _extra12: &'static MemoryArgLayout,
    pub _extra13: &'static MemoryArgLayout,
    pub _extra14: &'static MemoryArgLayout,
    pub _extra15: &'static MemoryArgLayout,
}
impl risc0_zkp::layout::Component for Control0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct Control0Arm7Layout {
    pub _extra40: &'static ArgU8Layout,
    pub _extra41: &'static ArgU8Layout,
    pub _extra42: &'static ArgU8Layout,
    pub _extra43: &'static ArgU8Layout,
    pub _extra44: &'static ArgU8Layout,
    pub _extra45: &'static ArgU8Layout,
    pub _extra46: &'static ArgU8Layout,
    pub _extra47: &'static ArgU8Layout,
    pub _extra48: &'static ArgU8Layout,
    pub _extra49: &'static ArgU8Layout,
    pub _extra50: &'static ArgU8Layout,
    pub _extra51: &'static ArgU8Layout,
    pub _extra52: &'static ArgU8Layout,
    pub _extra53: &'static ArgU8Layout,
    pub _extra54: &'static ArgU8Layout,
    pub _extra55: &'static ArgU8Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra28: &'static ArgU16Layout,
    pub _extra29: &'static ArgU16Layout,
    pub _extra30: &'static ArgU16Layout,
    pub _extra31: &'static ArgU16Layout,
    pub _extra32: &'static ArgU16Layout,
    pub _extra33: &'static ArgU16Layout,
    pub _extra34: &'static ArgU16Layout,
    pub _extra35: &'static ArgU16Layout,
    pub _extra36: &'static ArgU16Layout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra18: &'static CycleArgLayout,
    pub _extra19: &'static CycleArgLayout,
    pub _extra37: &'static ArgU16Layout,
    pub _extra38: &'static ArgU16Layout,
    pub _extra39: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra20: &'static CycleArgLayout,
    pub _extra21: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra22: &'static CycleArgLayout,
    pub _extra23: &'static CycleArgLayout,
    pub _extra12: &'static MemoryArgLayout,
    pub _extra13: &'static MemoryArgLayout,
    pub _extra14: &'static MemoryArgLayout,
    pub _extra15: &'static MemoryArgLayout,
}
impl risc0_zkp::layout::Component for Control0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra40", self._extra40)?;
        v.visit_component("_extra41", self._extra41)?;
        v.visit_component("_extra42", self._extra42)?;
        v.visit_component("_extra43", self._extra43)?;
        v.visit_component("_extra44", self._extra44)?;
        v.visit_component("_extra45", self._extra45)?;
        v.visit_component("_extra46", self._extra46)?;
        v.visit_component("_extra47", self._extra47)?;
        v.visit_component("_extra48", self._extra48)?;
        v.visit_component("_extra49", self._extra49)?;
        v.visit_component("_extra50", self._extra50)?;
        v.visit_component("_extra51", self._extra51)?;
        v.visit_component("_extra52", self._extra52)?;
        v.visit_component("_extra53", self._extra53)?;
        v.visit_component("_extra54", self._extra54)?;
        v.visit_component("_extra55", self._extra55)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct Control0_SuperLayout {
    pub arm0: &'static Control0Arm0Layout,
    pub arm1: &'static Control0Arm1Layout,
    pub arm2: &'static Control0Arm2Layout,
    pub arm3: &'static Control0Arm3Layout,
    pub arm4: &'static Control0Arm4Layout,
    pub arm5: &'static Control0Arm5Layout,
    pub arm6: &'static Control0Arm6Layout,
    pub arm7: &'static Control0Arm7Layout,
}
impl risc0_zkp::layout::Component for Control0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "Control0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub struct _Arguments_Control0_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
    pub arg_u16: &'static ArgU16Layout16LayoutArray,
    pub arg_u8: &'static ArgU8Layout16LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Control0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Control0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct Control0Layout {
    pub _super: &'static Control0_SuperLayout,
    pub arg: &'static CycleArgLayout,
    pub _arguments__super: &'static _Arguments_Control0_SuperLayout,
}
impl risc0_zkp::layout::Component for Control0Layout {
    fn ty_name(&self) -> &'static str {
        "Control0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arg", self.arg)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub type NondetRegLayout4LayoutArray = [&'static NondetRegLayout; 4];
pub struct OneHot_4_Layout {
    pub _super: &'static NondetRegLayout4LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_4_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_4_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct MachineECallLayout {
    pub load_inst: &'static MemoryReadLayout,
    pub dispatch_idx: &'static MemoryReadLayout,
    pub dispatch: &'static OneHot_4_Layout,
}
impl risc0_zkp::layout::Component for MachineECallLayout {
    fn ty_name(&self) -> &'static str {
        "MachineECallLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("load_inst", self.load_inst)?;
        v.visit_component("dispatch_idx", self.dispatch_idx)?;
        v.visit_component("dispatch", self.dispatch)?;
        Ok(())
    }
}
pub struct ECall0Arm0Layout {
    pub _super: &'static MachineECallLayout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static CycleArgLayout,
    pub _extra5: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ECall0Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        Ok(())
    }
}
pub struct ECallTerminateLayout {
    pub a0: &'static MemoryReadLayout,
    pub a1: &'static MemoryReadLayout,
}
impl risc0_zkp::layout::Component for ECallTerminateLayout {
    fn ty_name(&self) -> &'static str {
        "ECallTerminateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("a0", self.a0)?;
        v.visit_component("a1", self.a1)?;
        Ok(())
    }
}
pub struct ECall0Arm1Layout {
    pub _super: &'static ECallTerminateLayout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static CycleArgLayout,
    pub _extra5: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ECall0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        Ok(())
    }
}
pub struct DecomposeLow2Layout {
    pub low2_hot: &'static OneHot_4_Layout,
    pub high_zero: &'static IsZeroLayout,
    pub is_zero: &'static NondetRegLayout,
    pub high: &'static NondetRegLayout,
    pub low2: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for DecomposeLow2Layout {
    fn ty_name(&self) -> &'static str {
        "DecomposeLow2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low2_hot", self.low2_hot)?;
        v.visit_component("high_zero", self.high_zero)?;
        v.visit_component("is_zero", self.is_zero)?;
        v.visit_component("high", self.high)?;
        v.visit_component("low2", self.low2)?;
        Ok(())
    }
}
pub struct ECallHostReadSetupLayout {
    pub _0: &'static MemoryWriteLayout,
    pub fd: &'static MemoryReadLayout,
    pub ptr: &'static MemoryReadLayout,
    pub len: &'static MemoryReadLayout,
    pub diff: &'static U16RegLayout,
    pub new_len: &'static NondetU16RegLayout,
    pub ptr_decomp: &'static DecomposeLow2Layout,
    pub len_decomp: &'static DecomposeLow2Layout,
    pub len123: &'static NondetRegLayout,
    pub uneven: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ECallHostReadSetupLayout {
    fn ty_name(&self) -> &'static str {
        "ECallHostReadSetupLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("fd", self.fd)?;
        v.visit_component("ptr", self.ptr)?;
        v.visit_component("len", self.len)?;
        v.visit_component("diff", self.diff)?;
        v.visit_component("new_len", self.new_len)?;
        v.visit_component("ptr_decomp", self.ptr_decomp)?;
        v.visit_component("len_decomp", self.len_decomp)?;
        v.visit_component("len123", self.len123)?;
        v.visit_component("uneven", self.uneven)?;
        Ok(())
    }
}
pub struct ECallHostWriteLayout {
    pub _0: &'static MemoryWriteLayout,
    pub fd: &'static MemoryReadLayout,
    pub ptr: &'static MemoryReadLayout,
    pub len: &'static MemoryReadLayout,
    pub diff: &'static U16RegLayout,
    pub new_len: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for ECallHostWriteLayout {
    fn ty_name(&self) -> &'static str {
        "ECallHostWriteLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("fd", self.fd)?;
        v.visit_component("ptr", self.ptr)?;
        v.visit_component("len", self.len)?;
        v.visit_component("diff", self.diff)?;
        v.visit_component("new_len", self.new_len)?;
        Ok(())
    }
}
pub struct ECall0Arm4Layout {
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ECall0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub struct MemoryWriteUnconstrainedLayout {
    pub io: &'static MemoryIOLayout,
    pub _0: &'static IsForwardLayout,
}
impl risc0_zkp::layout::Component for MemoryWriteUnconstrainedLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryWriteUnconstrainedLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("io", self.io)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ECallHostReadWords__0_SuperLayout {
    pub addr: &'static NondetRegLayout,
    pub _0: &'static MemoryWriteUnconstrainedLayout,
}
impl risc0_zkp::layout::Component for ECallHostReadWords__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "ECallHostReadWords__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("addr", self.addr)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub type ECallHostReadWords__0_SuperLayout4LayoutArray =
    [&'static ECallHostReadWords__0_SuperLayout; 4];
pub struct ECallHostReadWordsLayout {
    pub len_decomp: &'static DecomposeLow2Layout,
    pub len_zero: &'static IsZeroLayout,
    pub words_decomp: &'static DecomposeLow2Layout,
    pub _0: &'static ECallHostReadWords__0_SuperLayout4LayoutArray,
}
impl risc0_zkp::layout::Component for ECallHostReadWordsLayout {
    fn ty_name(&self) -> &'static str {
        "ECallHostReadWordsLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("len_decomp", self.len_decomp)?;
        v.visit_component("len_zero", self.len_zero)?;
        v.visit_component("words_decomp", self.words_decomp)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct ECall0Arm5Layout {
    pub _super: &'static ECallHostReadWordsLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for ECall0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct ECall0Arm6Layout {
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ECall0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub struct ECall0Arm7Layout {
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for ECall0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub struct ECall0OutputLayout {
    pub arm0: &'static ECall0Arm0Layout,
    pub arm1: &'static ECall0Arm1Layout,
    pub arm2: &'static ECallHostReadSetupLayout,
    pub arm3: &'static ECallHostWriteLayout,
    pub arm4: &'static ECall0Arm4Layout,
    pub arm5: &'static ECall0Arm5Layout,
    pub arm6: &'static ECall0Arm6Layout,
    pub arm7: &'static ECall0Arm7Layout,
}
impl risc0_zkp::layout::Component for ECall0OutputLayout {
    fn ty_name(&self) -> &'static str {
        "ECall0OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        v.visit_component("arm3", self.arm3)?;
        v.visit_component("arm4", self.arm4)?;
        v.visit_component("arm5", self.arm5)?;
        v.visit_component("arm6", self.arm6)?;
        v.visit_component("arm7", self.arm7)?;
        Ok(())
    }
}
pub type MemoryArgLayout8LayoutArray = [&'static MemoryArgLayout; 8];
pub type CycleArgLayout4LayoutArray = [&'static CycleArgLayout; 4];
pub type ArgU16Layout2LayoutArray = [&'static ArgU16Layout; 2];
pub struct _Arguments_ECall0OutputLayout {
    pub memory_arg: &'static MemoryArgLayout8LayoutArray,
    pub cycle_arg: &'static CycleArgLayout4LayoutArray,
    pub arg_u16: &'static ArgU16Layout2LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_ECall0OutputLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_ECall0OutputLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct ECall0Layout {
    pub output: &'static ECall0OutputLayout,
    pub _arguments_output: &'static _Arguments_ECall0OutputLayout,
    pub pc_addr: &'static AddrDecomposeBitsLayout,
    pub is_decode: &'static IsZeroLayout,
    pub s0: &'static NondetRegLayout,
    pub add_pc: &'static NormalizeU32Layout,
    pub arg: &'static CycleArgLayout,
    pub is_p2_entry: &'static IsZeroLayout,
    pub s1: &'static NondetRegLayout,
    pub s2: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for ECall0Layout {
    fn ty_name(&self) -> &'static str {
        "ECall0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("output", self.output)?;
        v.visit_component("_arguments_output", self._arguments_output)?;
        v.visit_component("pc_addr", self.pc_addr)?;
        v.visit_component("is_decode", self.is_decode)?;
        v.visit_component("s0", self.s0)?;
        v.visit_component("add_pc", self.add_pc)?;
        v.visit_component("arg", self.arg)?;
        v.visit_component("is_p2_entry", self.is_p2_entry)?;
        v.visit_component("s1", self.s1)?;
        v.visit_component("s2", self.s2)?;
        Ok(())
    }
}
pub type NondetRegLayout24LayoutArray = [&'static NondetRegLayout; 24];
pub struct NondetExtRegLayout {
    pub _super: &'static Reg,
}
impl risc0_zkp::layout::Component for NondetExtRegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetExtRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct PoseidonStateLayout {
    pub has_state: &'static NondetRegLayout,
    pub inner: &'static NondetRegLayout24LayoutArray,
    pub state_addr: &'static NondetRegLayout,
    pub buf_out_addr: &'static NondetRegLayout,
    pub is_elem: &'static NondetRegLayout,
    pub check_out: &'static NondetRegLayout,
    pub load_tx_type: &'static NondetRegLayout,
    pub next_state: &'static NondetRegLayout,
    pub sub_state: &'static NondetRegLayout,
    pub buf_in_addr: &'static NondetRegLayout,
    pub count: &'static NondetRegLayout,
    pub mode: &'static NondetRegLayout,
    pub zcheck: &'static NondetExtRegLayout,
}
impl risc0_zkp::layout::Component for PoseidonStateLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonStateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("has_state", self.has_state)?;
        v.visit_component("inner", self.inner)?;
        v.visit_component("state_addr", self.state_addr)?;
        v.visit_component("buf_out_addr", self.buf_out_addr)?;
        v.visit_component("is_elem", self.is_elem)?;
        v.visit_component("check_out", self.check_out)?;
        v.visit_component("load_tx_type", self.load_tx_type)?;
        v.visit_component("next_state", self.next_state)?;
        v.visit_component("sub_state", self.sub_state)?;
        v.visit_component("buf_in_addr", self.buf_in_addr)?;
        v.visit_component("count", self.count)?;
        v.visit_component("mode", self.mode)?;
        v.visit_component("zcheck", self.zcheck)?;
        Ok(())
    }
}
pub struct PoseidonEntryArm0Layout {
    pub _super: &'static PoseidonStateLayout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for PoseidonEntryArm0Layout {
    fn ty_name(&self) -> &'static str {
        "PoseidonEntryArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub struct ReadAddrLayout {
    pub addr32: &'static MemoryReadLayout,
}
impl risc0_zkp::layout::Component for ReadAddrLayout {
    fn ty_name(&self) -> &'static str {
        "ReadAddrLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("addr32", self.addr32)?;
        Ok(())
    }
}
pub struct PoseidonEcallLayout {
    pub _super: &'static PoseidonStateLayout,
    pub state_addr: &'static ReadAddrLayout,
    pub buf_in_addr: &'static ReadAddrLayout,
    pub buf_out_addr: &'static ReadAddrLayout,
    pub bits_and_count: &'static MemoryReadLayout,
    pub _0: &'static IsZeroLayout,
    pub count_zero: &'static IsZeroLayout,
    pub is_elem: &'static NondetRegLayout,
    pub check_out: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for PoseidonEcallLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonEcallLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("state_addr", self.state_addr)?;
        v.visit_component("buf_in_addr", self.buf_in_addr)?;
        v.visit_component("buf_out_addr", self.buf_out_addr)?;
        v.visit_component("bits_and_count", self.bits_and_count)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("count_zero", self.count_zero)?;
        v.visit_component("is_elem", self.is_elem)?;
        v.visit_component("check_out", self.check_out)?;
        Ok(())
    }
}
pub struct PoseidonEntry_SuperLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static PoseidonEntryArm0Layout,
    pub arm1: &'static PoseidonEcallLayout,
}
impl risc0_zkp::layout::Component for PoseidonEntry_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonEntry_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _Arguments_PoseidonEntry_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout8LayoutArray,
    pub cycle_arg: &'static CycleArgLayout4LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_PoseidonEntry_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_PoseidonEntry_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        Ok(())
    }
}
pub struct PoseidonEntryLayout {
    pub _super: &'static PoseidonEntry_SuperLayout,
    pub _arguments__super: &'static _Arguments_PoseidonEntry_SuperLayout,
    pub pc_zero: &'static IsZeroLayout,
}
impl risc0_zkp::layout::Component for PoseidonEntryLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonEntryLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        v.visit_component("pc_zero", self.pc_zero)?;
        Ok(())
    }
}
pub struct Poseidon0Arm0Layout {
    pub _super: &'static PoseidonEntryLayout,
    pub _extra28: &'static ArgU8Layout,
    pub _extra29: &'static ArgU8Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
    pub _extra16: &'static ArgU16Layout,
    pub _extra17: &'static ArgU16Layout,
    pub _extra18: &'static ArgU16Layout,
    pub _extra19: &'static ArgU16Layout,
    pub _extra20: &'static ArgU16Layout,
    pub _extra21: &'static ArgU16Layout,
    pub _extra22: &'static ArgU16Layout,
    pub _extra23: &'static ArgU16Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra8: &'static CycleArgLayout,
    pub _extra9: &'static CycleArgLayout,
    pub _extra10: &'static CycleArgLayout,
    pub _extra11: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm0Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        Ok(())
    }
}
pub struct ReadElemLayout {
    pub elem32: &'static MemoryReadLayout,
}
impl risc0_zkp::layout::Component for ReadElemLayout {
    fn ty_name(&self) -> &'static str {
        "ReadElemLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("elem32", self.elem32)?;
        Ok(())
    }
}
pub type ReadElemLayout8LayoutArray = [&'static ReadElemLayout; 8];
pub struct PoseidonLoadStateLayout {
    pub _super: &'static PoseidonStateLayout,
    pub load_list: &'static ReadElemLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonLoadStateLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadStateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("load_list", self.load_list)?;
        Ok(())
    }
}
pub struct Poseidon0Arm1Layout {
    pub _super: &'static PoseidonLoadStateLayout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm1Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub type NondetRegLayout3LayoutArray = [&'static NondetRegLayout; 3];
pub struct OneHot_3_Layout {
    pub _super: &'static NondetRegLayout3LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_3_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_3_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct MemoryGetArm1Layout {
    pub _super: &'static MemoryPageInLayout,
    pub _extra0: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for MemoryGetArm1Layout {
    fn ty_name(&self) -> &'static str {
        "MemoryGetArm1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        Ok(())
    }
}
pub struct MemoryGet_SuperLayout {
    pub arm0: &'static MemoryReadLayout,
    pub arm1: &'static MemoryGetArm1Layout,
    pub arm2: &'static MemoryPageOutLayout,
}
impl risc0_zkp::layout::Component for MemoryGet_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryGet_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        Ok(())
    }
}
pub type MemoryArgLayout2LayoutArray = [&'static MemoryArgLayout; 2];
pub type CycleArgLayout1LayoutArray = [&'static CycleArgLayout; 1];
pub struct _Arguments_MemoryGet_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout2LayoutArray,
    pub cycle_arg: &'static CycleArgLayout1LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_MemoryGet_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_MemoryGet_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        Ok(())
    }
}
pub struct MemoryGetLayout {
    pub _super: &'static MemoryGet_SuperLayout,
    pub _arguments__super: &'static _Arguments_MemoryGet_SuperLayout,
}
impl risc0_zkp::layout::Component for MemoryGetLayout {
    fn ty_name(&self) -> &'static str {
        "MemoryGetLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub type MemoryGetLayout8LayoutArray = [&'static MemoryGetLayout; 8];
pub struct PoseidonLoadInShortLayout {
    pub _super: &'static PoseidonStateLayout,
    pub tx_type: &'static OneHot_3_Layout,
    pub load_list: &'static MemoryGetLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonLoadInShortLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadInShortLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("tx_type", self.tx_type)?;
        v.visit_component("load_list", self.load_list)?;
        Ok(())
    }
}
pub struct PoseidonLoadInLowLayout {
    pub _super: &'static PoseidonStateLayout,
    pub tx_type: &'static OneHot_3_Layout,
    pub load_list: &'static MemoryGetLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonLoadInLowLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadInLowLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("tx_type", self.tx_type)?;
        v.visit_component("load_list", self.load_list)?;
        Ok(())
    }
}
pub struct PoseidonLoadInHighLayout {
    pub _super: &'static PoseidonStateLayout,
    pub tx_type: &'static OneHot_3_Layout,
    pub load_list: &'static MemoryGetLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonLoadInHighLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadInHighLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("tx_type", self.tx_type)?;
        v.visit_component("load_list", self.load_list)?;
        Ok(())
    }
}
pub struct PoseidonLoadIn_SuperLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static PoseidonLoadInShortLayout,
    pub arm1: &'static PoseidonLoadInLowLayout,
    pub arm2: &'static PoseidonLoadInHighLayout,
}
impl risc0_zkp::layout::Component for PoseidonLoadIn_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadIn_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        v.visit_component("arm2", self.arm2)?;
        Ok(())
    }
}
pub struct _Arguments_PoseidonLoadIn_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_PoseidonLoadIn_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_PoseidonLoadIn_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        Ok(())
    }
}
pub struct PoseidonLoadInLayout {
    pub _super: &'static PoseidonLoadIn_SuperLayout,
    pub _0: &'static OneHot_3_Layout,
    pub _arguments__super: &'static _Arguments_PoseidonLoadIn_SuperLayout,
}
impl risc0_zkp::layout::Component for PoseidonLoadInLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonLoadInLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub struct Poseidon0Arm2Layout {
    pub _super: &'static PoseidonLoadInLayout,
    pub _extra16: &'static ArgU8Layout,
    pub _extra17: &'static ArgU8Layout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm2Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm2Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct Poseidon0Arm3Layout {
    pub _super: &'static PoseidonStateLayout,
    pub _extra40: &'static ArgU8Layout,
    pub _extra41: &'static ArgU8Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra28: &'static ArgU16Layout,
    pub _extra29: &'static ArgU16Layout,
    pub _extra30: &'static ArgU16Layout,
    pub _extra31: &'static ArgU16Layout,
    pub _extra32: &'static ArgU16Layout,
    pub _extra33: &'static ArgU16Layout,
    pub _extra34: &'static ArgU16Layout,
    pub _extra35: &'static ArgU16Layout,
    pub _extra36: &'static ArgU16Layout,
    pub _extra37: &'static ArgU16Layout,
    pub _extra38: &'static ArgU16Layout,
    pub _extra39: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra18: &'static CycleArgLayout,
    pub _extra19: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra12: &'static MemoryArgLayout,
    pub _extra13: &'static MemoryArgLayout,
    pub _extra14: &'static MemoryArgLayout,
    pub _extra15: &'static MemoryArgLayout,
    pub _extra20: &'static CycleArgLayout,
    pub _extra21: &'static CycleArgLayout,
    pub _extra22: &'static CycleArgLayout,
    pub _extra23: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm3Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm3Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra40", self._extra40)?;
        v.visit_component("_extra41", self._extra41)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        Ok(())
    }
}
pub struct Poseidon0Arm4Layout {
    pub _super: &'static PoseidonStateLayout,
    pub _extra40: &'static ArgU8Layout,
    pub _extra41: &'static ArgU8Layout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra28: &'static ArgU16Layout,
    pub _extra29: &'static ArgU16Layout,
    pub _extra30: &'static ArgU16Layout,
    pub _extra31: &'static ArgU16Layout,
    pub _extra32: &'static ArgU16Layout,
    pub _extra33: &'static ArgU16Layout,
    pub _extra34: &'static ArgU16Layout,
    pub _extra35: &'static ArgU16Layout,
    pub _extra36: &'static ArgU16Layout,
    pub _extra37: &'static ArgU16Layout,
    pub _extra38: &'static ArgU16Layout,
    pub _extra39: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra18: &'static CycleArgLayout,
    pub _extra19: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra12: &'static MemoryArgLayout,
    pub _extra13: &'static MemoryArgLayout,
    pub _extra14: &'static MemoryArgLayout,
    pub _extra15: &'static MemoryArgLayout,
    pub _extra20: &'static CycleArgLayout,
    pub _extra21: &'static CycleArgLayout,
    pub _extra22: &'static CycleArgLayout,
    pub _extra23: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm4Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm4Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra40", self._extra40)?;
        v.visit_component("_extra41", self._extra41)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra38", self._extra38)?;
        v.visit_component("_extra39", self._extra39)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        Ok(())
    }
}
pub struct PoseidonCheckOut__0_SuperLayout {
    pub goal: &'static ReadElemLayout,
}
impl risc0_zkp::layout::Component for PoseidonCheckOut__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonCheckOut__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("goal", self.goal)?;
        Ok(())
    }
}
pub type PoseidonCheckOut__0_SuperLayout8LayoutArray =
    [&'static PoseidonCheckOut__0_SuperLayout; 8];
pub struct PoseidonCheckOutLayout {
    pub _super: &'static PoseidonStateLayout,
    pub is_normal: &'static IsZeroLayout,
    pub ext_inv: &'static NondetExtRegLayout,
    pub _0: &'static PoseidonCheckOut__0_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonCheckOutLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonCheckOutLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("is_normal", self.is_normal)?;
        v.visit_component("ext_inv", self.ext_inv)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct PoseidonDoOutArm0Layout {
    pub _super: &'static PoseidonCheckOutLayout,
    pub _extra0: &'static ArgU16Layout,
    pub _extra1: &'static ArgU16Layout,
    pub _extra2: &'static ArgU16Layout,
    pub _extra3: &'static ArgU16Layout,
    pub _extra4: &'static ArgU16Layout,
    pub _extra5: &'static ArgU16Layout,
    pub _extra6: &'static ArgU16Layout,
    pub _extra7: &'static ArgU16Layout,
    pub _extra8: &'static ArgU16Layout,
    pub _extra9: &'static ArgU16Layout,
    pub _extra10: &'static ArgU16Layout,
    pub _extra11: &'static ArgU16Layout,
    pub _extra12: &'static ArgU16Layout,
    pub _extra13: &'static ArgU16Layout,
    pub _extra14: &'static ArgU16Layout,
    pub _extra15: &'static ArgU16Layout,
}
impl risc0_zkp::layout::Component for PoseidonDoOutArm0Layout {
    fn ty_name(&self) -> &'static str {
        "PoseidonDoOutArm0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        Ok(())
    }
}
pub struct PoseidonStoreOut__0_SuperLayout {
    pub _0: &'static MemoryWriteLayout,
    pub high: &'static U16RegLayout,
    pub low: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for PoseidonStoreOut__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonStoreOut__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("high", self.high)?;
        v.visit_component("low", self.low)?;
        Ok(())
    }
}
pub type PoseidonStoreOut__0_SuperLayout8LayoutArray =
    [&'static PoseidonStoreOut__0_SuperLayout; 8];
pub struct PoseidonStoreOutLayout {
    pub _super: &'static PoseidonStateLayout,
    pub is_normal: &'static IsZeroLayout,
    pub ext_inv: &'static NondetExtRegLayout,
    pub _0: &'static PoseidonStoreOut__0_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonStoreOutLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonStoreOutLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("is_normal", self.is_normal)?;
        v.visit_component("ext_inv", self.ext_inv)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct PoseidonDoOut_SuperLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static PoseidonDoOutArm0Layout,
    pub arm1: &'static PoseidonStoreOutLayout,
}
impl risc0_zkp::layout::Component for PoseidonDoOut_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonDoOut_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct _Arguments_PoseidonDoOut_SuperLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
    pub arg_u16: &'static ArgU16Layout16LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_PoseidonDoOut_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_PoseidonDoOut_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        Ok(())
    }
}
pub struct PoseidonDoOutLayout {
    pub _super: &'static PoseidonDoOut_SuperLayout,
    pub _arguments__super: &'static _Arguments_PoseidonDoOut_SuperLayout,
}
impl risc0_zkp::layout::Component for PoseidonDoOutLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonDoOutLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_arguments__super", self._arguments__super)?;
        Ok(())
    }
}
pub struct Poseidon0Arm5Layout {
    pub _super: &'static PoseidonDoOutLayout,
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm5Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm5Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct PoseidonPaging_SuperLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static PoseidonStateLayout,
    pub arm1: &'static PoseidonStateLayout,
    pub arm2: &'static PoseidonStateLayout,
    pub arm3: &'static PoseidonStateLayout,
    pub arm4: &'static PoseidonStateLayout,
    pub arm5: &'static PoseidonStateLayout,
}
impl risc0_zkp::layout::Component for PoseidonPaging_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonPaging_SuperLayout"
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
pub type NondetRegLayout6LayoutArray = [&'static NondetRegLayout; 6];
pub struct OneHot_6_Layout {
    pub _super: &'static NondetRegLayout6LayoutArray,
}
impl risc0_zkp::layout::Component for OneHot_6_Layout {
    fn ty_name(&self) -> &'static str {
        "OneHot_6_Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct U8RegLayout {
    pub ret: &'static NondetU8RegLayout,
}
impl risc0_zkp::layout::Component for U8RegLayout {
    fn ty_name(&self) -> &'static str {
        "U8RegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("ret", self.ret)?;
        Ok(())
    }
}
pub struct IsU24Layout {
    pub low16: &'static NondetU16RegLayout,
    pub _0: &'static U8RegLayout,
}
impl risc0_zkp::layout::Component for IsU24Layout {
    fn ty_name(&self) -> &'static str {
        "IsU24Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low16", self.low16)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct PoseidonPagingArm0_SuperLayout {
    pub _0: &'static IsU24Layout,
}
impl risc0_zkp::layout::Component for PoseidonPagingArm0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonPagingArm0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct PoseidonPagingArm1_SuperLayout {
    pub _0: &'static IsU24Layout,
}
impl risc0_zkp::layout::Component for PoseidonPagingArm1_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonPagingArm1_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct PoseidonPaging__0Layout {
    pub arm0: &'static PoseidonPagingArm0_SuperLayout,
    pub arm1: &'static PoseidonPagingArm1_SuperLayout,
}
impl risc0_zkp::layout::Component for PoseidonPaging__0Layout {
    fn ty_name(&self) -> &'static str {
        "PoseidonPaging__0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type ArgU16Layout1LayoutArray = [&'static ArgU16Layout; 1];
pub type ArgU8Layout1LayoutArray = [&'static ArgU8Layout; 1];
pub struct _Arguments_PoseidonPaging__1Layout {
    pub arg_u16: &'static ArgU16Layout1LayoutArray,
    pub arg_u8: &'static ArgU8Layout1LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_PoseidonPaging__1Layout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_PoseidonPaging__1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct PoseidonPagingLayout {
    pub _super: &'static PoseidonPaging_SuperLayout,
    pub mode_split: &'static OneHot_6_Layout,
    pub _2: &'static PoseidonPaging__0Layout,
    pub _0: &'static IsU24Layout,
    pub _arguments__1: &'static _Arguments_PoseidonPaging__1Layout,
    pub _3: &'static NondetRegLayout,
    pub cur_idx: &'static NondetRegLayout,
    pub cur_mode: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for PoseidonPagingLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonPagingLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("mode_split", self.mode_split)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_arguments__1", self._arguments__1)?;
        v.visit_component("_3", self._3)?;
        v.visit_component("cur_idx", self.cur_idx)?;
        v.visit_component("cur_mode", self.cur_mode)?;
        Ok(())
    }
}
pub struct Poseidon0Arm6Layout {
    pub _super: &'static PoseidonPagingLayout,
    pub _extra24: &'static ArgU16Layout,
    pub _extra25: &'static ArgU16Layout,
    pub _extra26: &'static ArgU16Layout,
    pub _extra27: &'static ArgU16Layout,
    pub _extra28: &'static ArgU16Layout,
    pub _extra29: &'static ArgU16Layout,
    pub _extra30: &'static ArgU16Layout,
    pub _extra31: &'static ArgU16Layout,
    pub _extra32: &'static ArgU16Layout,
    pub _extra33: &'static ArgU16Layout,
    pub _extra34: &'static ArgU16Layout,
    pub _extra35: &'static ArgU16Layout,
    pub _extra36: &'static ArgU16Layout,
    pub _extra37: &'static ArgU16Layout,
    pub _extra0: &'static MemoryArgLayout,
    pub _extra1: &'static MemoryArgLayout,
    pub _extra2: &'static MemoryArgLayout,
    pub _extra3: &'static MemoryArgLayout,
    pub _extra4: &'static MemoryArgLayout,
    pub _extra5: &'static MemoryArgLayout,
    pub _extra6: &'static MemoryArgLayout,
    pub _extra7: &'static MemoryArgLayout,
    pub _extra16: &'static CycleArgLayout,
    pub _extra17: &'static CycleArgLayout,
    pub _extra18: &'static CycleArgLayout,
    pub _extra19: &'static CycleArgLayout,
    pub _extra8: &'static MemoryArgLayout,
    pub _extra9: &'static MemoryArgLayout,
    pub _extra10: &'static MemoryArgLayout,
    pub _extra11: &'static MemoryArgLayout,
    pub _extra12: &'static MemoryArgLayout,
    pub _extra13: &'static MemoryArgLayout,
    pub _extra14: &'static MemoryArgLayout,
    pub _extra15: &'static MemoryArgLayout,
    pub _extra20: &'static CycleArgLayout,
    pub _extra21: &'static CycleArgLayout,
    pub _extra22: &'static CycleArgLayout,
    pub _extra23: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm6Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm6Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra24", self._extra24)?;
        v.visit_component("_extra25", self._extra25)?;
        v.visit_component("_extra26", self._extra26)?;
        v.visit_component("_extra27", self._extra27)?;
        v.visit_component("_extra28", self._extra28)?;
        v.visit_component("_extra29", self._extra29)?;
        v.visit_component("_extra30", self._extra30)?;
        v.visit_component("_extra31", self._extra31)?;
        v.visit_component("_extra32", self._extra32)?;
        v.visit_component("_extra33", self._extra33)?;
        v.visit_component("_extra34", self._extra34)?;
        v.visit_component("_extra35", self._extra35)?;
        v.visit_component("_extra36", self._extra36)?;
        v.visit_component("_extra37", self._extra37)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        v.visit_component("_extra2", self._extra2)?;
        v.visit_component("_extra3", self._extra3)?;
        v.visit_component("_extra4", self._extra4)?;
        v.visit_component("_extra5", self._extra5)?;
        v.visit_component("_extra6", self._extra6)?;
        v.visit_component("_extra7", self._extra7)?;
        v.visit_component("_extra16", self._extra16)?;
        v.visit_component("_extra17", self._extra17)?;
        v.visit_component("_extra18", self._extra18)?;
        v.visit_component("_extra19", self._extra19)?;
        v.visit_component("_extra8", self._extra8)?;
        v.visit_component("_extra9", self._extra9)?;
        v.visit_component("_extra10", self._extra10)?;
        v.visit_component("_extra11", self._extra11)?;
        v.visit_component("_extra12", self._extra12)?;
        v.visit_component("_extra13", self._extra13)?;
        v.visit_component("_extra14", self._extra14)?;
        v.visit_component("_extra15", self._extra15)?;
        v.visit_component("_extra20", self._extra20)?;
        v.visit_component("_extra21", self._extra21)?;
        v.visit_component("_extra22", self._extra22)?;
        v.visit_component("_extra23", self._extra23)?;
        Ok(())
    }
}
pub struct PoseidonStoreState__0_SuperLayout {
    pub _0: &'static MemoryWriteLayout,
    pub high: &'static U16RegLayout,
    pub low: &'static NondetU16RegLayout,
}
impl risc0_zkp::layout::Component for PoseidonStoreState__0_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonStoreState__0_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        v.visit_component("high", self.high)?;
        v.visit_component("low", self.low)?;
        Ok(())
    }
}
pub type PoseidonStoreState__0_SuperLayout8LayoutArray =
    [&'static PoseidonStoreState__0_SuperLayout; 8];
pub struct PoseidonStoreStateLayout {
    pub _super: &'static PoseidonStateLayout,
    pub _0: &'static PoseidonStoreState__0_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for PoseidonStoreStateLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonStoreStateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct Poseidon0Arm7Layout {
    pub _super: &'static PoseidonStoreStateLayout,
    pub _extra0: &'static ArgU8Layout,
    pub _extra1: &'static ArgU8Layout,
}
impl risc0_zkp::layout::Component for Poseidon0Arm7Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Arm7Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_extra0", self._extra0)?;
        v.visit_component("_extra1", self._extra1)?;
        Ok(())
    }
}
pub struct Poseidon0StateLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static Poseidon0Arm0Layout,
    pub arm1: &'static Poseidon0Arm1Layout,
    pub arm2: &'static Poseidon0Arm2Layout,
    pub arm3: &'static Poseidon0Arm3Layout,
    pub arm4: &'static Poseidon0Arm4Layout,
    pub arm5: &'static Poseidon0Arm5Layout,
    pub arm6: &'static Poseidon0Arm6Layout,
    pub arm7: &'static Poseidon0Arm7Layout,
}
impl risc0_zkp::layout::Component for Poseidon0StateLayout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0StateLayout"
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
        Ok(())
    }
}
pub type ArgU8Layout2LayoutArray = [&'static ArgU8Layout; 2];
pub struct _Arguments_Poseidon0StateLayout {
    pub memory_arg: &'static MemoryArgLayout16LayoutArray,
    pub cycle_arg: &'static CycleArgLayout8LayoutArray,
    pub arg_u16: &'static ArgU16Layout16LayoutArray,
    pub arg_u8: &'static ArgU8Layout2LayoutArray,
}
impl risc0_zkp::layout::Component for _Arguments_Poseidon0StateLayout {
    fn ty_name(&self) -> &'static str {
        "_Arguments_Poseidon0StateLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("arg_u8", self.arg_u8)?;
        Ok(())
    }
}
pub struct Poseidon0Layout {
    pub state: &'static Poseidon0StateLayout,
    pub arg: &'static CycleArgLayout,
    pub _arguments_state: &'static _Arguments_Poseidon0StateLayout,
}
impl risc0_zkp::layout::Component for Poseidon0Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("state", self.state)?;
        v.visit_component("arg", self.arg)?;
        v.visit_component("_arguments_state", self._arguments_state)?;
        Ok(())
    }
}
pub struct SBoxLayout {
    pub _super: &'static NondetRegLayout,
    pub cubed: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for SBoxLayout {
    fn ty_name(&self) -> &'static str {
        "SBoxLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("cubed", self.cubed)?;
        Ok(())
    }
}
pub type SBoxLayout24LayoutArray = [&'static SBoxLayout; 24];
pub struct DoExtRoundLayout {
    pub _0: &'static SBoxLayout24LayoutArray,
}
impl risc0_zkp::layout::Component for DoExtRoundLayout {
    fn ty_name(&self) -> &'static str {
        "DoExtRoundLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct DoExtRoundByIdxLayout {
    pub _super: &'static DoExtRoundLayout,
    pub idx_hot: &'static OneHot_8_Layout,
}
impl risc0_zkp::layout::Component for DoExtRoundByIdxLayout {
    fn ty_name(&self) -> &'static str {
        "DoExtRoundByIdxLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("idx_hot", self.idx_hot)?;
        Ok(())
    }
}
pub struct PoseidonExtRoundLayout {
    pub _super: &'static PoseidonStateLayout,
    pub next_inner: &'static DoExtRoundByIdxLayout,
    pub is_round3: &'static IsZeroLayout,
    pub is_round7: &'static IsZeroLayout,
    pub last_block: &'static IsZeroLayout,
}
impl risc0_zkp::layout::Component for PoseidonExtRoundLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonExtRoundLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("next_inner", self.next_inner)?;
        v.visit_component("is_round3", self.is_round3)?;
        v.visit_component("is_round7", self.is_round7)?;
        v.visit_component("last_block", self.last_block)?;
        Ok(())
    }
}
pub struct DoIntRoundLayout {
    pub sbox: &'static SBoxLayout,
}
impl risc0_zkp::layout::Component for DoIntRoundLayout {
    fn ty_name(&self) -> &'static str {
        "DoIntRoundLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("sbox", self.sbox)?;
        Ok(())
    }
}
pub type DoIntRoundLayout21LayoutArray = [&'static DoIntRoundLayout; 21];
pub struct DoIntRoundsLayout {
    pub _super: &'static DoIntRoundLayout21LayoutArray,
}
impl risc0_zkp::layout::Component for DoIntRoundsLayout {
    fn ty_name(&self) -> &'static str {
        "DoIntRoundsLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct PoseidonIntRoundsLayout {
    pub _super: &'static PoseidonStateLayout,
    pub next_inner: &'static DoIntRoundsLayout,
}
impl risc0_zkp::layout::Component for PoseidonIntRoundsLayout {
    fn ty_name(&self) -> &'static str {
        "PoseidonIntRoundsLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("next_inner", self.next_inner)?;
        Ok(())
    }
}
pub struct Poseidon1StateLayout {
    pub _super: &'static PoseidonStateLayout,
    pub arm0: &'static PoseidonExtRoundLayout,
    pub arm1: &'static PoseidonIntRoundsLayout,
    pub arm2: &'static PoseidonStateLayout,
    pub arm3: &'static PoseidonStateLayout,
    pub arm4: &'static PoseidonStateLayout,
    pub arm5: &'static PoseidonStateLayout,
    pub arm6: &'static PoseidonStateLayout,
    pub arm7: &'static PoseidonStateLayout,
}
impl risc0_zkp::layout::Component for Poseidon1StateLayout {
    fn ty_name(&self) -> &'static str {
        "Poseidon1StateLayout"
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
        Ok(())
    }
}
pub struct Poseidon1Layout {
    pub state: &'static Poseidon1StateLayout,
    pub arg: &'static CycleArgLayout,
}
impl risc0_zkp::layout::Component for Poseidon1Layout {
    fn ty_name(&self) -> &'static str {
        "Poseidon1Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("state", self.state)?;
        v.visit_component("arg", self.arg)?;
        Ok(())
    }
}
pub struct TopInstResultLayout {
    pub _selector: &'static NondetRegLayout11LayoutArray,
    pub arm0: &'static Misc0Layout,
    pub arm1: &'static Misc1Layout,
    pub arm2: &'static Misc2Layout,
    pub arm3: &'static Mul0Layout,
    pub arm4: &'static Div0Layout,
    pub arm5: &'static Mem0Layout,
    pub arm6: &'static Mem1Layout,
    pub arm7: &'static Control0Layout,
    pub arm8: &'static ECall0Layout,
    pub arm9: &'static Poseidon0Layout,
    pub arm10: &'static Poseidon1Layout,
}
impl risc0_zkp::layout::Component for TopInstResultLayout {
    fn ty_name(&self) -> &'static str {
        "TopInstResultLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
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
        Ok(())
    }
}
pub struct TopLayout {
    pub next_pc_low: &'static NondetRegLayout,
    pub next_pc_high: &'static NondetRegLayout,
    pub next_state_0: &'static NondetRegLayout,
    pub next_machine_mode: &'static NondetRegLayout,
    pub is_first_cycle: &'static NondetRegLayout,
    pub cycle_nd: &'static NondetRegLayout,
    pub cycle: &'static NondetRegLayout,
    pub major: &'static NondetRegLayout,
    pub minor: &'static NondetRegLayout,
    pub inst_input: &'static InstInputLayout,
    pub major_onehot: &'static OneHot_11_Layout,
    pub inst_result: &'static TopInstResultLayout,
}
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("next_pc_low", self.next_pc_low)?;
        v.visit_component("next_pc_high", self.next_pc_high)?;
        v.visit_component("next_state_0", self.next_state_0)?;
        v.visit_component("next_machine_mode", self.next_machine_mode)?;
        v.visit_component("is_first_cycle", self.is_first_cycle)?;
        v.visit_component("cycle_nd", self.cycle_nd)?;
        v.visit_component("cycle", self.cycle)?;
        v.visit_component("major", self.major)?;
        v.visit_component("minor", self.minor)?;
        v.visit_component("inst_input", self.inst_input)?;
        v.visit_component("major_onehot", self.major_onehot)?;
        v.visit_component("inst_result", self.inst_result)?;
        Ok(())
    }
}
pub struct DigestRegValues_SuperLayout {
    pub low: &'static NondetRegLayout,
    pub high: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for DigestRegValues_SuperLayout {
    fn ty_name(&self) -> &'static str {
        "DigestRegValues_SuperLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("low", self.low)?;
        v.visit_component("high", self.high)?;
        Ok(())
    }
}
pub type DigestRegValues_SuperLayout8LayoutArray = [&'static DigestRegValues_SuperLayout; 8];
pub struct DigestRegLayout {
    pub values: &'static DigestRegValues_SuperLayout8LayoutArray,
}
impl risc0_zkp::layout::Component for DigestRegLayout {
    fn ty_name(&self) -> &'static str {
        "DigestRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("values", self.values)?;
        Ok(())
    }
}
pub struct Arg_ArgU8Layout {
    pub val: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_ArgU8Layout {
    fn ty_name(&self) -> &'static str {
        "Arg_ArgU8Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("val", self.val)?;
        Ok(())
    }
}
pub struct Arg_ArgU16Layout {
    pub val: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_ArgU16Layout {
    fn ty_name(&self) -> &'static str {
        "Arg_ArgU16Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("val", self.val)?;
        Ok(())
    }
}
pub struct Arg_MemoryArgLayout {
    pub addr: &'static Reg,
    pub cycle: &'static Reg,
    pub data_low: &'static Reg,
    pub data_high: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_MemoryArgLayout {
    fn ty_name(&self) -> &'static str {
        "Arg_MemoryArgLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("addr", self.addr)?;
        v.visit_component("cycle", self.cycle)?;
        v.visit_component("data_low", self.data_low)?;
        v.visit_component("data_high", self.data_high)?;
        Ok(())
    }
}
pub struct Arg_CycleArgLayout {
    pub cycle: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_CycleArgLayout {
    fn ty_name(&self) -> &'static str {
        "Arg_CycleArgLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cycle", self.cycle)?;
        Ok(())
    }
}
pub struct _accumLayout {
    pub arg_u8: &'static Arg_ArgU8Layout,
    pub arg_u16: &'static Arg_ArgU16Layout,
    pub memory_arg: &'static Arg_MemoryArgLayout,
    pub cycle_arg: &'static Arg_CycleArgLayout,
    pub _offset: &'static Reg,
}
impl risc0_zkp::layout::Component for _accumLayout {
    fn ty_name(&self) -> &'static str {
        "_accumLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("arg_u8", self.arg_u8)?;
        v.visit_component("arg_u16", self.arg_u16)?;
        v.visit_component("memory_arg", self.memory_arg)?;
        v.visit_component("cycle_arg", self.cycle_arg)?;
        v.visit_component("_offset", self._offset)?;
        Ok(())
    }
}
pub type Reg19LayoutArray = [&'static Reg; 19];
pub struct LayoutAccumLayout {
    pub columns: &'static Reg19LayoutArray,
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
pub struct TestSuccRunLayout {
    pub _0: &'static TopLayout,
}
impl risc0_zkp::layout::Component for TestSuccRunLayout {
    fn ty_name(&self) -> &'static str {
        "TestSuccRunLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_0", self._0)?;
        Ok(())
    }
}
pub struct _globalLayout {
    pub input: &'static DigestRegLayout,
    pub is_terminate: &'static NondetRegLayout,
    pub output: &'static DigestRegLayout,
    pub rng: &'static NondetExtRegLayout,
    pub state_in: &'static DigestRegLayout,
    pub state_out: &'static DigestRegLayout,
    pub term_a0high: &'static NondetRegLayout,
    pub term_a0low: &'static NondetRegLayout,
    pub term_a1high: &'static NondetRegLayout,
    pub term_a1low: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("input", self.input)?;
        v.visit_component("is_terminate", self.is_terminate)?;
        v.visit_component("output", self.output)?;
        v.visit_component("rng", self.rng)?;
        v.visit_component("state_in", self.state_in)?;
        v.visit_component("state_out", self.state_out)?;
        v.visit_component("term_a0high", self.term_a0high)?;
        v.visit_component("term_a0low", self.term_a0low)?;
        v.visit_component("term_a1high", self.term_a1high)?;
        v.visit_component("term_a1low", self.term_a1low)?;
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
pub struct NondetExtRegStruct {
    pub _super: ExtVal,
}
#[derive(Copy, Clone, Debug)]
pub struct BitRegStruct {}
#[derive(Copy, Clone, Debug)]
pub struct NondetFakeTwitRegStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct FakeTwitRegStruct {}
#[derive(Copy, Clone, Debug)]
pub struct ArgU8Struct {
    pub count: NondetRegStruct,
    pub val: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct U8RegStruct {}
#[derive(Copy, Clone, Debug)]
pub struct ArgU16Struct {
    pub count: NondetRegStruct,
    pub val: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct U16RegStruct {
    pub _super: Val,
}
pub type Val5Array = [Val; 5];
pub type Val16Array = [Val; 16];
pub type NondetRegStruct5Array = [NondetRegStruct; 5];
#[derive(Copy, Clone, Debug)]
pub struct ToBits_5_Struct {
    pub _super: NondetRegStruct5Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ValU32Struct {
    pub low: Val,
    pub high: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct DenormedValU32Struct {
    pub low: Val,
    pub high: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct NormalizeU32Struct {
    pub _super: ValU32Struct,
    pub carry: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct AddrDecomposeStruct {
    pub _super: Val,
    pub low2: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct AddrDecomposeBitsStruct {
    pub _super: Val,
    pub low0: NondetRegStruct,
    pub low1: NondetRegStruct,
    pub low2: Val,
    pub addr: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct CmpEqualStruct {
    pub is_equal: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct CmpLessThanUnsignedStruct {
    pub is_less_than: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct CmpLessThanStruct {
    pub is_less_than: NondetRegStruct,
}
pub type NondetRegStruct16Array = [NondetRegStruct; 16];
#[derive(Copy, Clone, Debug)]
pub struct ToBits_16_Struct {
    pub _super: NondetRegStruct16Array,
}
#[derive(Copy, Clone, Debug)]
pub struct FromBits_16_Struct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct DecoderStruct {
    pub opcode: NondetRegStruct,
    pub rs1: Val,
    pub rs2: Val,
    pub rd: Val,
    pub func7: Val,
    pub func3: Val,
    pub imm_i: ValU32Struct,
    pub imm_s: ValU32Struct,
    pub imm_b: ValU32Struct,
    pub imm_u: ValU32Struct,
    pub imm_j: ValU32Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct MemoryArgStruct {
    pub count: NondetRegStruct,
    pub addr: NondetRegStruct,
    pub cycle: NondetRegStruct,
    pub data_low: NondetRegStruct,
    pub data_high: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct CycleArgStruct {
    pub count: NondetRegStruct,
    pub cycle: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct IsCycleStruct {}
#[derive(Copy, Clone, Debug)]
pub struct MemoryIOStruct {
    pub old_txn: MemoryArgStruct,
    pub new_txn: MemoryArgStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct IsForwardStruct {}
#[derive(Copy, Clone, Debug)]
pub struct GetDataStruct {
    pub _super: ValU32Struct,
    pub diff_low: Val,
    pub diff_high: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct MemoryWriteStruct {}
#[derive(Copy, Clone, Debug)]
pub struct MemoryWriteUnconstrainedStruct {}
pub type Val3Array = [Val; 3];
pub type NondetRegStruct3Array = [NondetRegStruct; 3];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_3_Struct {
    pub _super: NondetRegStruct3Array,
}
pub type Val8Array = [Val; 8];
pub type NondetRegStruct8Array = [NondetRegStruct; 8];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_8_Struct {
    pub _super: NondetRegStruct8Array,
    pub bits: NondetRegStruct8Array,
}
#[derive(Copy, Clone, Debug)]
pub struct InstInputStruct {
    pub pc_u32: ValU32Struct,
    pub state: Val,
    pub mode: Val,
    pub minor_onehot: OneHot_8_Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct WriteRdStruct {}
#[derive(Copy, Clone, Debug)]
pub struct ExpandU32Struct {
    pub b0: NondetRegStruct,
    pub b1: NondetRegStruct,
    pub b2: NondetRegStruct,
    pub b3: NondetRegStruct,
    pub neg: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct SplitTotalStruct {
    pub out: NondetRegStruct,
    pub carry: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct MultiplySettingsStruct {
    pub a_signed: Val,
    pub b_signed: Val,
    pub c_signed: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct MultiplyAccumulateStruct {
    pub out_low: ValU32Struct,
    pub out_high: ValU32Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct DivInputStruct {
    pub _super: InstInputStruct,
    pub ii: InstInputStruct,
    pub decoded: DecoderStruct,
    pub rs1: GetDataStruct,
    pub rs2: GetDataStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct DoDivStruct {
    pub quot: ValU32Struct,
    pub rem: ValU32Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct InstOutputStruct {
    pub new_pc: ValU32Struct,
    pub new_state: Val,
    pub new_mode: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct MiscInputStruct {
    pub _super: InstInputStruct,
    pub ii: InstInputStruct,
    pub decoded: DecoderStruct,
    pub rs1: GetDataStruct,
    pub rs2: GetDataStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct MiscOutputStruct {
    pub do_write: Val,
    pub to_write: DenormedValU32Struct,
    pub new_pc: DenormedValU32Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct MulInputStruct {
    pub _super: InstInputStruct,
    pub ii: InstInputStruct,
    pub decoded: DecoderStruct,
    pub rs1: GetDataStruct,
    pub rs2: GetDataStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct DoMulStruct {
    pub low: ValU32Struct,
    pub high: ValU32Struct,
}
#[derive(Copy, Clone, Debug)]
pub struct MemLoadInputStruct {
    pub ii: InstInputStruct,
    pub decoded: DecoderStruct,
    pub addr: AddrDecomposeBitsStruct,
    pub data_0: GetDataStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct MemStoreInputStruct {
    pub decoded: DecoderStruct,
    pub rs2: GetDataStruct,
    pub addr: AddrDecomposeBitsStruct,
    pub data_0: GetDataStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct MemStoreFinalizeStruct {}
#[derive(Copy, Clone, Debug)]
pub struct SplitWordStruct {
    pub byte0: NondetRegStruct,
    pub byte1: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct DigestRegValues_SuperStruct {
    pub low: NondetRegStruct,
    pub high: NondetRegStruct,
}
pub type DigestRegValues_SuperStruct8Array = [DigestRegValues_SuperStruct; 8];
#[derive(Copy, Clone, Debug)]
pub struct DigestRegStruct {
    pub values: DigestRegValues_SuperStruct8Array,
}
pub type ValU32Struct8Array = [ValU32Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct ControlLoadRoot__0Struct {}
pub type ControlLoadRoot__0Struct8Array = [ControlLoadRoot__0Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct ControlResumeArm1_Super__0Struct {}
pub type ControlResumeArm1_Super__0Struct8Array = [ControlResumeArm1_Super__0Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}
pub type GetDataStruct8Array = [GetDataStruct; 8];
#[derive(Copy, Clone, Debug)]
pub struct ControlTableArm0_Super__0Struct {}
#[derive(Copy, Clone, Debug)]
pub struct ControlTableArm1_Super__0Struct {}
pub type ControlTableArm0_Super__0Struct16Array = [ControlTableArm0_Super__0Struct; 16];
pub type ControlTableArm1_Super__0Struct16Array = [ControlTableArm1_Super__0Struct; 16];
pub type Val4Array = [Val; 4];
pub type NondetRegStruct4Array = [NondetRegStruct; 4];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_4_Struct {
    pub _super: NondetRegStruct4Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ECallOutputStruct {
    pub state: Val,
    pub s0: Val,
    pub s1: Val,
    pub s2: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct DecomposeLow2Struct {
    pub high: NondetRegStruct,
    pub low2: NondetRegStruct,
    pub low2_hot: OneHot_4_Struct,
    pub high_zero: NondetRegStruct,
    pub is_zero: NondetRegStruct,
    pub low2_nonzero: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct ECallHostReadWords__0Struct {}
pub type ECallHostReadWords__0Struct4Array = [ECallHostReadWords__0Struct; 4];
pub type Val24Array = [Val; 24];
#[derive(Copy, Clone, Debug)]
pub struct MultiplyByMInt_Super_SuperStruct {
    pub _super: Val,
}
pub type MultiplyByMInt_Super_SuperStruct24Array = [MultiplyByMInt_Super_SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct MultiplyByMIntStruct {
    pub _super: MultiplyByMInt_Super_SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct DoIntRounds__0_SuperStruct {
    pub _super: Val,
}
pub type DoIntRounds__0_SuperStruct21Array = [DoIntRounds__0_SuperStruct; 21];
#[derive(Copy, Clone, Debug)]
pub struct DoIntRoundsStruct {
    pub _super: Val24Array,
}
pub type NondetRegStruct24Array = [NondetRegStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct MultiplyByMExt_Super_SuperStruct {
    pub _super: Val,
}
pub type MultiplyByMExt_Super_SuperStruct24Array = [MultiplyByMExt_Super_SuperStruct; 24];
#[derive(Copy, Clone, Debug)]
pub struct MultiplyByMExtStruct {
    pub _super: MultiplyByMExt_Super_SuperStruct24Array,
}
#[derive(Copy, Clone, Debug)]
pub struct PoseidonStateStruct {
    pub has_state: NondetRegStruct,
    pub state_addr: NondetRegStruct,
    pub buf_out_addr: NondetRegStruct,
    pub is_elem: NondetRegStruct,
    pub check_out: NondetRegStruct,
    pub load_tx_type: NondetRegStruct,
    pub next_state: NondetRegStruct,
    pub sub_state: NondetRegStruct,
    pub buf_in_addr: NondetRegStruct,
    pub count: NondetRegStruct,
    pub mode: NondetRegStruct,
    pub inner: NondetRegStruct24Array,
    pub zcheck: NondetExtRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct PoseidonOpDefStruct {
    pub has_state: Val,
    pub state_addr: Val,
    pub buf_out_addr: Val,
    pub is_elem: Val,
    pub check_out: Val,
    pub load_tx_type: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct ReadAddrStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct ReadElemStruct {
    pub _super: Val,
}
pub type ReadElemStruct8Array = [ReadElemStruct; 8];
#[derive(Copy, Clone, Debug)]
pub struct PoseidonCheckOut__0Struct {}
pub type PoseidonCheckOut__0Struct8Array = [PoseidonCheckOut__0Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct PoseidonStoreOut__0Struct {}
pub type PoseidonStoreOut__0Struct8Array = [PoseidonStoreOut__0Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct PoseidonStoreState__0Struct {}
pub type PoseidonStoreState__0Struct8Array = [PoseidonStoreState__0Struct; 8];
#[derive(Copy, Clone, Debug)]
pub struct IsU24Struct {}
pub type Val6Array = [Val; 6];
pub type NondetRegStruct6Array = [NondetRegStruct; 6];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_6_Struct {
    pub _super: NondetRegStruct6Array,
    pub bits: NondetRegStruct6Array,
}
pub type Val11Array = [Val; 11];
pub type NondetRegStruct11Array = [NondetRegStruct; 11];
#[derive(Copy, Clone, Debug)]
pub struct OneHot_11_Struct {
    pub _super: NondetRegStruct11Array,
}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {}
