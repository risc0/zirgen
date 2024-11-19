struct NondetRegLayout {
  Reg _super;
};
using NondetRegLayout8LayoutArray = NondetRegLayout[8];
struct OneHot_8_Layout {
  NondetRegLayout8LayoutArray _super;
};
struct InstInputLayout {
  OneHot_8_Layout minorOnehot;
};
using NondetRegLayout11LayoutArray = NondetRegLayout[11];
struct OneHot_11_Layout {
  NondetRegLayout11LayoutArray _super;
};
struct MemoryArgLayout {
  NondetRegLayout count;
  NondetRegLayout addr;
  NondetRegLayout cycle;
  NondetRegLayout dataLow;
  NondetRegLayout dataHigh;
};
struct MemoryIOLayout {
  MemoryArgLayout oldTxn;
  MemoryArgLayout newTxn;
};
struct CycleArgLayout {
  NondetRegLayout count;
  NondetRegLayout cycle;
};
struct IsCycleLayout {
  CycleArgLayout arg;
};
struct IsForwardLayout {
  IsCycleLayout _0;
};
struct MemoryWriteLayout {
  MemoryIOLayout io;
  IsForwardLayout _0;
};
struct IsZeroLayout {
  NondetRegLayout _super;
  NondetRegLayout inv;
};
struct WriteRdLayout {
  MemoryWriteLayout _0;
  IsZeroLayout isRd0;
  NondetRegLayout writeAddr;
};
struct ArgU16Layout {
  NondetRegLayout count;
  NondetRegLayout val;
};
struct NondetU16RegLayout {
  ArgU16Layout arg;
};
struct NormalizeU32Layout {
  NondetU16RegLayout low16;
  NondetU16RegLayout high16;
  NondetRegLayout lowCarry;
  NondetRegLayout highCarry;
};
struct FinalizeMiscLayout {
  WriteRdLayout _0;
  NormalizeU32Layout writeData;
  NormalizeU32Layout pcNorm;
};
struct Misc0Arm0Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc0Arm1Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
using NondetRegLayout16LayoutArray = NondetRegLayout[16];
struct ToBits_16_Layout {
  NondetRegLayout16LayoutArray _super;
};
struct BitwiseAndU16Layout {
  ToBits_16_Layout bitsX;
  ToBits_16_Layout bitsY;
};
struct BitwiseAndLayout {
  BitwiseAndU16Layout _0;
  BitwiseAndU16Layout _1;
};
struct BitwiseXorLayout {
  BitwiseAndLayout andXy;
};
struct OpXORLayout {
  BitwiseXorLayout _0;
};
struct Misc0Arm2Layout {
  OpXORLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct BitwiseOrLayout {
  BitwiseAndLayout andXy;
};
struct OpORLayout {
  BitwiseOrLayout _0;
};
struct Misc0Arm3Layout {
  OpORLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpANDLayout {
  BitwiseAndLayout _0;
};
struct Misc0Arm4Layout {
  OpANDLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct GetSignU32Layout {
  NondetRegLayout _super;
  NondetU16RegLayout restTimesTwo;
};
struct CmpLessThanLayout {
  NormalizeU32Layout diff;
  GetSignU32Layout s1;
  GetSignU32Layout s2;
  GetSignU32Layout s3;
  NondetRegLayout overflow;
  NondetRegLayout isLessThan;
};
struct OpSLTLayout {
  CmpLessThanLayout cmp;
};
struct CmpLessThanUnsignedLayout {
  NormalizeU32Layout diff;
};
struct OpSLTULayout {
  CmpLessThanUnsignedLayout cmp;
};
struct Misc0Arm6Layout {
  OpSLTULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
};
struct Misc0Arm7Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc0MiscOutputLayout {
  Misc0Arm0Layout arm0;
  Misc0Arm1Layout arm1;
  Misc0Arm2Layout arm2;
  Misc0Arm3Layout arm3;
  Misc0Arm4Layout arm4;
  OpSLTLayout arm5;
  Misc0Arm6Layout arm6;
  Misc0Arm7Layout arm7;
};
using ArgU16Layout5LayoutArray = ArgU16Layout[5];
struct _Arguments_Misc0MiscOutputLayout {
  ArgU16Layout5LayoutArray argU16;
};
struct DecoderLayout {
  NondetRegLayout _f7_6;
  NondetRegLayout _f7_45;
  NondetRegLayout _f7_23;
  NondetRegLayout _f7_01;
  NondetRegLayout _rs2_34;
  NondetRegLayout _rs2_12;
  NondetRegLayout _rs2_0;
  NondetRegLayout _rs1_34;
  NondetRegLayout _rs1_12;
  NondetRegLayout _rs1_0;
  NondetRegLayout _f3_2;
  NondetRegLayout _f3_01;
  NondetRegLayout _rd_34;
  NondetRegLayout _rd_12;
  NondetRegLayout _rd_0;
  NondetRegLayout opcode;
};
struct U16RegLayout {
  NondetU16RegLayout ret;
};
struct AddrDecomposeLayout {
  NondetRegLayout low2;
  U16RegLayout upperDiff;
  IsZeroLayout _0;
  NondetU16RegLayout med14;
};
struct MemoryReadLayout {
  MemoryIOLayout io;
  IsForwardLayout _0;
};
struct DecodeInstLayout {
  DecoderLayout _super;
  CycleArgLayout arg;
  AddrDecomposeLayout pcAddr;
  MemoryReadLayout loadInst;
};
struct ReadRegLayout {
  MemoryReadLayout _super;
  NondetRegLayout addr;
};
struct MiscInputLayout {
  DecodeInstLayout decoded;
  ReadRegLayout rs1;
  ReadRegLayout rs2;
};
struct Misc0Layout {
  FinalizeMiscLayout _super;
  Misc0MiscOutputLayout miscOutput;
  _Arguments_Misc0MiscOutputLayout _argumentsMiscOutput;
  MiscInputLayout input;
};
struct OpXORILayout {
  BitwiseXorLayout _0;
};
struct Misc1Arm0Layout {
  OpXORILayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpORILayout {
  BitwiseOrLayout _0;
};
struct Misc1Arm1Layout {
  OpORILayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpANDILayout {
  BitwiseAndLayout _0;
};
struct Misc1Arm2Layout {
  OpANDILayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpSLTILayout {
  CmpLessThanLayout cmp;
};
struct OpSLTIULayout {
  CmpLessThanUnsignedLayout cmp;
};
struct Misc1Arm4Layout {
  OpSLTIULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
};
struct CmpEqualLayout {
  IsZeroLayout lowSame;
  IsZeroLayout highSame;
  NondetRegLayout isEqual;
};
struct OpBEQLayout {
  CmpEqualLayout cmp;
};
struct Misc1Arm5Layout {
  OpBEQLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpBNELayout {
  CmpEqualLayout cmp;
};
struct Misc1Arm6Layout {
  OpBNELayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct OpBLTLayout {
  CmpLessThanLayout cmp;
};
struct Misc1MiscOutputLayout {
  Misc1Arm0Layout arm0;
  Misc1Arm1Layout arm1;
  Misc1Arm2Layout arm2;
  OpSLTILayout arm3;
  Misc1Arm4Layout arm4;
  Misc1Arm5Layout arm5;
  Misc1Arm6Layout arm6;
  OpBLTLayout arm7;
};
struct _Arguments_Misc1MiscOutputLayout {
  ArgU16Layout5LayoutArray argU16;
};
struct Misc1Layout {
  FinalizeMiscLayout _super;
  Misc1MiscOutputLayout miscOutput;
  _Arguments_Misc1MiscOutputLayout _argumentsMiscOutput;
  MiscInputLayout input;
};
struct OpBGELayout {
  CmpLessThanLayout cmp;
};
struct OpBLTULayout {
  CmpLessThanUnsignedLayout cmp;
};
struct Misc2Arm1Layout {
  OpBLTULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
};
struct OpBGEULayout {
  CmpLessThanUnsignedLayout cmp;
};
struct Misc2Arm2Layout {
  OpBGEULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
};
struct Misc2Arm3Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc2Arm4Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc2Arm5Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc2Arm6Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc2Arm7Layout {
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
};
struct Misc2MiscOutputLayout {
  OpBGELayout arm0;
  Misc2Arm1Layout arm1;
  Misc2Arm2Layout arm2;
  Misc2Arm3Layout arm3;
  Misc2Arm4Layout arm4;
  Misc2Arm5Layout arm5;
  Misc2Arm6Layout arm6;
  Misc2Arm7Layout arm7;
};
struct _Arguments_Misc2MiscOutputLayout {
  ArgU16Layout5LayoutArray argU16;
};
struct Misc2Layout {
  FinalizeMiscLayout _super;
  Misc2MiscOutputLayout miscOutput;
  _Arguments_Misc2MiscOutputLayout _argumentsMiscOutput;
  MiscInputLayout input;
};
struct MulInputLayout {
  DecodeInstLayout decoded;
  ReadRegLayout rs1;
  ReadRegLayout rs2;
};
struct ArgU8Layout {
  NondetRegLayout count;
  NondetRegLayout val;
};
struct NondetU8RegLayout {
  ArgU8Layout arg;
};
struct ExpandU32Layout {
  NondetU8RegLayout b0;
  NondetU8RegLayout b1;
  NondetU8RegLayout b2;
  NondetU8RegLayout b3;
  NondetU8RegLayout b3Top7times2;
  NondetRegLayout topBit;
};
struct NondetFakeTwitRegLayout {
  NondetRegLayout reg0;
  NondetRegLayout reg1;
};
struct SplitTotalLayout {
  NondetU16RegLayout out;
  NondetU8RegLayout carryByte;
  NondetFakeTwitRegLayout carryExtra;
};
struct MultiplyAccumulateLayout {
  ExpandU32Layout ax;
  ExpandU32Layout bx;
  NondetRegLayout cSign;
  NondetU16RegLayout cRestTimes2;
  SplitTotalLayout s0;
  SplitTotalLayout s1;
  SplitTotalLayout s2;
  NondetU16RegLayout s3Out;
  NondetFakeTwitRegLayout s3Carry;
};
struct DoMulLayout {
  MultiplyAccumulateLayout mul;
};
using NondetRegLayout5LayoutArray = NondetRegLayout[5];
struct ToBits_5_Layout {
  NondetRegLayout5LayoutArray _super;
};
struct DynPo2Layout {
  ToBits_5_Layout low5;
  NondetU16RegLayout checkU16;
  NondetRegLayout b3;
  NondetRegLayout low;
  NondetRegLayout high;
};
struct OpSLLLayout {
  DoMulLayout _0;
  DynPo2Layout shiftMul;
};
struct OpSLLILayout {
  DoMulLayout _0;
  DynPo2Layout shiftMul;
};
struct OpMULLayout {
  DoMulLayout _0;
};
struct Mul0Arm2Layout {
  OpMULLayout _super;
  ArgU16Layout _extra0;
};
struct OpMULHLayout {
  DoMulLayout _0;
};
struct Mul0Arm3Layout {
  OpMULHLayout _super;
  ArgU16Layout _extra0;
};
struct OpMULHSULayout {
  DoMulLayout _0;
};
struct Mul0Arm4Layout {
  OpMULHSULayout _super;
  ArgU16Layout _extra0;
};
struct OpMULHULayout {
  DoMulLayout _0;
};
struct Mul0Arm5Layout {
  OpMULHULayout _super;
  ArgU16Layout _extra0;
};
struct Mul0Arm6Layout {
  ArgU16Layout _extra0;
  ArgU8Layout _extra6;
  ArgU8Layout _extra7;
  ArgU8Layout _extra8;
  ArgU8Layout _extra9;
  ArgU8Layout _extra10;
  ArgU8Layout _extra11;
  ArgU8Layout _extra12;
  ArgU8Layout _extra13;
  ArgU8Layout _extra14;
  ArgU8Layout _extra15;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU8Layout _extra18;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
};
struct Mul0Arm7Layout {
  ArgU16Layout _extra0;
  ArgU8Layout _extra6;
  ArgU8Layout _extra7;
  ArgU8Layout _extra8;
  ArgU8Layout _extra9;
  ArgU8Layout _extra10;
  ArgU8Layout _extra11;
  ArgU8Layout _extra12;
  ArgU8Layout _extra13;
  ArgU8Layout _extra14;
  ArgU8Layout _extra15;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU8Layout _extra18;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
};
struct Mul0MulOutputLayout {
  OpSLLLayout arm0;
  OpSLLILayout arm1;
  Mul0Arm2Layout arm2;
  Mul0Arm3Layout arm3;
  Mul0Arm4Layout arm4;
  Mul0Arm5Layout arm5;
  Mul0Arm6Layout arm6;
  Mul0Arm7Layout arm7;
};
using ArgU16Layout6LayoutArray = ArgU16Layout[6];
using ArgU8Layout13LayoutArray = ArgU8Layout[13];
struct _Arguments_Mul0MulOutputLayout {
  ArgU16Layout6LayoutArray argU16;
  ArgU8Layout13LayoutArray argU8;
};
struct Mul0Layout {
  MulInputLayout input;
  Mul0MulOutputLayout mulOutput;
  WriteRdLayout _0;
  _Arguments_Mul0MulOutputLayout _argumentsMulOutput;
  NormalizeU32Layout pcAdd;
};
struct DoDivLayout {
  NondetRegLayout quotLow;
  NondetRegLayout quotHigh;
  NondetU16RegLayout remLow;
  NondetU16RegLayout remHigh;
  MultiplyAccumulateLayout mul;
  NondetRegLayout topBitType;
};
struct OpSRLLayout {
  DoDivLayout _0;
  DynPo2Layout shiftMul;
};
struct Div0Arm0Layout {
  OpSRLLayout _super;
  ArgU16Layout _extra0;
};
struct TopBitLayout {
  NondetRegLayout _super;
  NondetU16RegLayout rest;
};
struct OpSRALayout {
  DoDivLayout _0;
  DynPo2Layout shiftMul;
  TopBitLayout flip;
};
struct OpSRLILayout {
  DoDivLayout _0;
  DynPo2Layout shiftMul;
};
struct Div0Arm2Layout {
  OpSRLILayout _super;
  ArgU16Layout _extra0;
};
struct OpSRAILayout {
  DoDivLayout _0;
  DynPo2Layout shiftMul;
  TopBitLayout flip;
};
struct OpDIVLayout {
  DoDivLayout _0;
};
struct Div0Arm4Layout {
  OpDIVLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
};
struct OpDIVULayout {
  DoDivLayout _0;
};
struct Div0Arm5Layout {
  OpDIVULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
};
struct OpREMLayout {
  DoDivLayout _0;
};
struct Div0Arm6Layout {
  OpREMLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
};
struct OpREMULayout {
  DoDivLayout _0;
};
struct Div0Arm7Layout {
  OpREMULayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
};
struct Div0MulOutputLayout {
  Div0Arm0Layout arm0;
  OpSRALayout arm1;
  Div0Arm2Layout arm2;
  OpSRAILayout arm3;
  Div0Arm4Layout arm4;
  Div0Arm5Layout arm5;
  Div0Arm6Layout arm6;
  Div0Arm7Layout arm7;
};
struct DivInputLayout {
  DecodeInstLayout decoded;
  ReadRegLayout rs1;
  ReadRegLayout rs2;
};
using ArgU16Layout9LayoutArray = ArgU16Layout[9];
struct _Arguments_Div0MulOutputLayout {
  ArgU8Layout13LayoutArray argU8;
  ArgU16Layout9LayoutArray argU16;
};
struct Div0Layout {
  Div0MulOutputLayout mulOutput;
  DivInputLayout input;
  WriteRdLayout _0;
  _Arguments_Div0MulOutputLayout _argumentsMulOutput;
  NormalizeU32Layout pcAdd;
};
struct AddrDecomposeBitsLayout {
  NondetRegLayout low0;
  NondetRegLayout low1;
  U16RegLayout upperDiff;
  IsZeroLayout _0;
  NondetU16RegLayout med14;
};
struct MemLoadInputLayout {
  DecodeInstLayout decoded;
  ReadRegLayout rs1;
  MemoryReadLayout data;
  AddrDecomposeBitsLayout addr;
  NormalizeU32Layout addrU32;
};
struct SplitWordLayout {
  NondetU8RegLayout byte0;
  NondetU8RegLayout byte1;
};
struct OpLBLayout {
  SplitWordLayout bytes;
  NondetU8RegLayout low7x2;
  NondetRegLayout highBit;
};
struct OpLHLayout {
  NondetU8RegLayout low15x2;
  NondetRegLayout highBit;
};
struct Mem0Arm1Layout {
  OpLHLayout _super;
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
};
struct Mem0Arm2Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
};
struct OpLBULayout {
  SplitWordLayout bytes;
};
struct Mem0Arm3Layout {
  OpLBULayout _super;
  ArgU8Layout _extra0;
};
struct Mem0Arm4Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
};
struct Mem0Arm5Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
};
struct Mem0Arm6Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
};
struct Mem0Arm7Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
};
struct Mem0OutputLayout {
  OpLBLayout arm0;
  Mem0Arm1Layout arm1;
  Mem0Arm2Layout arm2;
  Mem0Arm3Layout arm3;
  Mem0Arm4Layout arm4;
  Mem0Arm5Layout arm5;
  Mem0Arm6Layout arm6;
  Mem0Arm7Layout arm7;
};
using ArgU8Layout3LayoutArray = ArgU8Layout[3];
struct _Arguments_Mem0OutputLayout {
  ArgU8Layout3LayoutArray argU8;
};
struct Mem0Layout {
  MemLoadInputLayout input;
  WriteRdLayout _0;
  Mem0OutputLayout output;
  _Arguments_Mem0OutputLayout _argumentsOutput;
  NormalizeU32Layout pcAdd;
};
struct MemStoreInputLayout {
  DecodeInstLayout decoded;
  ReadRegLayout rs1;
  ReadRegLayout rs2;
  MemoryReadLayout data;
  AddrDecomposeBitsLayout addr;
  NormalizeU32Layout addrU32;
};
struct MemStoreFinalizeLayout {
  MemoryWriteLayout _0;
};
struct OpSBLayout {
  SplitWordLayout origBytes;
  SplitWordLayout newBytes;
};
struct Mem1Arm1Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm2Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm3Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm4Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm5Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm6Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1Arm7Layout {
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
};
struct Mem1OutputLayout {
  OpSBLayout arm0;
  Mem1Arm1Layout arm1;
  Mem1Arm2Layout arm2;
  Mem1Arm3Layout arm3;
  Mem1Arm4Layout arm4;
  Mem1Arm5Layout arm5;
  Mem1Arm6Layout arm6;
  Mem1Arm7Layout arm7;
};
using ArgU8Layout4LayoutArray = ArgU8Layout[4];
struct _Arguments_Mem1OutputLayout {
  ArgU8Layout4LayoutArray argU8;
};
struct Mem1Layout {
  MemStoreInputLayout input;
  MemStoreFinalizeLayout _0;
  Mem1OutputLayout output;
  _Arguments_Mem1OutputLayout _argumentsOutput;
  NormalizeU32Layout pcAdd;
};
struct MemoryPageInLayout {
  MemoryIOLayout io;
};
struct ControlLoadRoot__0_SuperLayout {
  MemoryPageInLayout mem;
};
using ControlLoadRoot__0_SuperLayout8LayoutArray = ControlLoadRoot__0_SuperLayout[8];
struct ControlLoadRootLayout {
  ControlLoadRoot__0_SuperLayout8LayoutArray _0;
};
struct Control0Arm0Layout {
  ControlLoadRootLayout _super;
  ArgU8Layout _extra24;
  ArgU8Layout _extra25;
  ArgU8Layout _extra26;
  ArgU8Layout _extra27;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU8Layout _extra32;
  ArgU8Layout _extra33;
  ArgU8Layout _extra34;
  ArgU8Layout _extra35;
  ArgU8Layout _extra36;
  ArgU8Layout _extra37;
  ArgU8Layout _extra38;
  ArgU8Layout _extra39;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
  ArgU16Layout _extra16;
  ArgU16Layout _extra17;
  ArgU16Layout _extra18;
  ArgU16Layout _extra19;
  ArgU16Layout _extra20;
  CycleArgLayout _extra0;
  CycleArgLayout _extra1;
  CycleArgLayout _extra2;
  CycleArgLayout _extra3;
  ArgU16Layout _extra21;
  ArgU16Layout _extra22;
  ArgU16Layout _extra23;
  CycleArgLayout _extra4;
  CycleArgLayout _extra5;
  CycleArgLayout _extra6;
  CycleArgLayout _extra7;
};
struct ControlResumeArm0_SuperLayout {
  MemoryReadLayout pc;
  MemoryReadLayout mode;
};
struct ControlResumeArm0Layout {
  ControlResumeArm0_SuperLayout _super;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  CycleArgLayout _extra12;
  CycleArgLayout _extra13;
  CycleArgLayout _extra14;
  CycleArgLayout _extra15;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
};
struct ControlResumeArm1_Super__0_SuperLayout {
  MemoryWriteLayout _0;
};
using ControlResumeArm1_Super__0_SuperLayout8LayoutArray = ControlResumeArm1_Super__0_SuperLayout[8];
struct ControlResumeArm1_SuperLayout {
  ControlResumeArm1_Super__0_SuperLayout8LayoutArray _0;
};
struct ControlResume_SuperLayout {
  ControlResumeArm0Layout arm0;
  ControlResumeArm1_SuperLayout arm1;
};
using MemoryArgLayout16LayoutArray = MemoryArgLayout[16];
using CycleArgLayout8LayoutArray = CycleArgLayout[8];
struct _Arguments_ControlResume_SuperLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
};
struct ControlResumeLayout {
  ControlResume_SuperLayout _super;
  IsZeroLayout pcZero;
  _Arguments_ControlResume_SuperLayout _arguments_Super;
};
struct Control0Arm1Layout {
  ControlResumeLayout _super;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU8Layout _extra18;
  ArgU8Layout _extra19;
  ArgU8Layout _extra20;
  ArgU8Layout _extra21;
  ArgU8Layout _extra22;
  ArgU8Layout _extra23;
  ArgU8Layout _extra24;
  ArgU8Layout _extra25;
  ArgU8Layout _extra26;
  ArgU8Layout _extra27;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct ControlUserECALLLayout {
  MemoryWriteLayout _1;
  MemoryReadLayout loadInst;
  U16RegLayout _0;
  MemoryReadLayout dispatchIdx;
  MemoryReadLayout newPcAddr;
  AddrDecomposeBitsLayout pcAddr;
  NondetRegLayout safeMode;
};
struct Control0Arm2Layout {
  ControlUserECALLLayout _super;
  ArgU8Layout _extra25;
  ArgU8Layout _extra26;
  ArgU8Layout _extra27;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU8Layout _extra32;
  ArgU8Layout _extra33;
  ArgU8Layout _extra34;
  ArgU8Layout _extra35;
  ArgU8Layout _extra36;
  ArgU8Layout _extra37;
  ArgU8Layout _extra38;
  ArgU8Layout _extra39;
  ArgU8Layout _extra40;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
  ArgU16Layout _extra16;
  ArgU16Layout _extra17;
  ArgU16Layout _extra18;
  ArgU16Layout _extra19;
  ArgU16Layout _extra20;
  ArgU16Layout _extra21;
  ArgU16Layout _extra22;
  ArgU16Layout _extra23;
  ArgU16Layout _extra24;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
};
struct ControlMRETLayout {
  NormalizeU32Layout pcAdd;
  MemoryReadLayout loadInst;
  MemoryReadLayout pc;
  AddrDecomposeBitsLayout pcAddr;
  NondetRegLayout safeMode;
};
struct Control0Arm3Layout {
  ControlMRETLayout _super;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU8Layout _extra32;
  ArgU8Layout _extra33;
  ArgU8Layout _extra34;
  ArgU8Layout _extra35;
  ArgU8Layout _extra36;
  ArgU8Layout _extra37;
  ArgU8Layout _extra38;
  ArgU8Layout _extra39;
  ArgU8Layout _extra40;
  ArgU8Layout _extra41;
  ArgU8Layout _extra42;
  ArgU8Layout _extra43;
  ArgU8Layout _extra44;
  ArgU8Layout _extra45;
  ArgU16Layout _extra18;
  ArgU16Layout _extra19;
  ArgU16Layout _extra20;
  ArgU16Layout _extra21;
  ArgU16Layout _extra22;
  ArgU16Layout _extra23;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  ArgU16Layout _extra28;
  ArgU16Layout _extra29;
  CycleArgLayout _extra12;
  CycleArgLayout _extra13;
  CycleArgLayout _extra14;
  CycleArgLayout _extra15;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
};
using MemoryReadLayout8LayoutArray = MemoryReadLayout[8];
struct ControlSuspendArm0_SuperLayout {
  MemoryReadLayout8LayoutArray _0;
};
struct ControlSuspendArm1_SuperLayout {
  MemoryWriteLayout _0;
  MemoryWriteLayout _1;
  NondetRegLayout state;
};
struct ControlSuspendArm1Layout {
  ControlSuspendArm1_SuperLayout _super;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  CycleArgLayout _extra12;
  CycleArgLayout _extra13;
  CycleArgLayout _extra14;
  CycleArgLayout _extra15;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
};
struct ControlSuspend_SuperLayout {
  ControlSuspendArm0_SuperLayout arm0;
  ControlSuspendArm1Layout arm1;
};
struct _Arguments_ControlSuspend_SuperLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
};
struct ControlSuspendLayout {
  ControlSuspend_SuperLayout _super;
  IsZeroLayout pcZero;
  _Arguments_ControlSuspend_SuperLayout _arguments_Super;
};
struct Control0Arm4Layout {
  ControlSuspendLayout _super;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU8Layout _extra18;
  ArgU8Layout _extra19;
  ArgU8Layout _extra20;
  ArgU8Layout _extra21;
  ArgU8Layout _extra22;
  ArgU8Layout _extra23;
  ArgU8Layout _extra24;
  ArgU8Layout _extra25;
  ArgU8Layout _extra26;
  ArgU8Layout _extra27;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct MemoryPageOutLayout {
  MemoryIOLayout io;
  IsForwardLayout _0;
};
using MemoryPageOutLayout8LayoutArray = MemoryPageOutLayout[8];
struct ControlStoreRootLayout {
  MemoryPageOutLayout8LayoutArray _0;
};
struct Control0Arm5Layout {
  ControlStoreRootLayout _super;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU8Layout _extra18;
  ArgU8Layout _extra19;
  ArgU8Layout _extra20;
  ArgU8Layout _extra21;
  ArgU8Layout _extra22;
  ArgU8Layout _extra23;
  ArgU8Layout _extra24;
  ArgU8Layout _extra25;
  ArgU8Layout _extra26;
  ArgU8Layout _extra27;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU8Layout _extra30;
  ArgU8Layout _extra31;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct ControlTableArm0_Super__0_SuperLayout {
  ArgU16Layout arg;
};
using ControlTableArm0_Super__0_SuperLayout16LayoutArray = ControlTableArm0_Super__0_SuperLayout[16];
struct ControlTableArm0_SuperLayout {
  IsZeroLayout done;
  ControlTableArm0_Super__0_SuperLayout16LayoutArray _0;
};
struct ControlTableArm0Layout {
  ControlTableArm0_SuperLayout _super;
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
  ArgU8Layout _extra2;
  ArgU8Layout _extra3;
  ArgU8Layout _extra4;
  ArgU8Layout _extra5;
  ArgU8Layout _extra6;
  ArgU8Layout _extra7;
  ArgU8Layout _extra8;
  ArgU8Layout _extra9;
  ArgU8Layout _extra10;
  ArgU8Layout _extra11;
  ArgU8Layout _extra12;
  ArgU8Layout _extra13;
  ArgU8Layout _extra14;
  ArgU8Layout _extra15;
};
struct ControlTableArm1_Super__0_SuperLayout {
  ArgU8Layout arg;
};
using ControlTableArm1_Super__0_SuperLayout16LayoutArray = ControlTableArm1_Super__0_SuperLayout[16];
struct ControlTableArm1_SuperLayout {
  IsZeroLayout done;
  ControlTableArm1_Super__0_SuperLayout16LayoutArray _0;
};
struct ControlTableArm1Layout {
  ControlTableArm1_SuperLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct ControlTable_SuperLayout {
  ControlTableArm0Layout arm0;
  ControlTableArm1Layout arm1;
};
using ArgU16Layout16LayoutArray = ArgU16Layout[16];
using ArgU8Layout16LayoutArray = ArgU8Layout[16];
struct _Arguments_ControlTable_SuperLayout {
  ArgU16Layout16LayoutArray argU16;
  ArgU8Layout16LayoutArray argU8;
};
struct ControlTableLayout {
  ControlTable_SuperLayout _super;
  _Arguments_ControlTable_SuperLayout _arguments_Super;
  NondetRegLayout entry;
  NondetRegLayout mode;
};
struct Control0Arm6Layout {
  ControlTableLayout _super;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  CycleArgLayout _extra18;
  CycleArgLayout _extra19;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra20;
  CycleArgLayout _extra21;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  CycleArgLayout _extra22;
  CycleArgLayout _extra23;
  MemoryArgLayout _extra12;
  MemoryArgLayout _extra13;
  MemoryArgLayout _extra14;
  MemoryArgLayout _extra15;
};
struct Control0Arm7Layout {
  ArgU8Layout _extra40;
  ArgU8Layout _extra41;
  ArgU8Layout _extra42;
  ArgU8Layout _extra43;
  ArgU8Layout _extra44;
  ArgU8Layout _extra45;
  ArgU8Layout _extra46;
  ArgU8Layout _extra47;
  ArgU8Layout _extra48;
  ArgU8Layout _extra49;
  ArgU8Layout _extra50;
  ArgU8Layout _extra51;
  ArgU8Layout _extra52;
  ArgU8Layout _extra53;
  ArgU8Layout _extra54;
  ArgU8Layout _extra55;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  ArgU16Layout _extra28;
  ArgU16Layout _extra29;
  ArgU16Layout _extra30;
  ArgU16Layout _extra31;
  ArgU16Layout _extra32;
  ArgU16Layout _extra33;
  ArgU16Layout _extra34;
  ArgU16Layout _extra35;
  ArgU16Layout _extra36;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  CycleArgLayout _extra18;
  CycleArgLayout _extra19;
  ArgU16Layout _extra37;
  ArgU16Layout _extra38;
  ArgU16Layout _extra39;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra20;
  CycleArgLayout _extra21;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  CycleArgLayout _extra22;
  CycleArgLayout _extra23;
  MemoryArgLayout _extra12;
  MemoryArgLayout _extra13;
  MemoryArgLayout _extra14;
  MemoryArgLayout _extra15;
};
struct Control0_SuperLayout {
  Control0Arm0Layout arm0;
  Control0Arm1Layout arm1;
  Control0Arm2Layout arm2;
  Control0Arm3Layout arm3;
  Control0Arm4Layout arm4;
  Control0Arm5Layout arm5;
  Control0Arm6Layout arm6;
  Control0Arm7Layout arm7;
};
struct _Arguments_Control0_SuperLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
  ArgU16Layout16LayoutArray argU16;
  ArgU8Layout16LayoutArray argU8;
};
struct Control0Layout {
  Control0_SuperLayout _super;
  CycleArgLayout arg;
  _Arguments_Control0_SuperLayout _arguments_Super;
};
using NondetRegLayout4LayoutArray = NondetRegLayout[4];
struct OneHot_4_Layout {
  NondetRegLayout4LayoutArray _super;
};
struct MachineECallLayout {
  MemoryReadLayout loadInst;
  MemoryReadLayout dispatchIdx;
  OneHot_4_Layout dispatch;
};
struct ECall0Arm0Layout {
  MachineECallLayout _super;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  CycleArgLayout _extra4;
  CycleArgLayout _extra5;
};
struct ECallTerminateLayout {
  MemoryReadLayout a0;
  MemoryReadLayout a1;
};
struct ECall0Arm1Layout {
  ECallTerminateLayout _super;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  CycleArgLayout _extra4;
  CycleArgLayout _extra5;
};
struct DecomposeLow2Layout {
  OneHot_4_Layout low2Hot;
  IsZeroLayout highZero;
  NondetRegLayout isZero;
  NondetRegLayout high;
  NondetRegLayout low2;
};
struct ECallHostReadSetupLayout {
  MemoryWriteLayout _0;
  MemoryReadLayout fd;
  MemoryReadLayout ptr;
  MemoryReadLayout len;
  U16RegLayout diff;
  NondetU16RegLayout newLen;
  DecomposeLow2Layout ptrDecomp;
  DecomposeLow2Layout lenDecomp;
  NondetRegLayout len123;
  NondetRegLayout uneven;
};
struct ECallHostWriteLayout {
  MemoryWriteLayout _0;
  MemoryReadLayout fd;
  MemoryReadLayout ptr;
  MemoryReadLayout len;
  U16RegLayout diff;
  NondetU16RegLayout newLen;
};
struct ECall0Arm4Layout {
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
};
struct MemoryWriteUnconstrainedLayout {
  MemoryIOLayout io;
  IsForwardLayout _0;
};
struct ECallHostReadWords__0_SuperLayout {
  NondetRegLayout addr;
  MemoryWriteUnconstrainedLayout _0;
};
using ECallHostReadWords__0_SuperLayout4LayoutArray = ECallHostReadWords__0_SuperLayout[4];
struct ECallHostReadWordsLayout {
  DecomposeLow2Layout lenDecomp;
  IsZeroLayout lenZero;
  DecomposeLow2Layout wordsDecomp;
  ECallHostReadWords__0_SuperLayout4LayoutArray _0;
};
struct ECall0Arm5Layout {
  ECallHostReadWordsLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
};
struct ECall0Arm6Layout {
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
};
struct ECall0Arm7Layout {
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
};
struct ECall0OutputLayout {
  ECall0Arm0Layout arm0;
  ECall0Arm1Layout arm1;
  ECallHostReadSetupLayout arm2;
  ECallHostWriteLayout arm3;
  ECall0Arm4Layout arm4;
  ECall0Arm5Layout arm5;
  ECall0Arm6Layout arm6;
  ECall0Arm7Layout arm7;
};
using MemoryArgLayout8LayoutArray = MemoryArgLayout[8];
using CycleArgLayout4LayoutArray = CycleArgLayout[4];
using ArgU16Layout2LayoutArray = ArgU16Layout[2];
struct _Arguments_ECall0OutputLayout {
  MemoryArgLayout8LayoutArray memoryArg;
  CycleArgLayout4LayoutArray cycleArg;
  ArgU16Layout2LayoutArray argU16;
};
struct ECall0Layout {
  ECall0OutputLayout output;
  _Arguments_ECall0OutputLayout _argumentsOutput;
  AddrDecomposeBitsLayout pcAddr;
  IsZeroLayout isDecode;
  NondetRegLayout s0;
  NormalizeU32Layout addPC;
  CycleArgLayout arg;
  IsZeroLayout isP2Entry;
  NondetRegLayout s1;
  NondetRegLayout s2;
};
using NondetRegLayout24LayoutArray = NondetRegLayout[24];
struct NondetExtRegLayout {
  Reg _super;
};
struct PoseidonStateLayout {
  NondetRegLayout hasState;
  NondetRegLayout24LayoutArray inner;
  NondetRegLayout stateAddr;
  NondetRegLayout bufOutAddr;
  NondetRegLayout isElem;
  NondetRegLayout checkOut;
  NondetRegLayout loadTxType;
  NondetRegLayout nextState;
  NondetRegLayout subState;
  NondetRegLayout bufInAddr;
  NondetRegLayout count;
  NondetRegLayout mode;
  NondetExtRegLayout zcheck;
};
struct PoseidonEntryArm0Layout {
  PoseidonStateLayout _super;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
};
struct ReadAddrLayout {
  MemoryReadLayout addr32;
};
struct PoseidonEcallLayout {
  PoseidonStateLayout _super;
  ReadAddrLayout stateAddr;
  ReadAddrLayout bufInAddr;
  ReadAddrLayout bufOutAddr;
  MemoryReadLayout bitsAndCount;
  IsZeroLayout _0;
  IsZeroLayout countZero;
  NondetRegLayout isElem;
  NondetRegLayout checkOut;
};
struct PoseidonEntry_SuperLayout {
  PoseidonStateLayout _super;
  PoseidonEntryArm0Layout arm0;
  PoseidonEcallLayout arm1;
};
struct _Arguments_PoseidonEntry_SuperLayout {
  MemoryArgLayout8LayoutArray memoryArg;
  CycleArgLayout4LayoutArray cycleArg;
};
struct PoseidonEntryLayout {
  PoseidonEntry_SuperLayout _super;
  _Arguments_PoseidonEntry_SuperLayout _arguments_Super;
  IsZeroLayout pcZero;
};
struct Poseidon0Arm0Layout {
  PoseidonEntryLayout _super;
  ArgU8Layout _extra28;
  ArgU8Layout _extra29;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
  ArgU16Layout _extra16;
  ArgU16Layout _extra17;
  ArgU16Layout _extra18;
  ArgU16Layout _extra19;
  ArgU16Layout _extra20;
  ArgU16Layout _extra21;
  ArgU16Layout _extra22;
  ArgU16Layout _extra23;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra8;
  CycleArgLayout _extra9;
  CycleArgLayout _extra10;
  CycleArgLayout _extra11;
};
struct ReadElemLayout {
  MemoryReadLayout elem32;
};
using ReadElemLayout8LayoutArray = ReadElemLayout[8];
struct PoseidonLoadStateLayout {
  PoseidonStateLayout _super;
  ReadElemLayout8LayoutArray loadList;
};
struct Poseidon0Arm1Layout {
  PoseidonLoadStateLayout _super;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
using NondetRegLayout3LayoutArray = NondetRegLayout[3];
struct OneHot_3_Layout {
  NondetRegLayout3LayoutArray _super;
};
struct MemoryGetArm1Layout {
  MemoryPageInLayout _super;
  CycleArgLayout _extra0;
};
struct MemoryGet_SuperLayout {
  MemoryReadLayout arm0;
  MemoryGetArm1Layout arm1;
  MemoryPageOutLayout arm2;
};
using MemoryArgLayout2LayoutArray = MemoryArgLayout[2];
using CycleArgLayout1LayoutArray = CycleArgLayout[1];
struct _Arguments_MemoryGet_SuperLayout {
  MemoryArgLayout2LayoutArray memoryArg;
  CycleArgLayout1LayoutArray cycleArg;
};
struct MemoryGetLayout {
  MemoryGet_SuperLayout _super;
  _Arguments_MemoryGet_SuperLayout _arguments_Super;
};
using MemoryGetLayout8LayoutArray = MemoryGetLayout[8];
struct PoseidonLoadInShortLayout {
  PoseidonStateLayout _super;
  OneHot_3_Layout txType;
  MemoryGetLayout8LayoutArray loadList;
};
struct PoseidonLoadInLowLayout {
  PoseidonStateLayout _super;
  OneHot_3_Layout txType;
  MemoryGetLayout8LayoutArray loadList;
};
struct PoseidonLoadInHighLayout {
  PoseidonStateLayout _super;
  OneHot_3_Layout txType;
  MemoryGetLayout8LayoutArray loadList;
};
struct PoseidonLoadIn_SuperLayout {
  PoseidonStateLayout _super;
  PoseidonLoadInShortLayout arm0;
  PoseidonLoadInLowLayout arm1;
  PoseidonLoadInHighLayout arm2;
};
struct _Arguments_PoseidonLoadIn_SuperLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
};
struct PoseidonLoadInLayout {
  PoseidonLoadIn_SuperLayout _super;
  OneHot_3_Layout _0;
  _Arguments_PoseidonLoadIn_SuperLayout _arguments_Super;
};
struct Poseidon0Arm2Layout {
  PoseidonLoadInLayout _super;
  ArgU8Layout _extra16;
  ArgU8Layout _extra17;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct Poseidon0Arm3Layout {
  PoseidonStateLayout _super;
  ArgU8Layout _extra40;
  ArgU8Layout _extra41;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  ArgU16Layout _extra28;
  ArgU16Layout _extra29;
  ArgU16Layout _extra30;
  ArgU16Layout _extra31;
  ArgU16Layout _extra32;
  ArgU16Layout _extra33;
  ArgU16Layout _extra34;
  ArgU16Layout _extra35;
  ArgU16Layout _extra36;
  ArgU16Layout _extra37;
  ArgU16Layout _extra38;
  ArgU16Layout _extra39;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  CycleArgLayout _extra18;
  CycleArgLayout _extra19;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  MemoryArgLayout _extra12;
  MemoryArgLayout _extra13;
  MemoryArgLayout _extra14;
  MemoryArgLayout _extra15;
  CycleArgLayout _extra20;
  CycleArgLayout _extra21;
  CycleArgLayout _extra22;
  CycleArgLayout _extra23;
};
struct Poseidon0Arm4Layout {
  PoseidonStateLayout _super;
  ArgU8Layout _extra40;
  ArgU8Layout _extra41;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  ArgU16Layout _extra28;
  ArgU16Layout _extra29;
  ArgU16Layout _extra30;
  ArgU16Layout _extra31;
  ArgU16Layout _extra32;
  ArgU16Layout _extra33;
  ArgU16Layout _extra34;
  ArgU16Layout _extra35;
  ArgU16Layout _extra36;
  ArgU16Layout _extra37;
  ArgU16Layout _extra38;
  ArgU16Layout _extra39;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  CycleArgLayout _extra18;
  CycleArgLayout _extra19;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  MemoryArgLayout _extra12;
  MemoryArgLayout _extra13;
  MemoryArgLayout _extra14;
  MemoryArgLayout _extra15;
  CycleArgLayout _extra20;
  CycleArgLayout _extra21;
  CycleArgLayout _extra22;
  CycleArgLayout _extra23;
};
struct PoseidonCheckOut__0_SuperLayout {
  ReadElemLayout goal;
};
using PoseidonCheckOut__0_SuperLayout8LayoutArray = PoseidonCheckOut__0_SuperLayout[8];
struct PoseidonCheckOutLayout {
  PoseidonStateLayout _super;
  IsZeroLayout isNormal;
  NondetExtRegLayout extInv;
  PoseidonCheckOut__0_SuperLayout8LayoutArray _0;
};
struct PoseidonDoOutArm0Layout {
  PoseidonCheckOutLayout _super;
  ArgU16Layout _extra0;
  ArgU16Layout _extra1;
  ArgU16Layout _extra2;
  ArgU16Layout _extra3;
  ArgU16Layout _extra4;
  ArgU16Layout _extra5;
  ArgU16Layout _extra6;
  ArgU16Layout _extra7;
  ArgU16Layout _extra8;
  ArgU16Layout _extra9;
  ArgU16Layout _extra10;
  ArgU16Layout _extra11;
  ArgU16Layout _extra12;
  ArgU16Layout _extra13;
  ArgU16Layout _extra14;
  ArgU16Layout _extra15;
};
struct PoseidonStoreOut__0_SuperLayout {
  MemoryWriteLayout _0;
  U16RegLayout high;
  NondetU16RegLayout low;
};
using PoseidonStoreOut__0_SuperLayout8LayoutArray = PoseidonStoreOut__0_SuperLayout[8];
struct PoseidonStoreOutLayout {
  PoseidonStateLayout _super;
  IsZeroLayout isNormal;
  NondetExtRegLayout extInv;
  PoseidonStoreOut__0_SuperLayout8LayoutArray _0;
};
struct PoseidonDoOut_SuperLayout {
  PoseidonStateLayout _super;
  PoseidonDoOutArm0Layout arm0;
  PoseidonStoreOutLayout arm1;
};
struct _Arguments_PoseidonDoOut_SuperLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
  ArgU16Layout16LayoutArray argU16;
};
struct PoseidonDoOutLayout {
  PoseidonDoOut_SuperLayout _super;
  _Arguments_PoseidonDoOut_SuperLayout _arguments_Super;
};
struct Poseidon0Arm5Layout {
  PoseidonDoOutLayout _super;
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
};
struct PoseidonPaging_SuperLayout {
  PoseidonStateLayout _super;
  PoseidonStateLayout arm0;
  PoseidonStateLayout arm1;
  PoseidonStateLayout arm2;
  PoseidonStateLayout arm3;
  PoseidonStateLayout arm4;
  PoseidonStateLayout arm5;
};
using NondetRegLayout6LayoutArray = NondetRegLayout[6];
struct OneHot_6_Layout {
  NondetRegLayout6LayoutArray _super;
};
struct U8RegLayout {
  NondetU8RegLayout ret;
};
struct IsU24Layout {
  NondetU16RegLayout low16;
  U8RegLayout _0;
};
struct PoseidonPagingArm0_SuperLayout {
  IsU24Layout _0;
};
struct PoseidonPagingArm1_SuperLayout {
  IsU24Layout _0;
};
struct PoseidonPaging__0Layout {
  PoseidonPagingArm0_SuperLayout arm0;
  PoseidonPagingArm1_SuperLayout arm1;
};
using ArgU16Layout1LayoutArray = ArgU16Layout[1];
using ArgU8Layout1LayoutArray = ArgU8Layout[1];
struct _Arguments_PoseidonPaging__1Layout {
  ArgU16Layout1LayoutArray argU16;
  ArgU8Layout1LayoutArray argU8;
};
struct PoseidonPagingLayout {
  PoseidonPaging_SuperLayout _super;
  NondetRegLayout curIdx;
  NondetRegLayout curMode;
  OneHot_6_Layout modeSplit;
  PoseidonPaging__0Layout _2;
  IsU24Layout _0;
  _Arguments_PoseidonPaging__1Layout _arguments__1;
  NondetRegLayout _3;
};
struct Poseidon0Arm6Layout {
  PoseidonPagingLayout _super;
  ArgU16Layout _extra24;
  ArgU16Layout _extra25;
  ArgU16Layout _extra26;
  ArgU16Layout _extra27;
  ArgU16Layout _extra28;
  ArgU16Layout _extra29;
  ArgU16Layout _extra30;
  ArgU16Layout _extra31;
  ArgU16Layout _extra32;
  ArgU16Layout _extra33;
  ArgU16Layout _extra34;
  ArgU16Layout _extra35;
  ArgU16Layout _extra36;
  ArgU16Layout _extra37;
  MemoryArgLayout _extra0;
  MemoryArgLayout _extra1;
  MemoryArgLayout _extra2;
  MemoryArgLayout _extra3;
  MemoryArgLayout _extra4;
  MemoryArgLayout _extra5;
  MemoryArgLayout _extra6;
  MemoryArgLayout _extra7;
  CycleArgLayout _extra16;
  CycleArgLayout _extra17;
  CycleArgLayout _extra18;
  CycleArgLayout _extra19;
  MemoryArgLayout _extra8;
  MemoryArgLayout _extra9;
  MemoryArgLayout _extra10;
  MemoryArgLayout _extra11;
  MemoryArgLayout _extra12;
  MemoryArgLayout _extra13;
  MemoryArgLayout _extra14;
  MemoryArgLayout _extra15;
  CycleArgLayout _extra20;
  CycleArgLayout _extra21;
  CycleArgLayout _extra22;
  CycleArgLayout _extra23;
};
struct PoseidonStoreState__0_SuperLayout {
  MemoryWriteLayout _0;
  U16RegLayout high;
  NondetU16RegLayout low;
};
using PoseidonStoreState__0_SuperLayout8LayoutArray = PoseidonStoreState__0_SuperLayout[8];
struct PoseidonStoreStateLayout {
  PoseidonStateLayout _super;
  PoseidonStoreState__0_SuperLayout8LayoutArray _0;
};
struct Poseidon0Arm7Layout {
  PoseidonStoreStateLayout _super;
  ArgU8Layout _extra0;
  ArgU8Layout _extra1;
};
struct Poseidon0StateLayout {
  PoseidonStateLayout _super;
  Poseidon0Arm0Layout arm0;
  Poseidon0Arm1Layout arm1;
  Poseidon0Arm2Layout arm2;
  Poseidon0Arm3Layout arm3;
  Poseidon0Arm4Layout arm4;
  Poseidon0Arm5Layout arm5;
  Poseidon0Arm6Layout arm6;
  Poseidon0Arm7Layout arm7;
};
using ArgU8Layout2LayoutArray = ArgU8Layout[2];
struct _Arguments_Poseidon0StateLayout {
  MemoryArgLayout16LayoutArray memoryArg;
  CycleArgLayout8LayoutArray cycleArg;
  ArgU16Layout16LayoutArray argU16;
  ArgU8Layout2LayoutArray argU8;
};
struct Poseidon0Layout {
  Poseidon0StateLayout state;
  CycleArgLayout arg;
  _Arguments_Poseidon0StateLayout _argumentsState;
};
struct SBoxLayout {
  NondetRegLayout _super;
  NondetRegLayout cubed;
};
using SBoxLayout24LayoutArray = SBoxLayout[24];
struct DoExtRoundLayout {
  SBoxLayout24LayoutArray _0;
};
struct DoExtRoundByIdxLayout {
  DoExtRoundLayout _super;
  OneHot_8_Layout idxHot;
};
struct PoseidonExtRoundLayout {
  PoseidonStateLayout _super;
  DoExtRoundByIdxLayout nextInner;
  IsZeroLayout isRound3;
  IsZeroLayout isRound7;
  IsZeroLayout lastBlock;
};
struct DoIntRoundLayout {
  SBoxLayout sbox;
};
using DoIntRoundLayout21LayoutArray = DoIntRoundLayout[21];
struct DoIntRoundsLayout {
  DoIntRoundLayout21LayoutArray _super;
};
struct PoseidonIntRoundsLayout {
  PoseidonStateLayout _super;
  DoIntRoundsLayout nextInner;
};
struct Poseidon1StateLayout {
  PoseidonStateLayout _super;
  PoseidonExtRoundLayout arm0;
  PoseidonIntRoundsLayout arm1;
  PoseidonStateLayout arm2;
  PoseidonStateLayout arm3;
  PoseidonStateLayout arm4;
  PoseidonStateLayout arm5;
  PoseidonStateLayout arm6;
  PoseidonStateLayout arm7;
};
struct Poseidon1Layout {
  Poseidon1StateLayout state;
  CycleArgLayout arg;
};
struct TopInstResultLayout {
  NondetRegLayout11LayoutArray _selector;
  Misc0Layout arm0;
  Misc1Layout arm1;
  Misc2Layout arm2;
  Mul0Layout arm3;
  Div0Layout arm4;
  Mem0Layout arm5;
  Mem1Layout arm6;
  Control0Layout arm7;
  ECall0Layout arm8;
  Poseidon0Layout arm9;
  Poseidon1Layout arm10;
};
struct TopLayout {
  NondetRegLayout nextPcLow;
  NondetRegLayout nextPcHigh;
  NondetRegLayout nextState_0;
  NondetRegLayout nextMachineMode;
  NondetRegLayout isFirstCycle;
  NondetRegLayout cycleND;
  NondetRegLayout cycle;
  NondetRegLayout major;
  NondetRegLayout minor;
  InstInputLayout instInput;
  OneHot_11_Layout majorOnehot;
  TopInstResultLayout instResult;
};
struct DigestRegValues_SuperLayout {
  NondetRegLayout low;
  NondetRegLayout high;
};
using DigestRegValues_SuperLayout8LayoutArray = DigestRegValues_SuperLayout[8];
struct DigestRegLayout {
  DigestRegValues_SuperLayout8LayoutArray values;
};
struct Arg_ArgU8Layout {
  Reg val;
};
struct Arg_ArgU16Layout {
  Reg val;
};
struct Arg_MemoryArgLayout {
  Reg addr;
  Reg cycle;
  Reg dataLow;
  Reg dataHigh;
};
struct Arg_CycleArgLayout {
  Reg cycle;
};
struct _accumLayout {
  Arg_ArgU8Layout argU8;
  Arg_ArgU16Layout argU16;
  Arg_MemoryArgLayout memoryArg;
  Arg_CycleArgLayout cycleArg;
  Reg _offset;
};
using Reg19LayoutArray = Reg[19];
struct LayoutAccumLayout {
  Reg19LayoutArray columns;
};
struct TestSuccRunLayout {
  TopLayout _0;
};
struct _globalLayout {
  DigestRegLayout input;
  NondetRegLayout isTerminate;
  DigestRegLayout output;
  NondetExtRegLayout rng;
  DigestRegLayout stateIn;
  DigestRegLayout stateOut;
  NondetRegLayout termA0high;
  NondetRegLayout termA0low;
  NondetRegLayout termA1high;
  NondetRegLayout termA1low;
};
struct _mixLayout {
  _accumLayout randomness;
};
struct NondetRegStruct {
  Val _super;
};
struct NondetExtRegStruct {
  ExtVal _super;
};
struct BitRegStruct {
};
struct NondetFakeTwitRegStruct {
  Val _super;
};
struct FakeTwitRegStruct {
};
struct ArgU8Struct {
  NondetRegStruct count;
  NondetRegStruct val;
};
struct U8RegStruct {
};
struct ArgU16Struct {
  NondetRegStruct count;
  NondetRegStruct val;
};
struct U16RegStruct {
  Val _super;
};
using Val5Array = Val[5];
using Val16Array = Val[16];
using NondetRegStruct5Array = NondetRegStruct[5];
struct ToBits_5_Struct {
  NondetRegStruct5Array _super;
};
struct ValU32Struct {
  Val low;
  Val high;
};
struct DenormedValU32Struct {
  Val low;
  Val high;
};
struct NormalizeU32Struct {
  ValU32Struct _super;
  NondetRegStruct carry;
};
struct AddrDecomposeStruct {
  Val _super;
  NondetRegStruct low2;
};
struct AddrDecomposeBitsStruct {
  Val _super;
  NondetRegStruct low0;
  NondetRegStruct low1;
  Val low2;
  Val addr;
};
struct CmpEqualStruct {
  NondetRegStruct isEqual;
};
struct CmpLessThanUnsignedStruct {
  Val isLessThan;
};
struct CmpLessThanStruct {
  NondetRegStruct isLessThan;
};
using NondetRegStruct16Array = NondetRegStruct[16];
struct ToBits_16_Struct {
  NondetRegStruct16Array _super;
};
struct FromBits_16_Struct {
  Val _super;
};
struct DecoderStruct {
  NondetRegStruct opcode;
  Val rs1;
  Val rs2;
  Val rd;
  Val func7;
  Val func3;
  ValU32Struct immI;
  ValU32Struct immS;
  ValU32Struct immB;
  ValU32Struct immU;
  ValU32Struct immJ;
};
struct MemoryArgStruct {
  NondetRegStruct count;
  NondetRegStruct addr;
  NondetRegStruct cycle;
  NondetRegStruct dataLow;
  NondetRegStruct dataHigh;
};
struct CycleArgStruct {
  NondetRegStruct count;
  NondetRegStruct cycle;
};
struct IsCycleStruct {
};
struct MemoryIOStruct {
  MemoryArgStruct oldTxn;
  MemoryArgStruct newTxn;
};
struct IsForwardStruct {
};
struct GetDataStruct {
  ValU32Struct _super;
  Val diffLow;
  Val diffHigh;
};
struct MemoryWriteStruct {
};
struct MemoryWriteUnconstrainedStruct {
};
using Val3Array = Val[3];
using NondetRegStruct3Array = NondetRegStruct[3];
struct OneHot_3_Struct {
  NondetRegStruct3Array _super;
};
using Val8Array = Val[8];
using NondetRegStruct8Array = NondetRegStruct[8];
struct OneHot_8_Struct {
  NondetRegStruct8Array _super;
  NondetRegStruct8Array bits;
};
struct InstInputStruct {
  ValU32Struct pcU32;
  Val state;
  Val mode;
  OneHot_8_Struct minorOnehot;
};
struct WriteRdStruct {
};
struct ExpandU32Struct {
  NondetRegStruct b0;
  NondetRegStruct b1;
  NondetRegStruct b2;
  NondetRegStruct b3;
  Val neg;
};
struct SplitTotalStruct {
  NondetRegStruct out;
  Val carry;
};
struct MultiplySettingsStruct {
  Val aSigned;
  Val bSigned;
  Val cSigned;
};
struct MultiplyAccumulateStruct {
  ValU32Struct outLow;
  ValU32Struct outHigh;
};
struct DivInputStruct {
  InstInputStruct _super;
  InstInputStruct ii;
  DecoderStruct decoded;
  GetDataStruct rs1;
  GetDataStruct rs2;
};
struct DoDivStruct {
  ValU32Struct quot;
  ValU32Struct rem;
};
struct InstOutputStruct {
  ValU32Struct newPc;
  Val newState;
  Val newMode;
};
struct MiscInputStruct {
  InstInputStruct _super;
  InstInputStruct ii;
  DecoderStruct decoded;
  GetDataStruct rs1;
  GetDataStruct rs2;
};
struct MiscOutputStruct {
  Val doWrite;
  DenormedValU32Struct toWrite;
  DenormedValU32Struct newPc;
};
struct MulInputStruct {
  InstInputStruct _super;
  InstInputStruct ii;
  DecoderStruct decoded;
  GetDataStruct rs1;
  GetDataStruct rs2;
};
struct DoMulStruct {
  ValU32Struct low;
  ValU32Struct high;
};
struct MemLoadInputStruct {
  InstInputStruct ii;
  DecoderStruct decoded;
  AddrDecomposeBitsStruct addr;
  GetDataStruct data;
};
struct MemStoreInputStruct {
  DecoderStruct decoded;
  GetDataStruct rs2;
  AddrDecomposeBitsStruct addr;
  GetDataStruct data;
};
struct MemStoreFinalizeStruct {
};
struct SplitWordStruct {
  NondetRegStruct byte0;
  NondetRegStruct byte1;
};
struct DigestRegValues_SuperStruct {
  NondetRegStruct low;
  NondetRegStruct high;
};
using DigestRegValues_SuperStruct8Array = DigestRegValues_SuperStruct[8];
struct DigestRegStruct {
  DigestRegValues_SuperStruct8Array values;
};
using ValU32Struct8Array = ValU32Struct[8];
struct ControlLoadRoot__0Struct {
};
using ControlLoadRoot__0Struct8Array = ControlLoadRoot__0Struct[8];
struct ControlResumeArm1_Super__0Struct {
};
using ControlResumeArm1_Super__0Struct8Array = ControlResumeArm1_Super__0Struct[8];
struct ComponentStruct {
};
using GetDataStruct8Array = GetDataStruct[8];
struct ControlTableArm0_Super__0Struct {
};
struct ControlTableArm1_Super__0Struct {
};
using ControlTableArm0_Super__0Struct16Array = ControlTableArm0_Super__0Struct[16];
using ControlTableArm1_Super__0Struct16Array = ControlTableArm1_Super__0Struct[16];
using Val4Array = Val[4];
using NondetRegStruct4Array = NondetRegStruct[4];
struct OneHot_4_Struct {
  NondetRegStruct4Array _super;
};
struct ECallOutputStruct {
  Val state;
  Val s0;
  Val s1;
  Val s2;
};
struct DecomposeLow2Struct {
  NondetRegStruct high;
  NondetRegStruct low2;
  OneHot_4_Struct low2Hot;
  NondetRegStruct highZero;
  NondetRegStruct isZero;
  Val low2Nonzero;
};
struct ECallHostReadWords__0Struct {
};
using ECallHostReadWords__0Struct4Array = ECallHostReadWords__0Struct[4];
using Val24Array = Val[24];
struct MultiplyByMInt_Super_SuperStruct {
  Val _super;
};
using MultiplyByMInt_Super_SuperStruct24Array = MultiplyByMInt_Super_SuperStruct[24];
struct MultiplyByMIntStruct {
  MultiplyByMInt_Super_SuperStruct24Array _super;
};
struct DoIntRounds__0_SuperStruct {
  Val _super;
};
using DoIntRounds__0_SuperStruct21Array = DoIntRounds__0_SuperStruct[21];
struct DoIntRoundsStruct {
  Val24Array _super;
};
using NondetRegStruct24Array = NondetRegStruct[24];
struct MultiplyByMExt_Super_SuperStruct {
  Val _super;
};
using MultiplyByMExt_Super_SuperStruct24Array = MultiplyByMExt_Super_SuperStruct[24];
struct MultiplyByMExtStruct {
  MultiplyByMExt_Super_SuperStruct24Array _super;
};
struct PoseidonStateStruct {
  NondetRegStruct hasState;
  NondetRegStruct stateAddr;
  NondetRegStruct bufOutAddr;
  NondetRegStruct isElem;
  NondetRegStruct checkOut;
  NondetRegStruct loadTxType;
  NondetRegStruct nextState;
  NondetRegStruct subState;
  NondetRegStruct bufInAddr;
  NondetRegStruct count;
  NondetRegStruct mode;
  NondetRegStruct24Array inner;
  NondetExtRegStruct zcheck;
};
struct PoseidonOpDefStruct {
  Val hasState;
  Val stateAddr;
  Val bufOutAddr;
  Val isElem;
  Val checkOut;
  Val loadTxType;
};
struct ReadAddrStruct {
  Val _super;
};
struct ReadElemStruct {
  Val _super;
};
using ReadElemStruct8Array = ReadElemStruct[8];
struct PoseidonCheckOut__0Struct {
};
using PoseidonCheckOut__0Struct8Array = PoseidonCheckOut__0Struct[8];
struct PoseidonStoreOut__0Struct {
};
using PoseidonStoreOut__0Struct8Array = PoseidonStoreOut__0Struct[8];
struct PoseidonStoreState__0Struct {
};
using PoseidonStoreState__0Struct8Array = PoseidonStoreState__0Struct[8];
struct IsU24Struct {
};
using Val6Array = Val[6];
using NondetRegStruct6Array = NondetRegStruct[6];
struct OneHot_6_Struct {
  NondetRegStruct6Array _super;
  NondetRegStruct6Array bits;
};
using Val11Array = Val[11];
using NondetRegStruct11Array = NondetRegStruct[11];
struct OneHot_11_Struct {
  NondetRegStruct11Array _super;
};
struct TopStruct {
};
