struct NondetRegLayout {
  Reg _super;
};
using NondetRegLayout12LayoutArray = NondetRegLayout[12];
struct WrapOneHotLayout {
  NondetRegLayout12LayoutArray _super;
};
struct ControlStateLayout {
  NondetRegLayout cycleType;
  NondetRegLayout subType;
  NondetRegLayout block;
  NondetRegLayout round;
};
struct ComputeCurrentStep_SuperArm2_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
};
struct KeccackNextRound_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
};
struct IsZeroLayout {
  NondetRegLayout _super;
  NondetRegLayout inv;
};
struct KeccackNextRoundLayout {
  KeccackNextRound_SuperLayout _super;
  IsZeroLayout isLast;
};
struct ShaNextRound_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
};
struct ShaNextRoundLayout {
  ShaNextRound_SuperLayout _super;
  IsZeroLayout isLast;
};
struct ShaNextBlock_SuperArm2_Super_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
};
struct ShaNextBlock_SuperArm2_SuperLayout {
  ShaNextBlock_SuperArm2_Super_SuperLayout _super;
  NondetRegLayout more;
};
struct ShaNextBlock_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
  ShaNextBlock_SuperArm2_SuperLayout arm2;
};
struct ShaNextBlockLayout {
  ShaNextBlock_SuperLayout _super;
  IsZeroLayout isLast;
  NondetRegLayout isSub0;
};
struct ComputeCurrentStep_SuperLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ControlStateLayout arm1;
  ComputeCurrentStep_SuperArm2_SuperLayout arm2;
  ControlStateLayout arm3;
  ControlStateLayout arm4;
  ControlStateLayout arm5;
  ControlStateLayout arm6;
  ControlStateLayout arm7;
  KeccackNextRoundLayout arm8;
  ShaNextRoundLayout arm9;
  ShaNextBlockLayout arm10;
  ControlStateLayout arm11;
};
struct TopControlStateLayout {
  ControlStateLayout _super;
  ControlStateLayout arm0;
  ComputeCurrentStep_SuperLayout arm1;
};
struct OneHot_12_Layout {
  NondetRegLayout12LayoutArray _super;
};
using NondetRegLayout800LayoutArray = NondetRegLayout[800];
using NondetRegLayout100LayoutArray = NondetRegLayout[100];
using NondetRegLayout16LayoutArray = NondetRegLayout[16];
struct TopStateLayout {
  NondetRegLayout800LayoutArray bits;
  NondetRegLayout100LayoutArray kflat;
  NondetRegLayout16LayoutArray sflat;
};
struct UnpackReg_800__16_Layout {
  NondetRegLayout800LayoutArray _super;
};
struct ReadCycleLayout {
  TopStateLayout _super;
  NondetRegLayout100LayoutArray load;
  UnpackReg_800__16_Layout bits;
};
struct ExpandCycleLayout {
  TopStateLayout _super;
  UnpackReg_800__16_Layout bits;
};
struct WriteCycleLayout {
  TopStateLayout _super;
  UnpackReg_800__16_Layout bits;
};
using NondetRegLayout64LayoutArray = NondetRegLayout[64];
struct ThetaP1_Super_SuperLayout {
  NondetRegLayout64LayoutArray _super;
};
using ThetaP1_Super_SuperLayout5LayoutArray = ThetaP1_Super_SuperLayout[5];
struct ThetaP1Layout {
  ThetaP1_Super_SuperLayout5LayoutArray _super;
};
struct KeccakRound0Layout {
  TopStateLayout _super;
  ThetaP1Layout b;
};
using NondetRegLayout24LayoutArray = NondetRegLayout[24];
struct OneHot_24_Layout {
  NondetRegLayout24LayoutArray _super;
};
struct RoundToArrayLayout {
  OneHot_24_Layout split;
};
struct IotaLayout {
  RoundToArrayLayout iotaArray;
};
struct KeccakRound34_0_Layout {
  TopStateLayout _super;
  IotaLayout _0;
};
struct KeccakRound34_1_Layout {
  TopStateLayout _super;
  IotaLayout _0;
};
using NondetRegLayout8LayoutArray = NondetRegLayout[8];
struct OneHot_8_Layout {
  NondetRegLayout8LayoutArray _super;
};
using NondetRegLayout4LayoutArray = NondetRegLayout[4];
struct OneHot_4_Layout {
  NondetRegLayout4LayoutArray _super;
};
struct LoadWinLayout {
  OneHot_4_Layout blockSelect;
};
using NondetRegLayout32LayoutArray = NondetRegLayout[32];
struct UnpackReg_32__16_Layout {
  NondetRegLayout32LayoutArray _super;
};
struct CarryExtractLayout {
  NondetRegLayout bit0;
  NondetRegLayout bit1;
  NondetRegLayout bit2;
};
struct CarryAndExpandLayout {
  UnpackReg_32__16_Layout _super;
  CarryExtractLayout lowCarry;
  CarryExtractLayout highCarry;
};
struct DoShaStepLayout {
  CarryAndExpandLayout w;
  CarryAndExpandLayout a;
  CarryAndExpandLayout e;
};
using DoShaStepLayout8LayoutArray = DoShaStepLayout[8];
struct ShaCycleLayout {
  TopStateLayout _super;
  OneHot_8_Layout oneHot;
  NondetRegLayout isLoad;
  LoadWinLayout win;
  DoShaStepLayout step0;
  DoShaStepLayout step1;
  DoShaStepLayout step2;
  DoShaStepLayout step3;
  DoShaStepLayout step4;
  DoShaStepLayout step5;
  DoShaStepLayout step6;
  DoShaStepLayout step7;
  DoShaStepLayout8LayoutArray steps;
};
using CarryAndExpandLayout4LayoutArray = CarryAndExpandLayout[4];
struct ShaNextBlockCycleLayout {
  TopStateLayout _super;
  CarryAndExpandLayout4LayoutArray totA;
  CarryAndExpandLayout4LayoutArray totE;
};
struct TopCurStateLayout {
  TopStateLayout _super;
  TopStateLayout arm0;
  ReadCycleLayout arm1;
  ExpandCycleLayout arm2;
  WriteCycleLayout arm3;
  KeccakRound0Layout arm4;
  TopStateLayout arm5;
  TopStateLayout arm6;
  KeccakRound34_0_Layout arm7;
  KeccakRound34_1_Layout arm8;
  ShaCycleLayout arm9;
  ShaNextBlockCycleLayout arm10;
  TopStateLayout arm11;
};
struct DigestRegLayout {
  NondetRegLayout16LayoutArray values;
};
struct TopLayout {
  NondetRegLayout isFirst;
  WrapOneHotLayout cycleMux;
  ControlStateLayout controlState;
  TopControlStateLayout controlStateRedef;
  OneHot_12_Layout oneHot;
  TopStateLayout curState;
  TopCurStateLayout curStateRedef;
};
struct _globalLayout {
  DigestRegLayout finalDigest;
};
struct NondetRegStruct {
  Val _super;
};
using Val5Array = Val[5];
using Val64Array = Val[64];
using Val64Array5Array = Val64Array[5];
using Val64Array5Array5Array = Val64Array5Array[5];
struct ThetaP1_Super_Super_SuperStruct {
  NondetRegStruct _super;
};
using ThetaP1_Super_Super_SuperStruct64Array = ThetaP1_Super_Super_SuperStruct[64];
struct ThetaP1_Super_SuperStruct {
  ThetaP1_Super_Super_SuperStruct64Array _super;
};
using ThetaP1_Super_SuperStruct5Array = ThetaP1_Super_SuperStruct[5];
struct ThetaP1Struct {
  ThetaP1_Super_SuperStruct5Array _super;
};
using Val24Array = Val[24];
using NondetRegStruct24Array = NondetRegStruct[24];
struct OneHot_24_Struct {
  NondetRegStruct24Array _super;
};
struct RoundToArray_Super_SuperStruct {
  Val _super;
};
using RoundToArray_Super_SuperStruct64Array = RoundToArray_Super_SuperStruct[64];
struct RoundToArrayStruct {
  RoundToArray_Super_SuperStruct64Array _super;
};
struct Iota_Super_Super_Super_SuperStruct {
  Val64Array _super;
};
using Iota_Super_Super_Super_SuperStruct5Array = Iota_Super_Super_Super_SuperStruct[5];
struct Iota_Super_SuperStruct {
  Iota_Super_Super_Super_SuperStruct5Array _super;
};
using Iota_Super_SuperStruct5Array = Iota_Super_SuperStruct[5];
struct IotaStruct {
  Iota_Super_SuperStruct5Array _super;
};
struct CarryExtractStruct {
  Val carry;
  Val out;
};
using Val2Array = Val[2];
struct DivStruct {
  Val _super;
};
using DivStruct32Array = DivStruct[32];
using Val32Array = Val[32];
struct UnpackReg_32__16__SuperStruct {
  NondetRegStruct _super;
};
using UnpackReg_32__16__SuperStruct32Array = UnpackReg_32__16__SuperStruct[32];
struct UnpackReg_32__16_Struct {
  UnpackReg_32__16__SuperStruct32Array _super;
};
using Val800Array = Val[800];
struct TopStateBitsStruct {
  NondetRegStruct _super;
};
using TopStateBitsStruct800Array = TopStateBitsStruct[800];
using Val100Array = Val[100];
struct TopStateKflatStruct {
  NondetRegStruct _super;
};
using TopStateKflatStruct100Array = TopStateKflatStruct[100];
using Val16Array = Val[16];
using NondetRegStruct16Array = NondetRegStruct[16];
struct TopStateStruct {
  TopStateBitsStruct800Array bits;
  TopStateKflatStruct100Array kflat;
  NondetRegStruct16Array sflat;
};
using Val8Array = Val[8];
using NondetRegStruct8Array = NondetRegStruct[8];
struct OneHot_8_Struct {
  NondetRegStruct8Array _super;
};
using Val32Array4Array = Val32Array[4];
using Val32Array16Array = Val32Array[16];
struct ShaStateStruct {
  Val32Array4Array a;
  Val32Array4Array e;
  Val32Array16Array w;
};
struct DoShaStepStruct {
  UnpackReg_32__16_Struct w;
  UnpackReg_32__16_Struct a;
  UnpackReg_32__16_Struct e;
  ShaStateStruct newState;
};
using Val4Array = Val[4];
using NondetRegStruct4Array = NondetRegStruct[4];
struct OneHot_4_Struct {
  NondetRegStruct4Array _super;
};
using Val2Array8Array = Val2Array[8];
struct LoadWinStruct {
  Val2Array8Array _super;
};
struct Pack_32__16__Super_SuperStruct {
  Val _super;
};
using Pack_32__16__Super_SuperStruct2Array = Pack_32__16__Super_SuperStruct[2];
struct Pack_32__16_Struct {
  Pack_32__16__Super_SuperStruct2Array _super;
};
using Pack_32__16_Struct4Array = Pack_32__16_Struct[4];
using NondetRegStruct2Array = NondetRegStruct[2];
struct ShaNextBlockCyclePrevA_SuperStruct {
  NondetRegStruct2Array _super;
};
using ShaNextBlockCyclePrevA_SuperStruct4Array = ShaNextBlockCyclePrevA_SuperStruct[4];
struct ShaNextBlockCyclePrevE_SuperStruct {
  NondetRegStruct2Array _super;
};
using ShaNextBlockCyclePrevE_SuperStruct4Array = ShaNextBlockCyclePrevE_SuperStruct[4];
using UnpackReg_32__16_Struct4Array = UnpackReg_32__16_Struct[4];
using Val50Array = Val[50];
using DivStruct800Array = DivStruct[800];
struct UnpackReg_800__16__SuperStruct {
  NondetRegStruct _super;
};
using UnpackReg_800__16__SuperStruct800Array = UnpackReg_800__16__SuperStruct[800];
struct UnpackReg_800__16_Struct {
  UnpackReg_800__16__SuperStruct800Array _super;
};
struct ReadCycleLoadStruct {
  NondetRegStruct _super;
};
using ReadCycleLoadStruct100Array = ReadCycleLoadStruct[100];
using TopStateKflatStruct50Array = TopStateKflatStruct[50];
struct DigestRegValuesStruct {
};
using DigestRegValuesStruct16Array = DigestRegValuesStruct[16];
struct DigestRegStruct {
};
struct ControlStateStruct {
  NondetRegStruct cycleType;
  NondetRegStruct subType;
  NondetRegStruct block;
  NondetRegStruct round;
};
using Val12Array = Val[12];
struct OneHot_12__SuperStruct {
  NondetRegStruct _super;
};
using OneHot_12__SuperStruct12Array = OneHot_12__SuperStruct[12];
struct OneHot_12_Struct {
  OneHot_12__SuperStruct12Array bits;
};
struct WrapOneHot_SuperStruct {
  NondetRegStruct _super;
};
using WrapOneHot_SuperStruct12Array = WrapOneHot_SuperStruct[12];
struct WrapOneHotStruct {
  WrapOneHot_SuperStruct12Array _super;
};
struct TopStruct {
};
