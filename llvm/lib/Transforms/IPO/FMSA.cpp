
#include "llvm/Transforms/IPO/FMSA.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include <llvm/IR/IRBuilder.h>

#include "llvm/Support/Error.h"
#include "llvm/Support/Timer.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Analysis/LoopInfo.h"
//#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include "llvm/Support/RandomNumberGenerator.h"

//#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Analysis/Utils/Local.h"
#include "llvm/Transforms/Utils/Local.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/FunctionComparator.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <list>

#include <limits.h>

#include <functional>
#include <queue>
#include <vector>

#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"

#define DEBUG_TYPE "MyFuncMerge"
//#define ENABLE_DEBUG_CODE

//#define FMSA_USE_JACCARD

//#define TIME_STEPS_DEBUG

using namespace llvm;

/* REQUIRES MERGED RETURN FUNCTIONS. */

struct MergedBBDynProf{
	std::vector<uint32_t> BBWeights;
	bool nonAligned;

	MergedBBDynProf(std::vector<uint32_t> BBWeightsPar, bool nonAlignedPar) : 
    	BBWeights(BBWeightsPar), nonAligned(nonAlignedPar) {}

};

class NeedlemanWunschSimilarityMatrix {
public:
  SmallVectorImpl<Value *> &F1;
  SmallVectorImpl<Value *> &F2;

  bool *Match;
  int *Matrix;
  unsigned NumRows;
  unsigned NumCols;
  unsigned MaxScore;
  unsigned MaxRow;
  unsigned MaxCol;

  const static unsigned END = 0;
  const static unsigned DIAGONAL = 1;
  const static unsigned UP = 2;
  const static unsigned LEFT = 3;

  // scoring scheme
  const int matchAward = 2;
  const int mismatchPenalty = -1;
  const int gapPenalty = -1;

  NeedlemanWunschSimilarityMatrix(SmallVectorImpl<Value *> &F1,
                                  SmallVectorImpl<Value *> &F2);

  ~NeedlemanWunschSimilarityMatrix();

  int calcScore(unsigned i, unsigned j);

  unsigned nextMove(unsigned i, unsigned j);
};

static bool matchIntrinsicCalls(Intrinsic::ID ID, const CallInst *CI1,
                                const CallInst *CI2);

static bool matchLandingPad(LandingPadInst *LP1, LandingPadInst *LP2);

static bool match(Instruction *I1, Instruction *I2);

static bool match(Value *V1, Value *V2);

static bool match(std::vector<Value *> &F1, std::vector<Value *> &F2,
                  unsigned i, unsigned j);

static bool match(SmallVectorImpl<Value *> &F1, SmallVectorImpl<Value *> &F2,
                  unsigned i, unsigned j);

std::map< std::string, double > FMSALegacyPass::bbProfiles;
std::map< std::string, double > FMSALegacyPass::opLuts;
std::set< std::string> FMSALegacyPass::BlackList;

std::map< std::string, double>  FMSALegacyPass::realAreaPerFunc;
std::map< BasicBlock *, double > FMSALegacyPass::mergedBBProfiles;
std::map< Function *, double > FMSALegacyPass::funcTimeSize;
//std::map<BasicBlock *, MergedBBDynProf> MergingNativeProfiling;
std::map<BasicBlock *, std::vector<uint32_t> > MergingNativeProfiling;
std::vector< std::pair<double, double> > codeAndTimeReduction;
std::set<std::string> mergedFunctions;
std::set< std::string > functionsWithEstimatedFuncSize;
std::map< std::string, double > global_num_calls; //function_name -> function_name -> double

static int merged_func_id = 0;

static cl::opt<std::string>
DynamicProfileFile("dyn-prof-file", cl::value_desc("filename"),
          cl::desc("A file containing the dynamic profile of the module being inspected"),
          cl::Hidden);


static cl::opt<std::string>
AreaModelFile("area-model-file", cl::value_desc("filename"),
          cl::desc("Load area cost model"),
          cl::Hidden);


static cl::opt<bool>
FuncConcatMode("function-concat",  cl::init(false), cl::Hidden,
          cl::desc("Just concatenate functions and leave merging to the HLS tool/design compiler."));



static cl::opt<std::string>
RealAreaFile("real-area-file", cl::value_desc("filename"),
          cl::desc("A cache with the real area for each function. Functions are merged"),
          cl::Hidden);


static cl::opt<std::string>
BlacklistFile("blacklist-file", cl::value_desc("filename"),
          cl::desc("A list of function names to not use for merging"),
          cl::Hidden);


static cl::opt<std::string>
BlacklistedFunc("blacklisted-func", cl::value_desc("Function to not merge until any circumstance"),
          cl::desc("Function to not merge under any scenario"),
          cl::Hidden);

static cl::opt<unsigned> ExplorationThreshold(
    "fmsa-explore", cl::init(10), cl::Hidden,
    cl::desc("Exploration threshold of evaluated functions"));

static cl::opt<int> MergingOverheadThreshold(
    "fmsa-threshold", cl::init(0), cl::Hidden,
    cl::desc("Threshold of allowed overhead for merging function"));


static cl::opt<double> LoadImbalanceThreshold( //orders of magnitude of differenence are tollerable
    "fmsa-li", cl::init(999999.0), cl::Hidden,
    cl::desc("Threshold of allowed load imbalance to merge"));

static cl::opt<bool>
    MaxParamScore("fmsa-max-param", cl::init(true), cl::Hidden,
                  cl::desc("Maximizing the score for merging parameters"));


//static cl::opt<double> AreaWinToSlowdownRatio("area-win-to-slowdown-ratio", cl::init(999999), cl::Hidden,

static cl::opt<double> AreaWinToSlowdownRatio("area-win-to-slowdown-ratio", cl::init(0), cl::Hidden,
                                   cl::desc("Define the ratio between the area improvement and the performance slowdown that can be tolerated"));

static cl::opt<bool> LatencyMerges("fmsa-latency-merges", cl::init(false), cl::Hidden,
                           cl::desc("Choose to do merges based on latency so that the critical and repetitive structures have minimum ovh"));

static cl::opt<bool> Debug("fmsa-debug", cl::init(false), cl::Hidden,
                           cl::desc("Outputs debug information"));

static cl::opt<bool> Verbose("fmsa-verbose", cl::init(false),
                             cl::Hidden, cl::desc("Outputs debug information"));

static cl::opt<bool>
    IdenticalType("fmsa-identic-type", cl::init(true), cl::Hidden,
                  cl::desc("Maximizing the score for merging parameters"));

static cl::opt<bool> ApplySimilarityHeuristic(
    "fmsa-similarity-pruning", cl::init(true), cl::Hidden,
    cl::desc("Maximizing the score for merging parameters"));


static cl::opt<bool>
    HasWholeProgram("fmsa-whole-program", cl::init(true), cl::Hidden,
                  cl::desc("Function merging applied on whole program"));

static cl::opt<bool>
    RunBruteForceExploration("fmsa-oracle", cl::init(false), cl::Hidden,
                  cl::desc("Run function merging's oracle"));


static cl::opt<bool>
    UseRandomShufflings("use-rand-shufflings", cl::init(false), cl::Hidden,
                  cl::desc("Use random shufflings for the linearization step"));


// static std::unique_ptr<RandomNumberGenerator> RandGen;

static std::list<unsigned> MergingDistance;

static unsigned LastMaxParamScore = 0;
static unsigned TotalParamScore = 0;

static int CountOpReorder = 0;
static int CountBinOps = 0;

static std::string GetValueName(const Value *V) {
  if (V) {
    std::string name;
    raw_string_ostream namestream(name);
    V->printAsOperand(namestream, false);
    return namestream.str();
  } else
    return "[null]";
}


static float getCycleSWDelayEstim(Instruction *Inst)    
   
  {

    switch (Inst->getOpcode()) {

    case Instruction::GetElementPtr: 
      return 0;

    case Instruction::Br:
      return 1;

    case Instruction::Alloca:
      return 1;

    case Instruction::PHI:
      return 1;

    case Instruction::Store:
      return 1;

    case Instruction::Load:
      return 1;

    case Instruction::Call:
      return 1;

    case Instruction::Fence:
      return 1;

    case Instruction::LandingPad:
      return 1;

    case Instruction::AtomicCmpXchg:
      return 1;

    case Instruction::AtomicRMW: 
      return 1;

    case Instruction::ExtractValue: 
      return 1;

    case Instruction::InsertValue:
      return 1;

    case Instruction::Switch: // Similar as branching.
      return 1;
    
    case Instruction::IndirectBr: 
      return 1;

    case Instruction::Invoke: 
      return 1;

    case Instruction::Resume: 
      return 1;

    case Instruction::Unreachable: 
      return 0;

    case Instruction::Ret:
      return 1;

    case Instruction::ShuffleVector: // * more complex *
      return 1;

    case Instruction::ExtractElement:
      return 1;

    case Instruction::InsertElement:
      return 1;

    case Instruction::Add:
      return 1;

    case Instruction::FAdd:
      return 2;

    case Instruction::Sub:
      return 1;

    case Instruction::FSub:  // This is not modelled properly (Double as rough estimation)
      return 2;

    case Instruction::Mul:
      return 1;

    case Instruction::FMul: // This is not modelled properly (Double as rough estimation)

    case Instruction::UDiv:
      return 6;

    case Instruction::SDiv:
      return 6;

    case Instruction::FDiv: // This is not modelled properly (Double as rough estimation)
      return 12;

    case Instruction::URem:
      return 6;

    case Instruction::SRem:
      return 6;

    case Instruction::FRem: // This is not modelled properly (Double as rough estimation)
      return 12;

    case Instruction::Shl: {

      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;

      return 1;
    
    }
    
    case Instruction::LShr: {
      
      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;
      
      return 1;
    }

    case Instruction::AShr:{
      
      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;
      
      return 1;
    }

    case Instruction::And:
      return 1;

    case Instruction::Or:
      return 1;

    case Instruction::Xor:
      return 1;

    case Instruction::Select: // This is my estimation!
      return 1;

    case Instruction::ICmp: // Check type of ICmp. (Equality or Relational) 
    {
      ICmpInst *Icmp = dyn_cast<ICmpInst>(&*Inst);

      if (Icmp->isEquality())
        return 1;
      else
        return 1;       
    }

    case Instruction::FCmp: // This is my estimation!
      return 1;

    case Instruction::ZExt:
      return 0;

    case Instruction::SExt:
      return 0;

    case Instruction::FPToUI:
      return 0;

    case Instruction::FPToSI:
      return 0;

    case Instruction::FPExt:
      return 0;

    case Instruction::PtrToInt:
      return 0;

    case Instruction::IntToPtr:
      return 0;

    case Instruction::SIToFP:
      return 0;

    case Instruction::UIToFP:
      return 0;

    case Instruction::Trunc:
      return 0;

    case Instruction::FPTrunc:
      return 0;

    case Instruction::BitCast:
      return 0;

    default:  // Assume default as 0.
      return 0;

    }// end of switch.
  }


static  float getHWDelayEstim(Instruction *Inst)
  
  {

    switch (Inst->getOpcode()) {

    case Instruction::GetElementPtr: 
      return 0;

    case Instruction::Br:
      return 0;

    case Instruction::Alloca:
      return 0;

    case Instruction::PHI:
    // #ifdef OLD_SETUP
    //   return 5.299;
    // #endif
    //#ifdef SYS_AWARE
    //  return 4.3;
    //#else
      return 0.23;
    //#endif
    return 0;

    case Instruction::Store:
      return 0;

    case Instruction::Load:
      return 0;

    case Instruction::Call:
      return 0;

    case Instruction::Fence:
      return 0;

    case Instruction::LandingPad:
      return 0;

    case Instruction::AtomicCmpXchg:
      return 0;

    case Instruction::AtomicRMW: 
      return 0;

    case Instruction::ExtractValue: 
      return 0;

    case Instruction::InsertValue:
      return 0;

    case Instruction::Switch: // ceil(log2(Number of Cases)) * 5.2999
    {

      SwitchInst *Switch = dyn_cast<SwitchInst>(&*Inst);
      unsigned int NumCases = Switch->getNumCases();
      // #ifdef OLD_SETUP
      //   return ceil(log2(NumCases)) * 5.2999 ;
      // #endif
      #ifdef SYS_AWARE
        return ceil(log2(NumCases)) * 4.3 ;
      #else
      return ceil(log2(NumCases)) * 0.23 ;
      #endif
    }
      
    case Instruction::IndirectBr: 
      return 0;

    case Instruction::Invoke: 
      return 0;

    case Instruction::Resume: 
      return 0;

    case Instruction::Unreachable: 
      return 0;

    case Instruction::Ret:
      return 0;

    case Instruction::ShuffleVector:
      return 0;

    case Instruction::ExtractElement:
      return 0;

    case Instruction::InsertElement:
      return 0;


    case Instruction::Add:
      // #ifdef OLD_SETUP
      //   return 9.323;
      // #endif
      #ifdef SYS_AWARE
        return 5.3;
      #else
      return 0.92;
      #endif

    case Instruction::FAdd: // This is not modelled properly. Or mark as forbidden.
      // #ifdef OLD_SETUP
      //   return 9.323;
      // #endif
      #ifdef SYS_AWARE
        return 5.3;
      #else
      return 0.92;
      #endif

    case Instruction::Sub:
      // #ifdef OLD_SETUP
      //   return 9.323;
      // #endif
      #ifdef SYS_AWARE
        return 5.3;
      #else
      return 0.92;
      #endif

    case Instruction::FSub: // This is not modelled properly. Or mark as forbidden.
      // #ifdef OLD_SETUP
      //   return 9.323;
      // #endif
      #ifdef SYS_AWARE
        return 5.3;
      #else
      return 0.92;
      #endif

    case Instruction::Mul:
    // #ifdef OLD_SETUP
    //   return 9.9;
    // #endif
      #ifdef SYS_AWARE
        return 8.5;
      #else
        return 1;
      #endif

    case Instruction::FMul:  // This is not modelled proerly. Or mark as forbidden.
      // #ifdef OLD_SETUP
      //   return 9.9;
      // #endif
      #ifdef SYS_AWARE
        return 8.5;
      #else
      return 1;
      #endif

    case Instruction::UDiv:
      // #ifdef OLD_SETUP
      //   return 59.535;
      // #endif
      #ifdef SYS_AWARE
        return 49.5;
      #else
      return 3.76;
      #endif
    
    case Instruction::SDiv:
      // #ifdef OLD_SETUP
      //   return 59.7;
      // #endif
      #ifdef SYS_AWARE
        return 53;
      #else
      return 3.76;
      #endif

    case Instruction::FDiv:
      // #ifdef OLD_SETUP
      //   return 59.7;
      // #endif
      #ifdef SYS_AWARE
        return 53;
      #else
      return 3.76;
      #endif

    case Instruction::URem:
      // #ifdef OLD_SETUP
      //   return 59.636;
      // #endif
      #ifdef SYS_AWARE
        return 52.6;
      #else
      return 4.04;
      #endif

    case Instruction::SRem:
      // #ifdef OLD_SETUP
      //   return 59.636;
      // #endif
      #ifdef SYS_AWARE
        return 55.4;
      #else
      return 4.04;
      #endif

    case Instruction::FRem:
      // #ifdef OLD_SETUP
      //   return 59.636;
      // #endif
      #ifdef SYS_AWARE
        return 55.4;
      #else
      return 4.04;
      #endif

    case Instruction::Shl:{

      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;

      // #ifdef OLD_SETUP
      //   return 8.862;
      // #endif
      #ifdef SYS_AWARE
        return 5.5;
      #else
      return 0.71;
      #endif
    
    }
    
    case Instruction::LShr: {
      
      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;
      
      // #ifdef OLD_SETUP
      //   return 8.862;
      // #endif
      #ifdef SYS_AWARE
        return 5.5;
      #else
      return 0.73;
      #endif
    }

    case Instruction::AShr:{
      
      Value *Operand_two = Inst->getOperand(1);

      if (Operand_two->getType()->isSingleValueType()) // Shift by a single value (e.g. integer)
        return 0;
      
      // #ifdef OLD_SETUP
      //   return 9.052;
      // #endif
      #ifdef SYS_AWARE
        return 6.6;
      #else
      return 0.65;
      #endif
    }

    case Instruction::And:
      // #ifdef OLD_SETUP
      //   return 8.862;
      // #endif
      #ifdef SYS_AWARE
        return 4.3;
      #else
      return 0.02;
      #endif

    case Instruction::Or:
      // #ifdef OLD_SETUP
      //   return 8.862;
      // #endif
      #ifdef SYS_AWARE
        return 4.3;
      #else
      return 0.03;
      #endif

    case Instruction::Xor:
      // #ifdef OLD_SETUP
      //   return 8.862;
      // #endif
      #ifdef SYS_AWARE
        return 4.3;
      #else
      return 0.03;
      #endif

    case Instruction::Select: // Same as Phi.
      // #ifdef OLD_SETUP
      //   return 5.299;
      // #endif
      
      //#ifdef SYS_AWARE //iulian's old version
      //  return 4.3;
      //#else
      return 0.23;
      //#endif
      return 0; //iulian's new version
    case Instruction::ICmp: // Check type of ICmp. (Equality or Relational) 
    {
      ICmpInst *Icmp = dyn_cast<ICmpInst>(&*Inst);

      if (Icmp->isEquality())
        // #ifdef OLD_SETUP
        //   return 9.276;
        // #endif
        #ifdef SYS_AWARE
          return 5;
        #else
        return 0.15;
        #endif
      else
        // #ifdef OLD_SETUP
        //   return 9.389;
        // #endif
       //#ifdef SYS_AWARE
      //    return 4.8;
       return 0;
        //#else
        //  return 0.37;
        //#endif      
    }

    case Instruction::FCmp: // This is my estimation!
      // #ifdef OLD_SETUP
      //   return 9.276;
      // #endif
      #ifdef SYS_AWARE
        return 5;
      #else
      return 0.15;
      #endif

    case Instruction::ZExt:
      return 0;

    case Instruction::SExt:
      return 0;

    case Instruction::FPToUI:
      return 0;

    case Instruction::FPToSI:
      return 0;

    case Instruction::FPExt:
      return 0;

    case Instruction::PtrToInt:
      return 0;

    case Instruction::IntToPtr:
      return 0;

    case Instruction::SIToFP:
      return 0;

    case Instruction::UIToFP:
      return 0;

    case Instruction::Trunc:
      return 0;

    case Instruction::FPTrunc:
      return 0;

    case Instruction::BitCast:
      return 0;

    default:  // Assume default as 0 LUTs in Area estimation.
      return 0;

    }// end of switch.
  }

static unsigned int getAreaEstimOld(Instruction *Inst) {
    if (FMSALegacyPass::opLuts.find(Inst->getOpcodeName()) == FMSALegacyPass::opLuts.end())
        errs() << "For some reason the opcode " << Inst->getOpcode() << " could not be found\n";
    return FMSALegacyPass::opLuts[Inst->getOpcodeName()];
}
  //Area Estimation for each DFG Node/Istruction in LUTs
// RegionSeeker  
static unsigned int getAreaEstim(Instruction *Inst) {
    //if (FMSALegacyPass::opLuts.find(Inst->getOpcodeName()) != FMSALegacyPass::opLuts.end())
    //    return FMSALegacyPass::opLuts[Inst->getOpcodeName()];
    //else {
        //errs() << "For some reason the opcode " << Inst->getOpcode() << " could not be found\n";
        switch (Inst->getOpcode()) {

        case Instruction::GetElementPtr: 
          return 0;//modified iuli

        case Instruction::Br: //this should add predictation bits of some sort.
          return 0;

        case Instruction::Alloca:
          return 0;

        case Instruction::PHI:
          //#ifdef SYS_AWARE
          //  //return 1;
          //  return 16;
          //#else
          //  return 33;
            //return 1;
          //#endif
          return 0;

        case Instruction::Store:
          return 0;

        case Instruction::Load:
          return 0;

        case Instruction::Call:
          return 0;

        case Instruction::Fence:
          return 0;

        case Instruction::LandingPad:
          return 0;

        case Instruction::AtomicCmpXchg:
          return 0;

        case Instruction::AtomicRMW: 
          return 0;

        case Instruction::ExtractValue: 
          return 0;

        case Instruction::InsertValue:
          return 0;

        case Instruction::Switch: // 33 * N (Number of Cases)
        {

          SwitchInst *Switch = dyn_cast<SwitchInst>(&*Inst);
          //errs() << " Number of switches is  " << Switch->getNumCases() << "\t";
          unsigned int NumCases = Switch->getNumCases();
          #ifdef SYS_AWARE
            return 16 * NumCases;
          #else

            return 33 * NumCases;
          #endif
        }
          
        case Instruction::IndirectBr: 
          return 0;

        case Instruction::Invoke: 
          return 0;

        case Instruction::Resume: 
          return 0;

        case Instruction::Unreachable: 
          return 0;

        case Instruction::Ret:
          return 0;

        case Instruction::ShuffleVector:
          return 0;

        case Instruction::ExtractElement:
          return 0;

        case Instruction::InsertElement:
          return 0;

        case Instruction::Add:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::FAdd:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::Sub:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::FSub:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::Mul:
          #ifdef SYS_AWARE
            //return 267;
         errs() << "Mul" << "\n";
            return 0;
          #else
            return 618;
          #endif
          
        case Instruction::FMul:
          #ifdef SYS_AWARE
            //return 267;
         errs() << "FMul" << "\n";
            return 0;
          #else
            return 618;
          #endif

        case Instruction::UDiv:
          #ifdef SYS_AWARE
           // return 1055;
        errs() << "UDiv" << "\n";
        return 320;
          #else
            return 1056;
          #endif

        case Instruction::SDiv:
          #ifdef SYS_AWARE
             // return 1214;
        errs() << "SDiv" << "\n";
        return 320;
          #else
            return 1185;
          #endif

        case Instruction::FDiv:
          #ifdef SYS_AWARE
            // return 1214;
        errs() << "FDiv" << "\n";
        return 320;
          #else
            return 1185;
          #endif

        case Instruction::URem:
          #ifdef SYS_AWARE
            // return 1122;
         errs() << "URem" << "\n";
         return 320;
          #else
            return 1312;
          #endif

        case Instruction::SRem:
          #ifdef SYS_AWARE
           // return 1299;
         errs() << "SRem" << "\n";
         return 320;
          #else
            return 1312;
          #endif

        case Instruction::FRem:
          #ifdef SYS_AWARE
            //return 1299;
         errs() << "FRem" << "\n";
            return 320;
          #else
            return 1312;
          #endif

        case Instruction::Shl:{

          Value *Operand_two = Inst->getOperand(1);

          if (Operand_two->getType()->isSingleValueType()) { // Shift by a single value (e.g. integer)
            errs() << "Weird shl\n";
            return 0;
          }

          #ifdef SYS_AWARE
            return 79;
          #else
            return 103;
          #endif
        
        }
        
        case Instruction::LShr: {
          
          Value *Operand_two = Inst->getOperand(1);

          if (Operand_two->getType()->isSingleValueType()) {// Shift by a single value (e.g. integer) 
            errs() << "Weird shl\n";
            return 0;
          }
             #ifdef SYS_AWARE
            return 79;
          #else
            return 101;
          #endif
        }

        case Instruction::AShr:{
          
          Value *Operand_two = Inst->getOperand(1);

          if (Operand_two->getType()->isSingleValueType()){ // Shift by a single value (e.g. integer)
            errs() << "Weird shl\n";
            return 0;
          }
          
          #ifdef SYS_AWARE
            return 99;
          #else
            return 145;
          #endif
        }

        case Instruction::And:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::Or:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::Xor:
          #ifdef SYS_AWARE
            return 32;
          #else
            return 33;
          #endif

        case Instruction::Select: // This is my estimation!
          //#ifdef SYS_AWARE //Iulian's old estimation
          //  return 16;
          //#else
          //  return 33;
          //#endif
          return 0;

        case Instruction::ICmp: // Check type of ICmp. (Equality or Relational) 
        {
          ICmpInst *Icmp = dyn_cast<ICmpInst>(&*Inst);

          if (Icmp->isEquality())
            #ifdef SYS_AWARE
              return 11;
            #else
              return 12;
            #endif
          else
            #ifdef SYS_AWARE
              return 16;
            #else
              return 17;
            #endif      
        }

        case Instruction::FCmp: // This is my estimation!
          #ifdef SYS_AWARE
            return 16;
          #else
            return 17;
          #endif 

        case Instruction::ZExt:
          return 0;

        case Instruction::SExt:
          return 0;

        case Instruction::FPToUI:
          return 0;

        case Instruction::FPToSI:
          return 0;

        case Instruction::FPExt:
          return 0;

        case Instruction::PtrToInt:
          return 0;

        case Instruction::IntToPtr:
          return 0;

        case Instruction::SIToFP:
          return 0;

        case Instruction::UIToFP:
          return 0;

        case Instruction::Trunc:
          return 0;

        case Instruction::FPTrunc:
          return 0;

        case Instruction::BitCast:
          return 0;

        default:  // Assume default as 0 LUTs in Area estimation.
          return 0;

        }// end of switch.
    //}
  }


static Value *GetAnyValue(Type *Ty) {
  /*
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
  case Type::HalfTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::PointerTyID:
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::TokenTyID:
    return Constant::getNullValue(Ty);
  default:
    return UndefValue::get(Ty);
  }
  */
  return UndefValue::get(Ty);
}

// Any two pointers in the same address space are equivalent, intptr_t and
// pointers are equivalent. Otherwise, standard type equivalence rules apply.
static bool isEquivalentType(Type *Ty1, Type *Ty2, const DataLayout *DL) {
  if (Ty1 == Ty2)
    return true;
  if (IdenticalType)
    return false;

  if (Ty1->getTypeID() != Ty2->getTypeID()) {
    LLVMContext &Ctx = Ty1->getContext();
    if (isa<PointerType>(Ty1) && Ty2 == DL->getIntPtrType(Ctx))
      return true;
    if (isa<PointerType>(Ty2) && Ty1 == DL->getIntPtrType(Ctx))
      return true;
    return false;
  }

  switch (Ty1->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release mode.
  case Type::IntegerTyID:
  case Type::VectorTyID:
    // Ty1 == Ty2 would have returned true earlier.
    return false;

  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return true;

  case Type::PointerTyID: {
    PointerType *PTy1 = cast<PointerType>(Ty1);
    PointerType *PTy2 = cast<PointerType>(Ty2);
    return (PTy1 == PTy2);
    // return isEquivalentType(PTy1->getElementType(),
    // PTy2->getElementType(),DL);  return PTy1->getAddressSpace() ==
    // PTy2->getAddressSpace();
  }

  case Type::StructTyID: {
    StructType *STy1 = cast<StructType>(Ty1);
    StructType *STy2 = cast<StructType>(Ty2);
    if (STy1->getNumElements() != STy2->getNumElements())
      return false;

    if (STy1->isPacked() != STy2->isPacked())
      return false;

    for (unsigned i = 0, e = STy1->getNumElements(); i != e; ++i) {
      if (!isEquivalentType(STy1->getElementType(i), STy2->getElementType(i),
                            DL))
        return false;
    }
    return true;
  }

  case Type::FunctionTyID: {
    FunctionType *FTy1 = cast<FunctionType>(Ty1);
    FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy1->getNumParams() != FTy2->getNumParams() ||
        FTy1->isVarArg() != FTy2->isVarArg())
      return false;

    if (!isEquivalentType(FTy1->getReturnType(), FTy2->getReturnType(), DL))
      return false;

    for (unsigned i = 0, e = FTy1->getNumParams(); i != e; ++i) {
      if (!isEquivalentType(FTy1->getParamType(i), FTy2->getParamType(i), DL))
        return false;
    }
    return true;
  }

  case Type::ArrayTyID: {
    ArrayType *ATy1 = cast<ArrayType>(Ty1);
    ArrayType *ATy2 = cast<ArrayType>(Ty2);
    return ATy1->getNumElements() == ATy2->getNumElements() &&
           isEquivalentType(ATy1->getElementType(), ATy2->getElementType(), DL);
  }
  }
}

/// Create a cast instruction if needed to cast V to type DstType. We treat
/// pointer and integer types of the same bitwidth as equivalent, so this can be
/// used to cast them to each other where needed. The function returns the Value
/// itself if no cast is needed, or a new CastInst instance inserted before
/// InsertBefore. The integer type equivalent to pointers must be passed as
/// IntPtrType (get it from DataLayout). This is guaranteed to generate no-op
/// casts, otherwise it will assert.
static Value *createCastIfNeeded(Value *V, Type *DstType, IRBuilder<> &Builder,
                                 Type *IntPtrType) {
  if (V->getType() == DstType || IdenticalType)
    return V;
  // BasicBlock *InsertAtEnd = dyn_cast<BasicBlock>(InstrOrBB);
  // Instruction *InsertBefore = dyn_cast<Instruction>(InstrOrBB);
  // BasicBlock *InsertBB = InsertAtEnd ? InsertAtEnd :
  // InsertBefore->getParent();

  Value *Result;
  Type *OrigType = V->getType();

  if (OrigType->isStructTy()) {
    assert(DstType->isStructTy());
    assert(OrigType->getStructNumElements() == DstType->getStructNumElements());

    // IRBuilder<> Builder(InsertBB);
    // if (InsertBefore)
    //  Builder.SetInsertPoint(InsertBefore);
    Result = UndefValue::get(DstType);
    for (unsigned int I = 0, E = OrigType->getStructNumElements(); I < E; ++I) {
      Value *ExtractedValue =
          Builder.CreateExtractValue(V, ArrayRef<unsigned int>(I));
      Value *Element =
          createCastIfNeeded(ExtractedValue, DstType->getStructElementType(I),
                             Builder, IntPtrType);
      Result =
          Builder.CreateInsertValue(Result, Element, ArrayRef<unsigned int>(I));
    }
    return Result;
  }
  assert(!DstType->isStructTy());

  if (OrigType->isPointerTy() &&
      (DstType->isIntegerTy() || DstType->isPointerTy())) {
    Result = Builder.CreatePointerCast(V, DstType, "merge_cast");
    // if (InsertBefore) {
    // Result = CastInst::CreatePointerCast(V, DstType, "", InsertBefore);
    //} else {
    // Result = CastInst::CreatePointerCast(V, DstType, "", InsertAtEnd);
    //}
  } else if (OrigType->isIntegerTy() && DstType->isPointerTy() &&
             OrigType == IntPtrType) {
    // Int -> Ptr
    Result = Builder.CreateCast(CastInst::IntToPtr, V, DstType, "merge_cast");
    // if (InsertBefore) {
    //  Result = CastInst::Create(CastInst::IntToPtr, V, DstType, "",
    //                            InsertBefore);
    //} else {
    //  Result = CastInst::Create(CastInst::IntToPtr, V, DstType, "",
    //                            InsertAtEnd);
    //}
  } else {
    llvm_unreachable("Can only cast int -> ptr or ptr -> (ptr or int)");
  }

  // assert(cast<CastInst>(Result)->isNoopCast(InsertAtEnd->getParent()->getParent()->getDataLayout())
  // &&
  //    "Cast is not a no-op cast. Potential loss of precision");

  return Result;
}

static bool valueEscapes(const Instruction *Inst) {
  const BasicBlock *BB = Inst->getParent();
  for (const User *U : Inst->users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (UI->getParent() != BB || isa<PHINode>(UI))
      return true;
  }
  return false;
}

// Helper for writeThunk,
// Selects proper bitcast operation,
// but a bit simpler then CastInst::getCastOpcode.
static Value *createCast(IRBuilder<> &Builder, Value *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy->isStructTy()) {
    assert(DestTy->isStructTy());
    assert(SrcTy->getStructNumElements() == DestTy->getStructNumElements());
    Value *Result = UndefValue::get(DestTy);
    for (unsigned int I = 0, E = SrcTy->getStructNumElements(); I < E; ++I) {
      Value *Element =
          createCast(Builder, Builder.CreateExtractValue(V, makeArrayRef(I)),
                     DestTy->getStructElementType(I));

      Result = Builder.CreateInsertValue(Result, Element, makeArrayRef(I));
    }
    return Result;
  }
  assert(!DestTy->isStructTy());
  if (SrcTy->isIntegerTy() && DestTy->isPointerTy())
    return Builder.CreateIntToPtr(V, DestTy);
  else if (SrcTy->isPointerTy() && DestTy->isIntegerTy())
    return Builder.CreatePtrToInt(V, DestTy);
  else
    return Builder.CreateBitCast(V, DestTy);
}

//-reg2mem
static void demoteRegToMem(Function &F) {
  if (F.isDeclaration())
    return;

  // Insert all new allocas into entry block.
  BasicBlock *BBEntry = &F.getEntryBlock();

  assert(pred_empty(BBEntry) &&
         "Entry block to function must not have predecessors!");

  // Find first non-alloca instruction and create insertion point. This is
  // safe if block is well-formed: it always have terminator, otherwise
  // we'll get and assertion.
  BasicBlock::iterator I = BBEntry->begin();
  while (isa<AllocaInst>(I))
    ++I;

  CastInst *AllocaInsertionPoint = new BitCastInst(
      Constant::getNullValue(Type::getInt32Ty(F.getContext())),
      Type::getInt32Ty(F.getContext()), "reg2mem alloca point", &*I);

  // Find the escaped instructions. But don't create stack slots for
  // allocas in entry block.
  std::list<Instruction *> WorkList;
  for (BasicBlock &ibb : F)
    for (BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie;
         ++iib) {
      if (!(isa<AllocaInst>(iib) && iib->getParent() == BBEntry) &&
          valueEscapes(&*iib)) {
        WorkList.push_front(&*iib);
      }
    }

  // Demote escaped instructions
  // NumRegsDemoted += WorkList.size();
  for (Instruction *ilb : WorkList)
    DemoteRegToStack(*ilb, false, AllocaInsertionPoint);

  WorkList.clear();

  // Find all phi's
  for (BasicBlock &ibb : F)
    for (BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie;
         ++iib)
      if (isa<PHINode>(iib))
        WorkList.push_front(&*iib);

  // Demote phi nodes
  // NumPhisDemoted += WorkList.size();
  for (Instruction *ilb : WorkList)
    DemotePHIToStack(cast<PHINode>(ilb), AllocaInsertionPoint);
}

//-mem2reg
static bool
promoteMemoryToRegister(Function &F,
                        DominatorTree &DT) { //, AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    // PromoteMemToReg(Allocas, DT, &AC);
    PromoteMemToReg(Allocas, DT, nullptr);
    // NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

//clearly the area has to be less than the two original areas summed
static double estimateFunctionTimeOvh(Function &F, TargetTransformInfo *TTI, double &timeSize) {
  double size = 0;
  for (Instruction &I : instructions(&F)) {
    size += TTI->getInstructionCost(
        &I, TargetTransformInfo::TargetCostKind::TCK_Latency);
        //&I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  }
  timeSize = 0.0;
  for (auto &BB : F) {
    
    std::string bbId = F.getName().str() + "::" + BB.getName().str();
    double functionProfile = 0.0; 
    if (FMSALegacyPass::mergedBBProfiles.find(&BB) != FMSALegacyPass::mergedBBProfiles.end()) {
      errs() << "Could find bb " << bbId << " which belongs to func " << F.getName().str() << "\n";
      functionProfile = FMSALegacyPass::mergedBBProfiles[&BB];
    }
    else
      errs() << "Couldn't find bb " << bbId << " in the dictionary of bb profiling for func " << F.getName().str() << "\n";
    
    if (functionProfile != 0.0)
      for (auto &I : BB) {
        timeSize += functionProfile*TTI->getInstructionCost(
            &I, TargetTransformInfo::TargetCostKind::TCK_Latency );
      }
  }
  return size;
}

typedef int instCounts;


static void printKeyValues(std::map<std::string, int> &dict) {
  for (auto it = dict.begin(); it != dict.end(); ++it) {
    errs() << it->first << ":" << it->second << "\n";
  }
}
  

static void printFunctionWideInstCounts(Function &F) {
  std::map<std::string, instCounts> functionWideInstCounts;
  for (Instruction &I : instructions(&F)) {
    if (functionWideInstCounts.find(std::string(I.getOpcodeName())) != functionWideInstCounts.end())
      functionWideInstCounts[std::string(I.getOpcodeName())] += 1;
    else
      functionWideInstCounts[std::string(I.getOpcodeName())] = 1;
  }
  errs() << "Function " << GetValueName(&F) << " has the following instruction types\n";
  printKeyValues(functionWideInstCounts);

  for(BasicBlock &BB : F) {
    std::map<std::string, instCounts> bbWideInstCounts;
    for (auto &I : BB) {
      if (bbWideInstCounts.find(std::string(I.getOpcodeName())) != bbWideInstCounts.end())
        bbWideInstCounts[std::string(I.getOpcodeName())] += 1;
      else
        bbWideInstCounts[std::string(I.getOpcodeName())] = 1;
    }
    errs() << "BB " << GetValueName(&F) + "::" + GetValueName(&BB) << " has the following instruction types\n";
    printKeyValues(functionWideInstCounts);
  }
}

enum EstimationType { AREA_ACCEL = 4, HW_LATENCY_ACCEL = 3, SW_LATENCY_ACCEL = 2 };

static void estimateNumberOfCallsForFunc(Function &F) {
    //BasicBlock &entryBB = F.getEntryBlock();
    //
    //if (FMSALegacyPass::bbProfiles.find(bbId) != FMSALegacyPass::bbProfiles.end()) {
    //    FMSALegacyPass::bbProfiles[bbId]/numInstr;
    //}

    for (auto &F_temp: *(F.getParent())) {
        if (F_temp.isDeclaration() || F_temp.isVarArg())
            continue;
        for (auto &BB : F_temp) {
            
            int numInstr = 0;
            for (auto &I:BB)
                ++numInstr;
            
            for (auto &I : BB) {
                CallInst *CI;
                if (CI = dyn_cast<CallInst>(&I)) {
                    
                    std::string bbId = GetValueName(&F_temp) + "::" + BB.getName().str();
                    
                    if (&F == CI->getCalledFunction() and FMSALegacyPass::bbProfiles.find(bbId) != FMSALegacyPass::bbProfiles.end()) {
                        double profile = FMSALegacyPass::bbProfiles[bbId]/numInstr;
                        global_num_calls[GetValueName(&F_temp) + " -> " + GetValueName(&F)] = profile;
                    }
                }
            }
        }
    }
    
}

static double estimateFunctionLatencyOrArea(Function &F, EstimationType estTy) {
  double size = 0.0;
  

  for (BasicBlock &BB : F){
    double functionProfile = 1.0;
      
    if (estTy == HW_LATENCY_ACCEL or estTy == SW_LATENCY_ACCEL)
      if ((FMSALegacyPass::bbProfiles.find(GetValueName(&F) + "::" + BB.getName().str()) != FMSALegacyPass::bbProfiles.end())) 
          functionProfile = FMSALegacyPass::bbProfiles[GetValueName(&F) + "::" + BB.getName().str()];
      else if (FMSALegacyPass::mergedBBProfiles.find(&BB) != FMSALegacyPass::mergedBBProfiles.end())
          functionProfile = FMSALegacyPass::mergedBBProfiles[&BB];
      else
          functionProfile = 0.0; //if a path is not touched by the profile, don't consider it.
    
    double bb_cost = 0.0;
    int i = 0;
    for (Instruction &I : BB){
      double i_cost;

      if (estTy == HW_LATENCY_ACCEL)
          i_cost = getHWDelayEstim(&I);
      else if (estTy == SW_LATENCY_ACCEL)
          i_cost = getCycleSWDelayEstim(&I);
      else if (estTy == AREA_ACCEL)
          i_cost = getAreaEstim(&I);
      else {
          errs() << "This function is not being used properly\n";
          exit(-1);
      }

      //size += i_cost;
      bb_cost += i_cost;
      ++i;
    }
    
    size += bb_cost*functionProfile;

    //errs() << "The time spent on bb " << BB.getName() << " is " << bb_cost << "\n";
    //errs() << "The number of instructions is " << i << " \n";
  }

  return size;
}

struct LLA{
    double area;
    double sw_lat;
    double hw_lat;

    LLA( double _area, double _sw_lat, double _hw_lat) {
        area = _area;
        sw_lat = _sw_lat;
        hw_lat = _hw_lat;
    }
    
	LLA& operator=(LLA const& copy)
    {
		area = copy.area;
		sw_lat = copy.sw_lat;
		hw_lat = copy.hw_lat;
        return *this;
    }

};

std::map<Function *, LLA> numbersAtMergeTime;
//std::map<Function *, double> areaAtMergeTime;
//std::map<Function *, double> swLatAtMergeTime;
//std::map<Function *, double> hwLatAtMergeTime;
//std::map<Function *, std::string> nameAtMergeTime;
std::map<std::string, double> areaAtMergeTime;
std::map<std::string, double> swLatAtMergeTime;
std::map<std::string, double> hwLatAtMergeTime;
std::map<std::string, long long int> numCallsAtMergeTime;
std::map<std::string, double> funcCodeSizeAtMergeTime;
std::set<std::string> nameAtMergeTime;


static double estimateFunctionCodeSize(Function &F, TargetTransformInfo *TTI) {
  double size = 0.0; 
  
  TargetTransformInfo::TargetCostKind costModel;

  costModel = TargetTransformInfo::TargetCostKind::TCK_CodeSize;

  for (BasicBlock &BB : F){
    double functionProfile = 1.0; 

    double bb_cost = 0.0; 
    int i = 0; 
    for (Instruction &I : BB){ 
      double i_cost = TTI->getInstructionCost(&I, costModel);

      bb_cost += i_cost;
      ++i;
    }
   
    size += bb_cost;

  }    

  return size;
}


static void saveFunctionEstimations(Function *F, TargetTransformInfo *TTI) {
    //numbersAtMergeTime[F] = LLA(estimateFunctionLatencyOrArea(*F, AREA_ACCEL), estimateFunctionLatencyOrArea(*F, SW_LATENCY_ACCEL), estimateFunctionLatencyOrArea(*F, HW_LATENCY_ACCEL));
	areaAtMergeTime[GetValueName(F)] = estimateFunctionLatencyOrArea(*F, AREA_ACCEL);
	swLatAtMergeTime[GetValueName(F)] = estimateFunctionLatencyOrArea(*F, SW_LATENCY_ACCEL);
	hwLatAtMergeTime[GetValueName(F)] = estimateFunctionLatencyOrArea(*F, HW_LATENCY_ACCEL);

	funcCodeSizeAtMergeTime[GetValueName(F)] = estimateFunctionCodeSize(*F, TTI);
    //numCallsAtMergeTime[GetValueName(F)] = estimateNumberOfCallsForFunc(*F);
    estimateNumberOfCallsForFunc(*F);
	nameAtMergeTime.insert(GetValueName(F));
    errs() << "The early area size for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, AREA_ACCEL)) << "\n";
    errs() << "The early hw latency for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, HW_LATENCY_ACCEL)) << "\n";
    errs() << "The early sw latency for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, SW_LATENCY_ACCEL)) << "\n";
}

static void updateDictOld(std::map<std::string, int> &opCounts,  Function &F, const int weight) {
    for (Instruction &I : instructions(&F)) {
        opCounts[I.getOpcodeName()] += weight;
    }
}

static void updateDict(std::map<std::string, int> &opCounts, Function &F, const int weight, std::set<std::string> &visitedFunctions) {
    if (visitedFunctions.find(F.getName()) != visitedFunctions.end()) {
        return;
    }
    visitedFunctions.insert(F.getName());

    for (Instruction &I : instructions(&F)) {
        opCounts[I.getOpcodeName()] += weight;

        CallInst *CI;
        if (CI = dyn_cast<CallInst>(&I)) {
            Function *CF;
            if (CF = CI->getCalledFunction()) {
                updateDict(opCounts, *CF, weight, visitedFunctions);
            }
        }
    }
}

static double estimateFunctionDifference(Function &FMerged, Function &FInput1, Function &FInput2) {
  std::map<std::string, int> opCounts;
  std::map<std::string, int> opCounts1;
  std::map<std::string, int> opCounts2;
  std::map<std::string, int> opCountsMerged;
  for (auto it = FMSALegacyPass::opLuts.begin(); it != FMSALegacyPass::opLuts.end(); ++it)
    opCounts[it->first] = 0;

  std::set<std::string> visitedFunctions1;
  updateDict(opCounts, FInput1, 1, visitedFunctions1);
  std::set<std::string> visitedFunctions2;
  updateDict(opCounts, FInput2, 1, visitedFunctions2);
  std::set<std::string> visitedMerged;
  updateDict(opCounts, FMerged, -1, visitedMerged);

  //this is just for debug purposes to separate area sizes by function
  std::set<std::string> visitedFunctionsDebug1;
  updateDict(opCounts1, FInput1, 1, visitedFunctionsDebug1);
  std::set<std::string> visitedFunctionsDebug2;
  updateDict(opCounts2, FInput2, 1, visitedFunctionsDebug2);
  std::set<std::string> visitedMergedDebug;
  updateDict(opCountsMerged, FMerged, -1, visitedMergedDebug);

  

  for (auto it = visitedFunctions1.begin(); it != visitedFunctions1.end(); ++it) {
        if (visitedMerged.find(*it) == visitedMerged.end())
            errs() << "We could not find function1 " << *it << " in the visitedMerged set\n";
  }

  for (auto it = visitedFunctions2.begin(); it != visitedFunctions2.end(); ++it) {
        if (visitedMerged.find(*it) == visitedMerged.end())
            errs() << "We could not find function2 " << *it << " in the visitedMerged set\n";
  }

  double lutDifference = 0; 
  errs() << "Op lut differences and their weight printed for func " << GetValueName(&FMerged) << "\n";
  for (auto it = FMSALegacyPass::opLuts.begin(); it != FMSALegacyPass::opLuts.end(); ++it) {
        errs() << "op=" << it->first << " counts=" << opCounts[it->first] << ";";
        lutDifference += opCounts[it->first]*(it->second);
  }

  errs() << "Op lut counts and their weight printed for func " << GetValueName(&FMerged) << "\n";
  for (auto it = FMSALegacyPass::opLuts.begin(); it != FMSALegacyPass::opLuts.end(); ++it) {
        errs() << "op=" << it->first << " counts=" << opCounts1[it->first] << ";";
  }

  for (auto it = FMSALegacyPass::opLuts.begin(); it != FMSALegacyPass::opLuts.end(); ++it) {
        errs() << "op=" << it->first << " counts=" << opCounts2[it->first] << ";";
  }

  for (auto it = FMSALegacyPass::opLuts.begin(); it != FMSALegacyPass::opLuts.end(); ++it) {
        errs() << "op=" << it->first << " counts=" << opCountsMerged[it->first] << ";";
  }

  errs() << "\n";
  lutDifference += FMSALegacyPass::opLuts["intercept"];
        //FMSALegacyPass::opLuts[Inst->getOpcodeName()];
  return lutDifference;
}
static double estimateFunctionSize(Function &F, TargetTransformInfo *TTI, double &timeSize) {
  saveFunctionEstimations(&F, TTI);
  //printFunctionWideInstCounts(F); You used this to debug dynamic information
  double size = 0;
  //if (bb_profiles.empty()){
      for (Instruction &I : instructions(&F)) {

        //size += TTI->getInstructionCost(
        //    &I, TargetTransformInfo::TargetCostKind::TCK_Latency);

        size += getAreaEstim(&I);
        //size += TTI->getInstructionCost(
        //    &I, TargetTransformInfo::TargetCostKind::TCK_Latency);
            //&I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
      }

  //else
  
  //if (FMSALegacyPass::bbProfiles.empty()) {
  //  report_fatal_error("No basic blocks were read for this module");
  //}

  //the most ugly situation is if the function F cannot be found in bb_profiles
  //at all
  timeSize = 0.0;
  double sumProfForFunc = 0.0;
  //errs() << "Estimating the time size for function " << F.getName() << " \n";
  for (auto &BB : F) {
    
    //errs() << "Estimating the time size for BB " << BB.getName() << " \n";
    std::string bbId = GetValueName(&F) + "::" + BB.getName().str();
    double functionProfile = 0.0; 
    if (FMSALegacyPass::mergedBBProfiles.find(&BB) != FMSALegacyPass::mergedBBProfiles.end()) {
      functionProfile = FMSALegacyPass::mergedBBProfiles[&BB];
      //errs() << "The function profile you've found for bb " << bbId << " is in mergedBBProfiles " << functionProfile << "\n";
    } else if (FMSALegacyPass::bbProfiles.find(bbId) != FMSALegacyPass::bbProfiles.end()) {
      functionProfile = FMSALegacyPass::bbProfiles[bbId];
      //errs() << "The function profile you've found for bb " << bbId << " is in bbProfiles " << functionProfile << "\n";
    
    } else if (FMSALegacyPass::bbProfiles.empty()) {
      functionProfile = 1.0;
      //errs() << "The function profile you've found for bb " << bbId << " is in noBBProfiles " << functionProfile << "\n";
    }
    else {
        //errs() << "Couldn't find bb " << bbId << " in the dictionary of bb profiling for func " << GetValueName(&F) << "\n";
        functionProfile = 0.0; //if there are no profiles, you merge based on static data
    }
    sumProfForFunc += functionProfile;
    double bbProfile = 0.0;
    double summedLats = 0.0;
    if (functionProfile != 0.0)
      for (auto &I : BB) {
        //double iProfile = functionProfile*TTI->getInstructionCost(
        //    &I, TargetTransformInfo::TargetCostKind::TCK_Latency );

        double iProfile = getCycleSWDelayEstim(&I);
        summedLats += iProfile;
        timeSize += iProfile*functionProfile;
        bbProfile += iProfile*functionProfile;
      }
    //errs() << "Complete bb profile for " << bbId << " is " << bbProfile << "; summedLats=" << summedLats << "; funcProfile" << functionProfile << "\n";
  }
  
  //errs() << "The size in terms of profile for function " << GetValueName(&F) << " is " << sumProfForFunc << "\n";
  //if (LatencyMerges and not FMSALegacyPass::bbProfiles.empty()) {
  //  errs() << "The size without rescaling is " << size << " and with time scaling it is " << timeSize << "\n";
  //  size = timeSize;
  //}

     
  return size;
}

static bool fixNotDominatedUses(Function *F, DominatorTree &DT) {

  std::list<Instruction *> WorkList;
  std::map<Instruction *, Value *> StoredAddress;

  std::map< Instruction *, std::map< Instruction *, std::list<unsigned> > >
      UpdateList;

  for (Instruction &I : instructions(*F)) {
    for (auto *U : I.users()) {
      Instruction *UI = dyn_cast<Instruction>(U);
      if (UI && !DT.dominates(&I, UI)) {
        auto &ListOperands = UpdateList[&I][UI];
        for (unsigned i = 0; i < UI->getNumOperands(); i++) {
          if (UI->getOperand(i) == (Value *)(&I)) {
            ListOperands.push_back(i);
          }
        }
      }
    }
    if (UpdateList[&I].size() > 0) {
      IRBuilder<> Builder(&*F->getEntryBlock().getFirstInsertionPt());
      StoredAddress[&I] = Builder.CreateAlloca(I.getType());
      //Builder.CreateStore(GetAnyValue(I.getType()), StoredAddress[&I]);
      Value *V = &I;
      if (I.getParent()->getTerminator()) {
        InvokeInst *II = dyn_cast<InvokeInst>(I.getParent()->getTerminator());
        if ((&I)==I.getParent()->getTerminator() && II!=nullptr) {
          BasicBlock *SrcBB = I.getParent();
          BasicBlock *DestBB = II->getNormalDest();
          Builder.SetInsertPoint(DestBB->getFirstNonPHI());
          //create PHI
          if (DestBB->getSinglePredecessor()==nullptr) {
            PHINode *PHI = Builder.CreatePHI( I.getType(), 0 );
            for (auto it = pred_begin(DestBB), et = pred_end(DestBB); it != et; ++it) {
              BasicBlock *BB = *it;
              if (BB==SrcBB) {
                PHI->addIncoming(&I,BB);
              } else {
                PHI->addIncoming( GetAnyValue(I.getType()) ,BB);
              }
            }
            V = PHI;
          }
        } else {
          Builder.SetInsertPoint(I.getParent()->getTerminator());
        }
      } else {
        Builder.SetInsertPoint(I.getParent());
      }
      Builder.CreateStore(V, StoredAddress[&I]);
    }
  }
  for (auto &kv1 : UpdateList) {
    Instruction *I = kv1.first;
    for (auto &kv : kv1.second) {
      Instruction *UI = kv.first;
      IRBuilder<> Builder(UI);
      Value *V = Builder.CreateLoad(StoredAddress[I]);
      for (unsigned i : kv.second) {
        UI->setOperand(i, V);
      }
    }
  }

  return true;
}

static void removeRedundantInstructions(Function *F, DominatorTree &DT,
                                   std::vector<Instruction *> &ListInsts) {

  std::set<Instruction *> SkipList;

  std::map<Instruction *, std::list<Instruction *>> UpdateList;

  for (Instruction *I1 : ListInsts) {
    if (SkipList.find(I1) != SkipList.end())
      continue;
    for (Instruction *I2 : ListInsts) {
      if (I1 == I2)
        continue;
      if (SkipList.find(I2) != SkipList.end())
        continue;
      assert(I1->getNumOperands() == I2->getNumOperands() &&
             "Should have the same num of operands!");
      bool AllEqual = true;
      for (unsigned i = 0; i < I1->getNumOperands(); ++i) {
        AllEqual = AllEqual && (I1->getOperand(i) == I2->getOperand(i));
      }

      if (AllEqual && DT.dominates(I1, I2)) {
        UpdateList[I1].push_back(I2);
        SkipList.insert(I2);
        SkipList.insert(I1);
      }
    }
  }

  for (auto &kv : UpdateList) {
    for (auto *I : kv.second) {
      I->replaceAllUsesWith(kv.first);
      I->eraseFromParent();
    }
  }
}

static bool matchIntrinsicCalls(Intrinsic::ID ID, const CallInst *CI1,
                                const CallInst *CI2) {
  Intrinsic::ID ID1;
  Intrinsic::ID ID2;
  if (Function *F = CI1->getCalledFunction())
    ID1 = (Intrinsic::ID)F->getIntrinsicID();
  if (Function *F = CI2->getCalledFunction())
    ID2 = (Intrinsic::ID)F->getIntrinsicID();

  if (ID1 != ID)
    return false;
  if (ID1 != ID2)
    return false;

  switch (ID) {
  default:
    break;
  case Intrinsic::coro_id: {
    /*
    auto *InfoArg = CS.getArgOperand(3)->stripPointerCasts();
    if (isa<ConstantPointerNull>(InfoArg))
      break;
    auto *GV = dyn_cast<GlobalVariable>(InfoArg);
    Assert(GV && GV->isConstant() && GV->hasDefinitiveInitializer(),
      "info argument of llvm.coro.begin must refer to an initialized "
      "constant");
    Constant *Init = GV->getInitializer();
    Assert(isa<ConstantStruct>(Init) || isa<ConstantArray>(Init),
      "info argument of llvm.coro.begin must refer to either a struct or "
      "an array");
    */
    break;
  }
  case Intrinsic::ctlz: // llvm.ctlz
  case Intrinsic::cttz: // llvm.cttz
    // Assert(isa<ConstantInt>(CS.getArgOperand(1)),
    //       "is_zero_undef argument of bit counting intrinsics must be a "
    //       "constant int",
    //       CS);
    return CI1->getArgOperand(1) == CI2->getArgOperand(1);
    break;
  case Intrinsic::experimental_constrained_fadd:
  case Intrinsic::experimental_constrained_fsub:
  case Intrinsic::experimental_constrained_fmul:
  case Intrinsic::experimental_constrained_fdiv:
  case Intrinsic::experimental_constrained_frem:
  case Intrinsic::experimental_constrained_fma:
  case Intrinsic::experimental_constrained_sqrt:
  case Intrinsic::experimental_constrained_pow:
  case Intrinsic::experimental_constrained_powi:
  case Intrinsic::experimental_constrained_sin:
  case Intrinsic::experimental_constrained_cos:
  case Intrinsic::experimental_constrained_exp:
  case Intrinsic::experimental_constrained_exp2:
  case Intrinsic::experimental_constrained_log:
  case Intrinsic::experimental_constrained_log10:
  case Intrinsic::experimental_constrained_log2:
  case Intrinsic::experimental_constrained_rint:
  case Intrinsic::experimental_constrained_nearbyint:
    // visitConstrainedFPIntrinsic(
    //    cast<ConstrainedFPIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_declare: // llvm.dbg.declare
    // Assert(isa<MetadataAsValue>(CS.getArgOperand(0)),
    //       "invalid llvm.dbg.declare intrinsic call 1", CS);
    // visitDbgIntrinsic("declare",
    // cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_addr: // llvm.dbg.addr
    // visitDbgIntrinsic("addr", cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_value: // llvm.dbg.value
    // visitDbgIntrinsic("value", cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_label: // llvm.dbg.label
    // visitDbgLabelIntrinsic("label",
    // cast<DbgLabelInst>(*CS.getInstruction()));
    break;
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset: {
    /*
    const auto *MI = cast<MemIntrinsic>(CS.getInstruction());
    auto IsValidAlignment = [&](unsigned Alignment) -> bool {
      return Alignment == 0 || isPowerOf2_32(Alignment);
    };
    Assert(IsValidAlignment(MI->getDestAlignment()),
           "alignment of arg 0 of memory intrinsic must be 0 or a power of 2",
           CS);
    if (const auto *MTI = dyn_cast<MemTransferInst>(MI)) {
      Assert(IsValidAlignment(MTI->getSourceAlignment()),
             "alignment of arg 1 of memory intrinsic must be 0 or a power of 2",
             CS);
    }
    Assert(isa<ConstantInt>(CS.getArgOperand(3)),
           "isvolatile argument of memory intrinsics must be a constant int",
           CS);
    */

    /*//TODO: fix here
    const auto *MI1 = dyn_cast<MemIntrinsic>(CI1);
    const auto *MI2 = dyn_cast<MemIntrinsic>(CI2);
    if (MI1->getDestAlignment()!=MI2->getDestAlignment()) return false;
    const auto *MTI1 = dyn_cast<MemTransferInst>(CI1);
    const auto *MTI2 = dyn_cast<MemTransferInst>(CI2);
    if (MTI1!=nullptr) {
       if(MTI2==nullptr) return false;
       if (MTI1->getSourceAlignment()!=MTI2->getSourceAlignment()) return false;
    }
    */
    return CI1->getArgOperand(3) == CI2->getArgOperand(3);

    break;
  }
  case Intrinsic::memcpy_element_unordered_atomic:
  case Intrinsic::memmove_element_unordered_atomic:
  case Intrinsic::memset_element_unordered_atomic: {
    /*
    const auto *AMI = cast<AtomicMemIntrinsic>(CS.getInstruction());

    ConstantInt *ElementSizeCI =
        dyn_cast<ConstantInt>(AMI->getRawElementSizeInBytes());
    Assert(ElementSizeCI,
           "element size of the element-wise unordered atomic memory "
           "intrinsic must be a constant int",
           CS);
    const APInt &ElementSizeVal = ElementSizeCI->getValue();
    Assert(ElementSizeVal.isPowerOf2(),
           "element size of the element-wise atomic memory intrinsic "
           "must be a power of 2",
           CS);

    if (auto *LengthCI = dyn_cast<ConstantInt>(AMI->getLength())) {
      uint64_t Length = LengthCI->getZExtValue();
      uint64_t ElementSize = AMI->getElementSizeInBytes();
      Assert((Length % ElementSize) == 0,
             "constant length must be a multiple of the element size in the "
             "element-wise atomic memory intrinsic",
             CS);
    }

    auto IsValidAlignment = [&](uint64_t Alignment) {
      return isPowerOf2_64(Alignment) && ElementSizeVal.ule(Alignment);
    };
    uint64_t DstAlignment = AMI->getDestAlignment();
    Assert(IsValidAlignment(DstAlignment),
           "incorrect alignment of the destination argument", CS);
    if (const auto *AMT = dyn_cast<AtomicMemTransferInst>(AMI)) {
      uint64_t SrcAlignment = AMT->getSourceAlignment();
      Assert(IsValidAlignment(SrcAlignment),
             "incorrect alignment of the source argument", CS);
    }
    */
    break;
  }
  case Intrinsic::gcroot:
  case Intrinsic::gcwrite:
  case Intrinsic::gcread:
    /*
    if (ID == Intrinsic::gcroot) {
      AllocaInst *AI =
        dyn_cast<AllocaInst>(CS.getArgOperand(0)->stripPointerCasts());
      Assert(AI, "llvm.gcroot parameter #1 must be an alloca.", CS);
      Assert(isa<Constant>(CS.getArgOperand(1)),
             "llvm.gcroot parameter #2 must be a constant.", CS);
      if (!AI->getAllocatedType()->isPointerTy()) {
        Assert(!isa<ConstantPointerNull>(CS.getArgOperand(1)),
               "llvm.gcroot parameter #1 must either be a pointer alloca, "
               "or argument #2 must be a non-null constant.",
               CS);
      }
    }

    Assert(CS.getParent()->getParent()->hasGC(),
           "Enclosing function does not use GC.", CS);
    */
    break;
  case Intrinsic::init_trampoline:
    /*
    Assert(isa<Function>(CS.getArgOperand(1)->stripPointerCasts()),
           "llvm.init_trampoline parameter #2 must resolve to a function.",
           CS);
    */
    break;
  case Intrinsic::prefetch:
    /*
    Assert(isa<ConstantInt>(CS.getArgOperand(1)) &&
               isa<ConstantInt>(CS.getArgOperand(2)) &&
               cast<ConstantInt>(CS.getArgOperand(1))->getZExtValue() < 2 &&
               cast<ConstantInt>(CS.getArgOperand(2))->getZExtValue() < 4,
           "invalid arguments to llvm.prefetch", CS);
    */
    return (CI1->getArgOperand(1) == CI2->getArgOperand(1) &&
            CI1->getArgOperand(2) == CI2->getArgOperand(2));

    break;
  case Intrinsic::stackprotector:
    /*
    Assert(isa<AllocaInst>(CS.getArgOperand(1)->stripPointerCasts()),
           "llvm.stackprotector parameter #2 must resolve to an alloca.", CS);
    */
    break;
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
    /*
    Assert(isa<ConstantInt>(CS.getArgOperand(0)),
           "size argument of memory use markers must be a constant integer",
           CS);
    */
    return CI1->getArgOperand(0) == CI2->getArgOperand(0);
    break;
  case Intrinsic::invariant_end:
    /*
    Assert(isa<ConstantInt>(CS.getArgOperand(1)),
           "llvm.invariant.end parameter #2 must be a constant integer", CS);
    */
    return CI1->getArgOperand(1) == CI2->getArgOperand(1);
    break;

  case Intrinsic::localescape: {
    /*
    BasicBlock *BB = CS.getParent();
    Assert(BB == &BB->getParent()->front(),
           "llvm.localescape used outside of entry block", CS);
    Assert(!SawFrameEscape,
           "multiple calls to llvm.localescape in one function", CS);
    for (Value *Arg : CS.args()) {
      if (isa<ConstantPointerNull>(Arg))
        continue; // Null values are allowed as placeholders.
      auto *AI = dyn_cast<AllocaInst>(Arg->stripPointerCasts());
      Assert(AI && AI->isStaticAlloca(),
             "llvm.localescape only accepts static allocas", CS);
    }
    FrameEscapeInfo[BB->getParent()].first = CS.getNumArgOperands();
    SawFrameEscape = true;
    */
    break;
  }
  case Intrinsic::localrecover: {
    /*
    Value *FnArg = CS.getArgOperand(0)->stripPointerCasts();
    Function *Fn = dyn_cast<Function>(FnArg);
    Assert(Fn && !Fn->isDeclaration(),
           "llvm.localrecover first "
           "argument must be function defined in this module",
           CS);
    auto *IdxArg = dyn_cast<ConstantInt>(CS.getArgOperand(2));
    Assert(IdxArg, "idx argument of llvm.localrecover must be a constant int",
           CS);
    auto &Entry = FrameEscapeInfo[Fn];
    Entry.second = unsigned(
        std::max(uint64_t(Entry.second), IdxArg->getLimitedValue(~0U) + 1));
    */
    break;
  }
    /*
    case Intrinsic::experimental_gc_statepoint:
      Assert(!CS.isInlineAsm(),
             "gc.statepoint support for inline assembly unimplemented", CS);
      Assert(CS.getParent()->getParent()->hasGC(),
             "Enclosing function does not use GC.", CS);

      verifyStatepoint(CS);
      break;
    case Intrinsic::experimental_gc_result: {
      Assert(CS.getParent()->getParent()->hasGC(),
             "Enclosing function does not use GC.", CS);
      // Are we tied to a statepoint properly?
      CallSite StatepointCS(CS.getArgOperand(0));
      const Function *StatepointFn =
        StatepointCS.getInstruction() ? StatepointCS.getCalledFunction() :
    nullptr; Assert(StatepointFn && StatepointFn->isDeclaration() &&
                 StatepointFn->getIntrinsicID() ==
                     Intrinsic::experimental_gc_statepoint,
             "gc.result operand #1 must be from a statepoint", CS,
             CS.getArgOperand(0));

      // Assert that result type matches wrapped callee.
      const Value *Target = StatepointCS.getArgument(2);
      auto *PT = cast<PointerType>(Target->getType());
      auto *TargetFuncType = cast<FunctionType>(PT->getElementType());
      Assert(CS.getType() == TargetFuncType->getReturnType(),
             "gc.result result type does not match wrapped callee", CS);
      break;
    }
    case Intrinsic::experimental_gc_relocate: {
      Assert(CS.getNumArgOperands() == 3, "wrong number of arguments", CS);

      Assert(isa<PointerType>(CS.getType()->getScalarType()),
             "gc.relocate must return a pointer or a vector of pointers", CS);

      // Check that this relocate is correctly tied to the statepoint

      // This is case for relocate on the unwinding path of an invoke statepoint
      if (LandingPadInst *LandingPad =
            dyn_cast<LandingPadInst>(CS.getArgOperand(0))) {

        const BasicBlock *InvokeBB =
            LandingPad->getParent()->getUniquePredecessor();

        // Landingpad relocates should have only one predecessor with invoke
        // statepoint terminator
        Assert(InvokeBB, "safepoints should have unique landingpads",
               LandingPad->getParent());
        Assert(InvokeBB->getTerminator(), "safepoint block should be well
    formed", InvokeBB); Assert(isStatepoint(InvokeBB->getTerminator()), "gc
    relocate should be linked to a statepoint", InvokeBB);
      }
      else {
        // In all other cases relocate should be tied to the statepoint
    directly.
        // This covers relocates on a normal return path of invoke statepoint
    and
        // relocates of a call statepoint.
        auto Token = CS.getArgOperand(0);
        Assert(isa<Instruction>(Token) &&
    isStatepoint(cast<Instruction>(Token)), "gc relocate is incorrectly tied to
    the statepoint", CS, Token);
      }

      // Verify rest of the relocate arguments.

      ImmutableCallSite StatepointCS(
          cast<GCRelocateInst>(*CS.getInstruction()).getStatepoint());

      // Both the base and derived must be piped through the safepoint.
      Value* Base = CS.getArgOperand(1);
      Assert(isa<ConstantInt>(Base),
             "gc.relocate operand #2 must be integer offset", CS);

      Value* Derived = CS.getArgOperand(2);
      Assert(isa<ConstantInt>(Derived),
             "gc.relocate operand #3 must be integer offset", CS);

      const int BaseIndex = cast<ConstantInt>(Base)->getZExtValue();
      const int DerivedIndex = cast<ConstantInt>(Derived)->getZExtValue();
      // Check the bounds
      Assert(0 <= BaseIndex && BaseIndex < (int)StatepointCS.arg_size(),
             "gc.relocate: statepoint base index out of bounds", CS);
      Assert(0 <= DerivedIndex && DerivedIndex < (int)StatepointCS.arg_size(),
             "gc.relocate: statepoint derived index out of bounds", CS);

      // Check that BaseIndex and DerivedIndex fall within the 'gc parameters'
      // section of the statepoint's argument.
      Assert(StatepointCS.arg_size() > 0,
             "gc.statepoint: insufficient arguments");
      Assert(isa<ConstantInt>(StatepointCS.getArgument(3)),
             "gc.statement: number of call arguments must be constant integer");
      const unsigned NumCallArgs =
          cast<ConstantInt>(StatepointCS.getArgument(3))->getZExtValue();
      Assert(StatepointCS.arg_size() > NumCallArgs + 5,
             "gc.statepoint: mismatch in number of call arguments");
      Assert(isa<ConstantInt>(StatepointCS.getArgument(NumCallArgs + 5)),
             "gc.statepoint: number of transition arguments must be "
             "a constant integer");
      const int NumTransitionArgs =
          cast<ConstantInt>(StatepointCS.getArgument(NumCallArgs + 5))
              ->getZExtValue();
      const int DeoptArgsStart = 4 + NumCallArgs + 1 + NumTransitionArgs + 1;
      Assert(isa<ConstantInt>(StatepointCS.getArgument(DeoptArgsStart)),
             "gc.statepoint: number of deoptimization arguments must be "
             "a constant integer");
      const int NumDeoptArgs =
          cast<ConstantInt>(StatepointCS.getArgument(DeoptArgsStart))
              ->getZExtValue();
      const int GCParamArgsStart = DeoptArgsStart + 1 + NumDeoptArgs;
      const int GCParamArgsEnd = StatepointCS.arg_size();
      Assert(GCParamArgsStart <= BaseIndex && BaseIndex < GCParamArgsEnd,
             "gc.relocate: statepoint base index doesn't fall within the "
             "'gc parameters' section of the statepoint call",
             CS);
      Assert(GCParamArgsStart <= DerivedIndex && DerivedIndex < GCParamArgsEnd,
             "gc.relocate: statepoint derived index doesn't fall within the "
             "'gc parameters' section of the statepoint call",
             CS);

      // Relocated value must be either a pointer type or vector-of-pointer
    type,
      // but gc_relocate does not need to return the same pointer type as the
      // relocated pointer. It can be casted to the correct type later if it's
      // desired. However, they must have the same address space and
    'vectorness' GCRelocateInst &Relocate =
    cast<GCRelocateInst>(*CS.getInstruction());
      Assert(Relocate.getDerivedPtr()->getType()->isPtrOrPtrVectorTy(),
             "gc.relocate: relocated value must be a gc pointer", CS);

      auto ResultType = CS.getType();
      auto DerivedType = Relocate.getDerivedPtr()->getType();
      Assert(ResultType->isVectorTy() == DerivedType->isVectorTy(),
             "gc.relocate: vector relocates to vector and pointer to pointer",
             CS);
      Assert(
          ResultType->getPointerAddressSpace() ==
              DerivedType->getPointerAddressSpace(),
          "gc.relocate: relocating a pointer shouldn't change its address
    space", CS); break;
    }
    case Intrinsic::eh_exceptioncode:
    case Intrinsic::eh_exceptionpointer: {
      Assert(isa<CatchPadInst>(CS.getArgOperand(0)),
             "eh.exceptionpointer argument must be a catchpad", CS);
      break;
    }
    case Intrinsic::masked_load: {
      Assert(CS.getType()->isVectorTy(), "masked_load: must return a vector",
    CS);

      Value *Ptr = CS.getArgOperand(0);
      //Value *Alignment = CS.getArgOperand(1);
      Value *Mask = CS.getArgOperand(2);
      Value *PassThru = CS.getArgOperand(3);
      Assert(Mask->getType()->isVectorTy(),
             "masked_load: mask must be vector", CS);

      // DataTy is the overloaded type
      Type *DataTy = cast<PointerType>(Ptr->getType())->getElementType();
      Assert(DataTy == CS.getType(),
             "masked_load: return must match pointer type", CS);
      Assert(PassThru->getType() == DataTy,
             "masked_load: pass through and data type must match", CS);
      Assert(Mask->getType()->getVectorNumElements() ==
             DataTy->getVectorNumElements(),
             "masked_load: vector mask must be same length as data", CS);
      break;
    }
    case Intrinsic::masked_store: {
      Value *Val = CS.getArgOperand(0);
      Value *Ptr = CS.getArgOperand(1);
      //Value *Alignment = CS.getArgOperand(2);
      Value *Mask = CS.getArgOperand(3);
      Assert(Mask->getType()->isVectorTy(),
             "masked_store: mask must be vector", CS);

      // DataTy is the overloaded type
      Type *DataTy = cast<PointerType>(Ptr->getType())->getElementType();
      Assert(DataTy == Val->getType(),
             "masked_store: storee must match pointer type", CS);
      Assert(Mask->getType()->getVectorNumElements() ==
             DataTy->getVectorNumElements(),
             "masked_store: vector mask must be same length as data", CS);
      break;
    }

    case Intrinsic::experimental_guard: {
      Assert(CS.isCall(), "experimental_guard cannot be invoked", CS);
      Assert(CS.countOperandBundlesOfType(LLVMContext::OB_deopt) == 1,
             "experimental_guard must have exactly one "
             "\"deopt\" operand bundle");
      break;
    }

    case Intrinsic::experimental_deoptimize: {
      Assert(CS.isCall(), "experimental_deoptimize cannot be invoked", CS);
      Assert(CS.countOperandBundlesOfType(LLVMContext::OB_deopt) == 1,
             "experimental_deoptimize must have exactly one "
             "\"deopt\" operand bundle");
      Assert(CS.getType() ==
    CS.getInstruction()->getFunction()->getReturnType(),
             "experimental_deoptimize return type must match caller return
    type");

      if (CS.isCall()) {
        auto *DeoptCI = CS.getInstruction();
        auto *RI = dyn_cast<ReturnInst>(DeoptCI->getNextNode());
        Assert(RI,
               "calls to experimental_deoptimize must be followed by a return");

        if (!CS.getType()->isVoidTy() && RI)
          Assert(RI->getReturnValue() == DeoptCI,
                 "calls to experimental_deoptimize must be followed by a return
    " "of the value computed by experimental_deoptimize");
      }

      break;
    }
    */
  };
  return false; // TODO: change to false by default
}

static bool matchLandingPad(LandingPadInst *LP1, LandingPadInst *LP2) {
  if (LP1->getType() != LP2->getType())
    return false;
  if (LP1->isCleanup() != LP2->isCleanup())
    return false;
  if (LP1->getNumClauses() != LP2->getNumClauses())
    return false;
  for (unsigned i = 0; i < LP1->getNumClauses(); i++) {
    if (LP1->isCatch(i) != LP2->isCatch(i))
      return false;
    if (LP1->isFilter(i) != LP2->isFilter(i))
      return false;
    if (LP1->getClause(i) != LP2->getClause(i))
      return false;
  }
  return true;
}

static bool match(Instruction *I1, Instruction *I2) {

  if (I1->getOpcode() != I2->getOpcode()) return false;

  //Returns are special cases that can differ in the number of operands
  if (I1->getOpcode() == Instruction::Ret)
    return true;

  if (I1->getNumOperands() != I2->getNumOperands())
    return false;

  bool sameType = false;
  if (IdenticalType) {
    sameType = (I1->getType() == I2->getType());
    for (unsigned i = 0; i < I1->getNumOperands(); i++) {
      sameType = sameType &&
                 (I1->getOperand(i)->getType() == I2->getOperand(i)->getType());
    }
  } else {
    const DataLayout *DT =
        &((Module *)I1->getParent()->getParent()->getParent())->getDataLayout();
    sameType = isEquivalentType(I1->getType(), I2->getType(), DT);
    for (unsigned i = 0; i < I1->getNumOperands(); i++) {
      sameType = sameType && isEquivalentType(I1->getOperand(i)->getType(),
                                              I2->getOperand(i)->getType(), DT);
    }
  }
  if (!sameType)
    return false;

  switch (I1->getOpcode()) {

  case Instruction::Load: {
    const LoadInst *LI = dyn_cast<LoadInst>(I1);
    const LoadInst *LI2 = cast<LoadInst>(I2);
    return LI->isVolatile() == LI2->isVolatile() &&
           LI->getAlignment() == LI2->getAlignment() &&
           LI->getOrdering() == LI2->getOrdering(); // &&
    // LI->getSyncScopeID() == LI2->getSyncScopeID() &&
    // LI->getMetadata(LLVMContext::MD_range)
    //  == LI2->getMetadata(LLVMContext::MD_range);
  }
  case Instruction::Store: {
    const StoreInst *SI = dyn_cast<StoreInst>(I1);
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I2)->getAlignment() &&
           SI->getOrdering() == cast<StoreInst>(I2)->getOrdering(); // &&
    // SI->getSyncScopeID() == cast<StoreInst>(I2)->getSyncScopeID();
  }
  case Instruction::Alloca: {
    const AllocaInst *AI = dyn_cast<AllocaInst>(I1);
    if (AI->getArraySize() != cast<AllocaInst>(I2)->getArraySize() ||
        AI->getAlignment() != cast<AllocaInst>(I2)->getAlignment())
      return false;

    /*
    // If size is known, I2 can be seen as equivalent to I1 if it allocates
    // the same or less memory.
    if (DL->getTypeAllocSize(AI->getAllocatedType())
          < DL->getTypeAllocSize(cast<AllocaInst>(I2)->getAllocatedType()))
      return false;

    return true;
    */
    break;
  }
  case Instruction::GetElementPtr: {
    GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(I1);
    GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(I2);

    SmallVector<Value *, 8> Indices1(GEP1->idx_begin(), GEP1->idx_end());
    SmallVector<Value *, 8> Indices2(GEP2->idx_begin(), GEP2->idx_end());
    if (Indices1.size() != Indices2.size())
      return false;

    /*
    //TODO: some indices must be constant depending on the type being indexed.
    //For simplicity, whenever a given index is constant, keep it constant.
    //This simplification may degrade the merging quality.
    for (unsigned i = 0; i < Indices1.size(); i++) {
      if (isa<ConstantInt>(Indices1[i]) && isa<ConstantInt>(Indices2[i]) && Indices1[i] != Indices2[i])
        return false; // if different constant values
    }
    */

    Type *AggTy1 = GEP1->getSourceElementType();
    Type *AggTy2 = GEP2->getSourceElementType();

    unsigned CurIdx = 1;
    for (; CurIdx != Indices1.size(); ++CurIdx) {
      CompositeType *CTy1 = dyn_cast<CompositeType>(AggTy1);
      CompositeType *CTy2 = dyn_cast<CompositeType>(AggTy2);
      if (!CTy1 || CTy1->isPointerTy()) return false;
      if (!CTy2 || CTy2->isPointerTy()) return false;
      Value *Idx1 = Indices1[CurIdx];
      Value *Idx2 = Indices2[CurIdx];
      //if (!CT->indexValid(Index)) return nullptr;
      
      //validate indices
      if (isa<StructType>(CTy1) || isa<StructType>(CTy2)) {
        //if types are structs, the indices must be and remain constants
        if (!isa<ConstantInt>(Idx1) || !isa<ConstantInt>(Idx2)) return false;
        if (Idx1!=Idx2) return false;
      }

      AggTy1 = CTy1->getTypeAtIndex(Idx1);
      AggTy2 = CTy2->getTypeAtIndex(Idx2);

      //sanity check: matching indexed types
      bool sameType = (AggTy1 == AggTy2);
      if (!IdenticalType) {
        const DataLayout *DT =
          &((Module *)GEP1->getParent()->getParent()->getParent())->getDataLayout();
        sameType = isEquivalentType(AggTy1, AggTy2, DT);
      }
      if (!sameType) return false;
    }
   
    break;
  }
  case Instruction::Switch: {
    SwitchInst *SI1 = dyn_cast<SwitchInst>(I1);
    SwitchInst *SI2 = dyn_cast<SwitchInst>(I2);
    if (SI1->getNumCases() == SI2->getNumCases()) {
      auto CaseIt1 = SI1->case_begin(), CaseEnd1 = SI1->case_end();
      auto CaseIt2 = SI2->case_begin(), CaseEnd2 = SI2->case_end();
      do {
        auto *Case1 = &*CaseIt1;
        auto *Case2 = &*CaseIt2;
        if (Case1 != Case2)
          return false; // TODO: could allow permutation!
        ++CaseIt1;
        ++CaseIt2;
      } while (CaseIt1 != CaseEnd1 && CaseIt2 != CaseEnd2);
      return true;
    }
    return false;
  }
  case Instruction::Call: {
    CallInst *CI1 = dyn_cast<CallInst>(I1);
    CallInst *CI2 = dyn_cast<CallInst>(I2);
    if (CI1->isInlineAsm() || CI2->isInlineAsm())
      return false;
    if (CI1->getCalledFunction() != CI2->getCalledFunction())
      return false;
    if (Function *F = CI1->getCalledFunction()) {
      if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {

        if (!matchIntrinsicCalls(ID, CI1, CI2))
          return false;
      }
    }

    return CI1->getCallingConv() ==
           CI2->getCallingConv(); // &&
                                  // CI->getAttributes() ==
                                  // cast<CallInst>(I2)->getAttributes();
  }
  case Instruction::Invoke: {
    InvokeInst *CI1 = dyn_cast<InvokeInst>(I1);
    InvokeInst *CI2 = dyn_cast<InvokeInst>(I2);
    return CI1->getCallingConv() == CI2->getCallingConv() &&
           matchLandingPad(CI1->getLandingPadInst(), CI2->getLandingPadInst());
    // CI->getAttributes() == cast<InvokeInst>(I2)->getAttributes();
  }
  case Instruction::InsertValue: {
    const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1);
    return IVI->getIndices() == cast<InsertValueInst>(I2)->getIndices();
  }
  case Instruction::ExtractValue: {
    const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1);
    return EVI->getIndices() == cast<ExtractValueInst>(I2)->getIndices();
  }
  case Instruction::Fence: {
    const FenceInst *FI = dyn_cast<FenceInst>(I1);
    return FI->getOrdering() == cast<FenceInst>(I2)->getOrdering() &&
           FI->getSyncScopeID() == cast<FenceInst>(I2)->getSyncScopeID();
  }
  case Instruction::AtomicCmpXchg: {
    const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I1);
    const AtomicCmpXchgInst *CXI2 = cast<AtomicCmpXchgInst>(I2);
    return CXI->isVolatile() == CXI2->isVolatile() &&
           CXI->isWeak() == CXI2->isWeak() &&
           CXI->getSuccessOrdering() == CXI2->getSuccessOrdering() &&
           CXI->getFailureOrdering() == CXI2->getFailureOrdering() &&
           CXI->getSyncScopeID() == CXI2->getSyncScopeID();
  }
  case Instruction::AtomicRMW: {
    const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I1);
    return RMWI->getOperation() == cast<AtomicRMWInst>(I2)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I2)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I2)->getOrdering() &&
           RMWI->getSyncScopeID() == cast<AtomicRMWInst>(I2)->getSyncScopeID();
  }
  default:
    if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
      return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  }

  return true;
}

static bool match(Value *V1, Value *V2) {
  if (isa<Instruction>(V1) && isa<Instruction>(V2)) {
    return match(dyn_cast<Instruction>(V1), dyn_cast<Instruction>(V2));
  } else if (isa<BasicBlock>(V1) && isa<BasicBlock>(V2)) {
    BasicBlock *BB1 = dyn_cast<BasicBlock>(V1);
    BasicBlock *BB2 = dyn_cast<BasicBlock>(V2);
    if (BB1->isLandingPad() || BB2->isLandingPad()) {
      LandingPadInst *LP1 = BB1->getLandingPadInst();
      LandingPadInst *LP2 = BB2->getLandingPadInst();
      if (LP1 == nullptr || LP2 == nullptr)
        return false;
      return matchLandingPad(LP1, LP2);
    } else return true;
  }
  return false;
}

static bool match(std::vector<Value *> &F1, std::vector<Value *> &F2,
                  unsigned i, unsigned j) {
  return match(F1[i], F2[j]);
}

static bool match(SmallVectorImpl<Value *> &F1, SmallVectorImpl<Value *> &F2,
                  unsigned i, unsigned j) {
  return match(F1[i], F2[j]);
}


  NeedlemanWunschSimilarityMatrix::NeedlemanWunschSimilarityMatrix(SmallVectorImpl<Value *> &F1,
                                  SmallVectorImpl<Value *> &F2)
      : F1(F1), F2(F2) {
    NumRows = F1.size() + 1;             // rows
    NumCols = F2.size() + 1;             // cols
    Matrix = new int[NumRows * NumCols]; // last element keeps the max
    Match = new bool[F1.size()*F2.size()]; // last element keeps the max
                                         // value
    // memset(Matrix,0, sizeof(int)*NumRows*NumCols);
    // for (unsigned i = 0; i < NumRows; i++)
    //  for (unsigned j = 0; j < NumCols; j++)
    //    Matrix[i * NumCols + j] = 0;

    #pragma omp parallel for
    for (unsigned i = 0; i < F1.size(); i++)
      for (unsigned j = 0; j < F2.size(); j++)
        Match[i * F2.size() + j] = match(F1[i], F2[j]);
    
    for (unsigned i = 0; i < NumRows; i++)
      Matrix[i * NumCols + 0] = i * gapPenalty;
    for (unsigned j = 0; j < NumCols; j++)
      Matrix[0 * NumCols + j] = j * gapPenalty;

    MaxScore = 0;
    MaxRow = 0;
    MaxCol = 0;

    /*
    unsigned blockIncr = 32;
    for (unsigned i = 1; i < NumRows; i++) {
      for (unsigned j = 1; j < NumCols; j++) {
        unsigned limitRow = std::min( i + blockIncr, NumRows );
        unsigned limitRow = std::min( i + blockIncr, NumRows );
        for (unsigned xi = i; xi < limitRow; xi++) {
        int score = calcScore(i, j);
        Matrix[i * NumCols + j] = score;
      }
    }
    */
    const unsigned blockIncr = 64;
    for (unsigned i = 1; i < NumRows; i += blockIncr) {
      for (unsigned j = 1; j < NumCols; j += blockIncr) {
        unsigned limitRows = std::min(i + blockIncr, NumRows);
        unsigned limitCols = std::min(j + blockIncr, NumCols);
        for (unsigned xi = i; xi < limitRows; xi++) {
          // Matrix[ xi*NumCols + 0 ] = xi*gapPenalty;
          for (unsigned xj = j; xj < limitCols; xj++) {
            // int score = calcScore(xi, xj);

            int similarity =
                Match[(xi - 1)*F2.size() + (xj - 1)] ? matchAward : mismatchPenalty;
            //    match(F1[xi - 1], F2[xj - 1]) ? matchAward : mismatchPenalty;
            int diagonal = Matrix[(xi - 1) * NumCols + xj - 1] + similarity;
            int upper = Matrix[(xi - 1) * NumCols + xj] + gapPenalty;
            int left = Matrix[xi * NumCols + xj - 1] + gapPenalty;
            int score = std::max(std::max(diagonal, upper), left);

            Matrix[xi * NumCols + xj] = score;
          }
        }
      }
    }

    MaxRow = NumRows - 1;
    MaxCol = NumCols - 1;
  }

  NeedlemanWunschSimilarityMatrix::~NeedlemanWunschSimilarityMatrix() {
    if (Matrix)
      delete Matrix;
    Matrix = nullptr;
    if (Match)
      delete Match;
    Match = nullptr;
  }

  int NeedlemanWunschSimilarityMatrix::calcScore(unsigned i, unsigned j) {
    //int similarity = match(F1[i - 1], F2[j - 1]) ? matchAward : mismatchPenalty;
    int similarity = Match[(i - 1)*F2.size() + (j - 1)] ? matchAward : mismatchPenalty;
    int diagonal = Matrix[(i - 1) * NumCols + j - 1] + similarity;
    int upper = Matrix[(i - 1) * NumCols + j] + gapPenalty;
    int left = Matrix[i * NumCols + j - 1] + gapPenalty;
    return std::max(std::max(diagonal, upper), left);
  }

  unsigned NeedlemanWunschSimilarityMatrix::nextMove(unsigned i, unsigned j) {
    if (i <= 0 || j <= 0)
      return END;
    int diagonal = Matrix[(i - 1) * NumCols + j - 1];
    int upper = Matrix[(i - 1) * NumCols + j];
    int left = Matrix[i * NumCols + j - 1];
    if (diagonal >= upper && diagonal >= left)
      return diagonal > 0 ? DIAGONAL : END; // Diagonal Move
    else if (upper > diagonal && upper >= left)
      return upper > 0 ? UP : END; // Up move
    else if (left > diagonal && left >= upper)
      return left > 0 ? LEFT : END; // Left move
    return END;
  }

class SearchItem {
public:
  unsigned row;
  unsigned col;
  int val;
  SearchItem(unsigned row, unsigned col, int val)
      : row(row), col(col), val(val) {}
  bool operator<(const SearchItem &SI) const { return val < SI.val; }
  bool operator>(const SearchItem &SI) const { return val > SI.val; }
};

class AStarSimilarityMatrix {
public:
  SmallVectorImpl<Value *> &F1;
  SmallVectorImpl<Value *> &F2;

  int *Matrix;
  unsigned NumRows;
  unsigned NumCols;
  unsigned MaxScore;
  unsigned MaxRow;
  unsigned MaxCol;

  const static unsigned END = 0;
  const static unsigned DIAGONAL = 1;
  const static unsigned UP = 2;
  const static unsigned LEFT = 3;

  // scoring scheme
  const int matchAward = 2;
  const int mismatchPenalty = -1;
  const int gapPenalty = -1;

  AStarSimilarityMatrix(SmallVectorImpl<Value *> &F1,
                        SmallVectorImpl<Value *> &F2)
      : F1(F1), F2(F2) {
    const unsigned N1 = F1.size();
    const unsigned N2 = F2.size();
    NumRows = N1 + 1;                    // rows
    NumCols = N2 + 1;                    // cols
    Matrix = new int[NumRows * NumCols]; // last element keeps the max
                                         // value
    // memset(Matrix,INT_MIN, sizeof(int)*NumRows*NumCols);
    for (unsigned i = 0; i < (NumRows * NumCols); i++)
      Matrix[i] = INT_MIN;
    for (unsigned i = 0; i < NumRows; i++)
      Matrix[i * NumCols + 0] = i * gapPenalty;
    for (unsigned j = 0; j < NumCols; j++)
      Matrix[0 * NumCols + j] = j * gapPenalty;

    std::priority_queue<SearchItem> q;

    {
      int similarity = match(F1[0], F2[0]) ? matchAward : mismatchPenalty;
      int diagonal = Matrix[0 * NumCols + 0] + similarity;
      int upper = Matrix[0 * NumCols + 1] + gapPenalty;
      int left = Matrix[1 * NumCols + 0] + gapPenalty;
      int score = std::max(std::max(diagonal, upper), left);
      Matrix[1 * NumCols + 1] = score;
      q.push(SearchItem(1, 1, score));
    }

    while (true) {
      SearchItem Item = q.top();
      q.pop();

      if (Item.row == N1 && Item.col == N2)
        break;

      unsigned i, j;

      // neighbor right
      i = Item.row;
      j = Item.col + 1;
      if (j < NumCols) {
        if (Matrix[i * NumCols + j] == INT_MIN) {
          int similarity =
              match(F1[i - 1], F2[j - 1]) ? matchAward : mismatchPenalty;
          int val;
          val = Matrix[(i - 1) * NumCols + j - 1];
          int diagonal = (val == INT_MIN) ? INT_MIN : (val + similarity);
          val = Matrix[(i - 1) * NumCols + j];
          int upper = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          val = Matrix[i * NumCols + j - 1];
          int left = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          int score = std::max(std::max(diagonal, upper), left);
          Matrix[i * NumCols + j] = score;
          if (i == N1 && j == N2)
            break;
          q.push(SearchItem(i, j, score));
        }
      }

      // neighbor down
      i = Item.row + 1;
      j = Item.col;
      if (i < NumRows) {
        if (Matrix[i * NumCols + j] == INT_MIN) {
          int similarity =
              match(F1[i - 1], F2[j - 1]) ? matchAward : mismatchPenalty;
          int val;
          val = Matrix[(i - 1) * NumCols + j - 1];
          int diagonal = (val == INT_MIN) ? INT_MIN : (val + similarity);
          val = Matrix[(i - 1) * NumCols + j];
          int upper = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          val = Matrix[i * NumCols + j - 1];
          int left = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          int score = std::max(std::max(diagonal, upper), left);
          Matrix[i * NumCols + j] = score;
          if (i == N1 && j == N2)
            break;
          q.push(SearchItem(i, j, score));
        }
      }

      // neighbor diagonal
      i = Item.row + 1;
      j = Item.col + 1;
      if (i < NumRows && j < NumCols) {
        if (Matrix[i * NumCols + j] == INT_MIN) {
          int similarity =
              match(F1[i - 1], F2[j - 1]) ? matchAward : mismatchPenalty;
          int val;
          val = Matrix[(i - 1) * NumCols + j - 1];
          int diagonal = (val == INT_MIN) ? INT_MIN : (val + similarity);
          val = Matrix[(i - 1) * NumCols + j];
          int upper = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          val = Matrix[i * NumCols + j - 1];
          int left = (val == INT_MIN) ? INT_MIN : (val + gapPenalty);
          int score = std::max(std::max(diagonal, upper), left);
          Matrix[i * NumCols + j] = score;
          if (i == N1 && j == N2)
            break;
          q.push(SearchItem(i, j, score));
        }
      }
    }

    MaxRow = NumRows - 1;
    MaxCol = NumCols - 1;
    MaxScore = Matrix[MaxRow * NumCols + MaxCol];
  }

  ~AStarSimilarityMatrix() {
    if (Matrix)
      delete Matrix;
    Matrix = nullptr;
  }

  int calcScore(unsigned i, unsigned j) {
    int similarity = match(F1[i - 1], F2[j - 1]) ? matchAward : mismatchPenalty;
    int diagonal = Matrix[(i - 1) * NumCols + j - 1] + similarity;
    int upper = Matrix[(i - 1) * NumCols + j] + gapPenalty;
    int left = Matrix[i * NumCols + j - 1] + gapPenalty;
    return std::max(std::max(diagonal, upper), left);
  }

  unsigned nextMove(unsigned i, unsigned j) {
    if (i <= 0 || j <= 0)
      return END;
    int diagonal = Matrix[(i - 1) * NumCols + j - 1];
    int upper = Matrix[(i - 1) * NumCols + j];
    int left = Matrix[i * NumCols + j - 1];
    if (diagonal >= upper && diagonal >= left)
      return diagonal > 0 ? DIAGONAL : END; // Diagonal Move
    else if (upper > diagonal && upper >= left)
      return upper > 0 ? UP : END; // Up move
    else if (left > diagonal && left >= upper)
      return left > 0 ? LEFT : END; // Left move
    return END;
  }
};

static unsigned
RandomLinearizationOfBlocks(BasicBlock *BB,
                            std::list<BasicBlock *> &OrederedBBs,
                            std::set<BasicBlock *> &Visited) {
  if (Visited.find(BB) != Visited.end())
    return 0;
  Visited.insert(BB);

  TerminatorInst *TI = BB->getTerminator();

  std::vector<BasicBlock *> NextBBs;
  for (unsigned i = 0; i < TI->getNumSuccessors(); i++) {
    NextBBs.push_back(TI->getSuccessor(i));
  }
  std::random_shuffle(NextBBs.begin(), NextBBs.end());

  unsigned SumSizes = 0;
  for (BasicBlock *NextBlock : NextBBs) {
    SumSizes += RandomLinearizationOfBlocks(NextBlock, OrederedBBs, Visited);
  }

  OrederedBBs.push_front(BB);
  return SumSizes + BB->size();
}

static unsigned
RandomLinearizationOfBlocks(Function *F, std::list<BasicBlock *> &OrederedBBs) {
  std::set<BasicBlock *> Visited;
  return RandomLinearizationOfBlocks(&F->getEntryBlock(), OrederedBBs, Visited);
}

static unsigned
CanonicalLinearizationOfBlocks(BasicBlock *BB,
                               std::list<BasicBlock *> &OrederedBBs,
                               std::set<BasicBlock *> &Visited) {
  if (Visited.find(BB) != Visited.end())
    return 0;
  Visited.insert(BB);

  TerminatorInst *TI = BB->getTerminator();

  unsigned SumSizes = 0;
  for (unsigned i = 0; i < TI->getNumSuccessors(); i++) {
    SumSizes += CanonicalLinearizationOfBlocks(TI->getSuccessor(i), OrederedBBs,
                                               Visited);
  }

  OrederedBBs.push_front(BB);
  return SumSizes + BB->size();
}

//static unsigned
//FingerprintLinearizationOfBlocks(Function *F, std::list<BasicBlock *> &OrderedBBs, ) {
//    
//}
//
////OrderedBBs in this case comes as predetermined and you will insert a new BB in the list trying to be a successor of the previous element and ideally also a predecessor of the next one.
////I think visited shouldn't contain nodes when u call it for the first time in order to
//
//
//static unsigned
//FingerprintLinearizationOfBlocks(BasicBlock *BB,
//                               std::list<BasicBlock *> &OrederedBBs,
//                               std::set<BasicBlock *> &Visited, 
//                               std::set<BasicBlock *> SetOfPreordered,
//                               bool AntecessorIsPreordered) {
//  if (Visited.find(BB) != Visited.end())
//    return 0;
//  Visited.insert(BB);
//
//  bbIsPreorderedpreordered = SetOfPreordered.find(BB) == SetOfPreordered.end();
//
//  TerminatorInst *TI = BB->getTerminator();
//
//  unsigned SumSizes = 0;
//  for (unsigned i = 0; i < TI->getNumSuccessors(); i++) {
//    SumSizes += FingerprintLinearizationOfBlocks(TI->getSuccessor(i), OrederedBBs_F, Visited, SetOfPreordered, bbIsPreordered);
//  }
//
//  if (not bbIsPreordered) {
//    // you just want to continue linearizing other functions than this one.
//    //you can just insert it after the antecessor if the antecessor is in the set of preordered
//    OrederedBBs.push_front(BB);
//  }
//
//  return SumSizes + BB->size();
//}

static unsigned
CanonicalLinearizationOfBlocks(Function *F,
                               std::list<BasicBlock *> &OrederedBBs) {
  std::set<BasicBlock *> Visited;
  return CanonicalLinearizationOfBlocks(&F->getEntryBlock(), OrederedBBs,
                                        Visited);
}

enum LinearizationKind { LK_Random, LK_Canonical };


static void CreateFVec(std::vector<BasicBlock *> &OrderedBBs, SmallVectorImpl<Value *> &FVec, unsigned numInstsInF) {
  FVec.reserve(numInstsInF + OrderedBBs.size());
  for (BasicBlock *BB : OrderedBBs) {
    FVec.push_back(BB);
    for (Instruction &I : *BB) {
      if (!isa<LandingPadInst>(&I))
        FVec.push_back(&I);
    }
  }
}

static void CreateFVec(std::list<BasicBlock *> &OrderedBBs, SmallVectorImpl<Value *> &FVec, unsigned numInstsInF) {
  FVec.reserve(numInstsInF + OrderedBBs.size());
  for (BasicBlock *BB : OrderedBBs) {
    FVec.push_back(BB);
    for (Instruction &I : *BB) {
      if (!isa<LandingPadInst>(&I))
        FVec.push_back(&I);
    }
  }
}

static void Linearization(Function *F, SmallVectorImpl<Value *> &FVec, 
                          std::list<BasicBlock *> &OrderedBBs, LinearizationKind LK) {

  unsigned FReserve = 0;
  switch (LK) {
  case LinearizationKind::LK_Random:
    FReserve = RandomLinearizationOfBlocks(F, OrderedBBs);
  case LinearizationKind::LK_Canonical:
  default:
    FReserve = CanonicalLinearizationOfBlocks(F, OrderedBBs);
  }
  CreateFVec(OrderedBBs, FVec, FReserve);
}

class MergedFunction {
public:
  Function *F1;
  Function *F2;
  Function *MergedFunc;
  bool hasFuncIdArg;
  double RoughReduction;

  std::map<unsigned, unsigned> ParamMap1;
  std::map<unsigned, unsigned> ParamMap2;

  MergedFunction(Function *F1, Function *F2, Function *MergedFunc)
      : F1(F1), F2(F2), MergedFunc(MergedFunc), hasFuncIdArg(false) {}
};

static bool validMergeTypes(Function *F1, Function *F2) {
  bool EquivTypes = isEquivalentType(F1->getReturnType(), F2->getReturnType(),
                                     &F1->getParent()->getDataLayout());
  if (!EquivTypes) {
    if (!F1->getReturnType()->isVoidTy() && !F2->getReturnType()->isVoidTy()) {
      return false;
    }
  }

  return true;
}

struct SelectCacheEntry {
public:
  Value *Cond;
  Value *ValTrue;
  Value *ValFalse;
  BasicBlock *Block;

  SelectCacheEntry(Value *C, Value *V1, Value *V2, BasicBlock *BB)
      : Cond(C), ValTrue(V1), ValFalse(V2), Block(BB) {}

  bool operator<(const SelectCacheEntry &Other) const {
    if (Cond != Other.Cond)
      return Cond < Other.Cond;
    if (ValTrue != Other.ValTrue)
      return ValTrue < Other.ValTrue;
    if (ValFalse != Other.ValFalse)
      return ValFalse < Other.ValFalse;
    if (Block != Other.Block)
      return Block < Other.Block;
    return false;
  }
};

//static unsigned countNumInstsInF(std::list<BasicBlock *> &OrderedBBs) {
//  unsigned numInsts = 0;
//  for (auto BB : OrderedBBs) {
//    for (auto II : BB) {
//      numInsts += 1;
//    }
//  }
//  return numInsts;
//}



static double
AlignLinearizedCFGs(SmallVectorImpl<Value *> &F1Vec,
                    SmallVectorImpl<Value *> &F2Vec,
                    std::list<std::pair<Value *, Value *>> &AlignedInstsList) {
  int countMerges = 0;

  bool WasMerge = true;
  int EstimatedSize = 0;

  NeedlemanWunschSimilarityMatrix SimMat(F1Vec, F2Vec);
  // AStarSimilarityMatrix SimMat(F1Vec, F2Vec);
  // std::list< std::pair<Value *, Value *> > AlignedInstsList;

  if (FuncConcatMode) {
    for (unsigned i = SimMat.NumRows - 1; i > 0; i--) {
      AlignedInstsList.push_front(
          std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
      WasMerge = false;
      EstimatedSize++;
    }
    for (unsigned j = SimMat.NumCols - 1; j > 0; j--) {
      AlignedInstsList.push_front(
          std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
      WasMerge = false;
      EstimatedSize++;
    }
    double EstimatedReduction = 0;
    return EstimatedReduction;
  } else {

  for (unsigned i = SimMat.NumRows - 1; i > SimMat.MaxRow; i--) {
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
    WasMerge = false;
    EstimatedSize++;
  }
  for (unsigned j = SimMat.NumCols - 1; j > SimMat.MaxCol; j--) {
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
    WasMerge = false;
    EstimatedSize++;
  }

  unsigned i = SimMat.MaxRow, j = SimMat.MaxCol;

  if (match(F1Vec, F2Vec, i - 1, j - 1)) {
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(F1Vec[i - 1], F2Vec[j - 1]));
    countMerges++;
    if (!WasMerge)
      EstimatedSize += 2;
    EstimatedSize++;
    WasMerge = true;
  } else {
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
    WasMerge = false;
    EstimatedSize += 2;
  }

  unsigned move = SimMat.nextMove(i, j);
  while (move != SimMat.END) {
    switch (move) {
    case SimMat.DIAGONAL:
      i--;
      j--;
      //if (match(F1Vec, F2Vec, i - 1, j - 1)) {
      if (SimMat.Match[ (i-1)*F2Vec.size() + (j - 1)]) {
        AlignedInstsList.push_front(
            std::pair<Value *, Value *>(F1Vec[i - 1], F2Vec[j - 1]));
        countMerges++;
        if (!WasMerge)
          EstimatedSize += 2;
        EstimatedSize++;
        WasMerge = true;
      } else {
        AlignedInstsList.push_front(
            std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
        AlignedInstsList.push_front(
            std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
        if (WasMerge)
          EstimatedSize++;
        EstimatedSize += 2;
        WasMerge = false;
      }
      break;
    case SimMat.UP:
      i--;
      AlignedInstsList.push_front(
          std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
      if (WasMerge)
        EstimatedSize += 2;
      EstimatedSize++;
      WasMerge = false;
      break;
    case SimMat.LEFT:
      j--;
      AlignedInstsList.push_front(
          std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
      if (WasMerge)
        EstimatedSize += 2;
      EstimatedSize++;
      WasMerge = false;
      break;
    default:
      break;
    }
    move = SimMat.nextMove(i, j);
  }

  while (i > 1) {
    i--;
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(F1Vec[i - 1], nullptr));
    EstimatedSize++;
  }
  while (j > 1) {
    j--;
    AlignedInstsList.push_front(
        std::pair<Value *, Value *>(nullptr, F2Vec[j - 1]));
    EstimatedSize++;
  }

  // std::vector< std::pair<Value *, Value *> > AlignedInsts(
  //    AlignedInstsList.begin(), AlignedInstsList.end());

  double EstimatedReduction =
      ((1 - ((double)EstimatedSize) / (F1Vec.size() + F2Vec.size())) * 100);
  // errs() << "Rough Size: " << EstimatedSize << " F1 Size: " << F1Vec.size()
  // << " F2 Size: " << F2Vec.size() << " : " << EstimatedReduction << "%\n";
  errs() << "Estimated size is " << EstimatedSize << " and the sum of the inputs is " << (F1Vec.size() + F2Vec.size());
    return EstimatedReduction;
  }
}


static std::vector<std::pair<unsigned, unsigned> > GetRandomIdxPairs(unsigned numBBs, unsigned numRandomSwappings) {
    unsigned maxRandomSwappings = std::min(numRandomSwappings, numBBs/2);
    std::vector<unsigned> idxs(numBBs);
    for (unsigned i = 0; i < numBBs; ++i)
        idxs[i] = i;

    unsigned seed = 0;
    std::shuffle(idxs.begin(), idxs.end(), std::default_random_engine(seed));
    
    std::vector<std::pair<unsigned, unsigned> > pairsArr;
    for (unsigned i = 0; i < maxRandomSwappings; ++i) {
        pairsArr.push_back(std::pair<unsigned, unsigned>(idxs[i], idxs[i+1]));
    }
    return pairsArr;
}

static double
AlignLinearizedCFGsRandomShufflings(SmallVectorImpl<Value *> &F1Vec,
                    SmallVectorImpl<Value *> &F2Vec,
                    std::vector<BasicBlock *> &OrderedBBs1,
                    std::vector<BasicBlock *> &OrderedBBs2,
                    std::list<std::pair<Value *, Value *>> &AlignedInstsList) {
  
  std::list<std::pair<Value *, Value *>> BestAlignedInstsList;
  double bestSizeReduction = AlignLinearizedCFGs(F1Vec, F2Vec, BestAlignedInstsList);

  int numRandomSwappings = 10;
  unsigned numInstsAndBBsF1 = F1Vec.size();
  unsigned numBBsF1 = OrderedBBs1.size();
  std::vector<std::pair<unsigned, unsigned> > randIdxPairs = GetRandomIdxPairs(numBBsF1, numRandomSwappings);
  for (auto &f1IdxPair : randIdxPairs) {
    auto BB1A = OrderedBBs1[f1IdxPair.first];
    OrderedBBs1[f1IdxPair.first] = OrderedBBs1[f1IdxPair.second];
    OrderedBBs1[f1IdxPair.second] = BB1A; //swapping linearized basic blocks
    SmallVector<Value *, 64> F1VecNew;
    CreateFVec(OrderedBBs1, F1VecNew, numInstsAndBBsF1-numBBsF1);
    std::list<std::pair<Value *, Value *>> NewAlignedInstsList;
    double sizeReduction = AlignLinearizedCFGs(F1VecNew, F2Vec, NewAlignedInstsList);
    if (sizeReduction > bestSizeReduction) {
      errs() << "Improvement in size reduction occurs at linearization time from " << bestSizeReduction << 
                " to " << sizeReduction << "\n";
      errs() << "Swapping bbs " << f1IdxPair.first << " and " << f1IdxPair.second << "\n";
      bestSizeReduction = sizeReduction;
      BestAlignedInstsList = NewAlignedInstsList; //I hope this is a complete copy
    }
  }
  AlignedInstsList = BestAlignedInstsList; 
  return bestSizeReduction;
}

void optimizeAlignment(std::list<std::pair<Value *, Value *>> &AlignedInsts) {
  if (AlignedInsts.size()==0) return;

  std::list<std::pair<Value *, Value *>> NewAlignedInsts;

  auto It = AlignedInsts.begin();
  auto NextIt = AlignedInsts.begin();
  NextIt++;

  int countAlignedSize = 0;
  while (NextIt!=AlignedInsts.end()) {
    if (It->first!=nullptr && It->second!=nullptr) {
      countAlignedSize++;
      if ((NextIt->first==nullptr || NextIt->second==nullptr) && countAlignedSize==1 ) {
        NewAlignedInsts.push_back( std::pair<Value*,Value*>(It->first, nullptr) );
        NewAlignedInsts.push_back( std::pair<Value*,Value*>(nullptr, It->second) );
        countAlignedSize = 0;
      } else {
        NewAlignedInsts.push_back( std::pair<Value*,Value*>(*It) );
      }
    } else {
      countAlignedSize = 0;
      NewAlignedInsts.push_back( std::pair<Value*,Value*>(*It) );
    }
    It++;
    NextIt++;
  }
  NewAlignedInsts.push_back( std::pair<Value*,Value*>(*It) );
  AlignedInsts = NewAlignedInsts;
}


#ifdef TIME_STEPS_DEBUG
Timer TimeAlign("Merge::Align", "Merge::Align");
Timer TimeParam("Merge::Param", "Merge::Param");
Timer TimeCodeGen1("Merge::CodeGen1", "Merge::CodeGen1");
Timer TimeCodeGen2("Merge::CodeGen2", "Merge::CodeGen2");
Timer TimeCodeGenFix("Merge::CodeGenFix", "Merge::CodeGenFix");
#endif

bool filterMergePair(Function *F1, Function *F2) {
  if (!F1->getSection().equals(F2->getSection())) return true;

  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 != PersonalityFn2) return true;
  }

  return false;
}

static void getAllUsesInfoMap(std::map<BasicBlock *, uint64_t> &bbWithWeightsPointing, Function *F) {
    for (auto &BB : *F) {
        TerminatorInst *TI = BB.getTerminator();

        BranchInst *BI;
        if ((BI = dyn_cast<BranchInst>(TI)) != nullptr and BI->isConditional()) {
            BasicBlock *BBT = BI->getSuccessor(0);
            BasicBlock *BBF = BI->getSuccessor(1);
            uint64_t trueWeight, falseWeight;
            if (BI->extractProfMetadata(trueWeight, falseWeight)) {
                if (bbWithWeightsPointing.find(BBT) == bbWithWeightsPointing.end())
                    bbWithWeightsPointing[BBT] = 0;
                if (bbWithWeightsPointing.find(BBF) == bbWithWeightsPointing.end())
                    bbWithWeightsPointing[BBF] = 0;
                
                bbWithWeightsPointing[BBT] += trueWeight;
                bbWithWeightsPointing[BBF] += falseWeight;
            }
        }
    }
}

/// Get Weights of a given terminator, the default weight is at the front
/// of the vector. If TI is a conditional eq, we need to swap the branch-weight
/// metadata.
static void GetBranchWeights(Instruction *TI,
                             std::vector<uint32_t> &Weights) {
  MDNode *MD = TI->getMetadata(LLVMContext::MD_prof);
  assert(MD);
  for (unsigned i = 1, e = MD->getNumOperands(); i < e; ++i) {
    ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(i));
    Weights.push_back(CI->getValue().getZExtValue());
  }

  // If TI is a conditional eq, the default case is the false case,
  // and the corresponding branch-weight data is at index 2. We swap the
  // default weight to be the first entry.
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    ICmpInst *ICI;
    if (Weights.size() == 2 and BI->isConditional() and (ICI = dyn_cast<ICmpInst>(BI->getCondition()))) {
      if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
        std::swap(Weights.front(), Weights.back());
    }
  }
}


static void setDynamicProfiling(BasicBlock *MergedResBB, BasicBlock *BB1, BasicBlock *BB2) {
    

    if (not FMSALegacyPass::bbProfiles.empty()) {
        std::string f1Name = GetValueName(BB1->getParent());
        std::string bb1Name = f1Name + "::" + BB1->getName().str();
        std::string f2Name = GetValueName(BB2->getParent());
        std::string bb2Name = f2Name + "::" + BB2->getName().str();
        double mergedResProfile = 0.0;
        
        if (FMSALegacyPass::bbProfiles.find(bb1Name) != FMSALegacyPass::bbProfiles.end()) {
            mergedResProfile = FMSALegacyPass::bbProfiles[bb1Name];
        }

        if (FMSALegacyPass::bbProfiles.find(bb2Name) != FMSALegacyPass::bbProfiles.end()) {
            mergedResProfile += FMSALegacyPass::bbProfiles[bb2Name];
        }

        FMSALegacyPass::bbProfiles[GetValueName(MergedResBB->getParent()) + "::" + MergedResBB->getName().str()] = mergedResProfile;
        FMSALegacyPass::mergedBBProfiles[MergedResBB] = mergedResProfile;
    }
}

static uint64_t getPrevWeights(BasicBlock *BB) {
	uint64_t totalWeight = 0;
    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
    	BasicBlock *PredBB = *PI; 
	    TerminatorInst *TI = PredBB->getTerminator();
	    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
	        for (int i=0; i < BI->getNumSuccessors(); ++i) {
	  		    if (BI->getSuccessor(i) == BB) {
    			    std::vector<uint32_t> BBWeights;
                    if (TI->hasMetadata() and TI->getMetadata(LLVMContext::MD_prof)) {
	  	                GetBranchWeights(TI, BBWeights);
	  			        totalWeight += BBWeights[i];
                    }
	  			    break;
                }
            }
	    }

	    //TODO: for each predecessor, you still need to find out through which edge the two bbs are connected
    }
	return totalWeight; 
}

void getKindOfTerminator(std::vector<uint32_t> &BBWeights, BasicBlock* BB) {
    bool ends_in_conditional_b = false;
    bool ends_in_unconditional_b = false;
    
    if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
        if (BI->isConditional())
            ends_in_conditional_b = true;
        else if (BI->isUnconditional())
            ends_in_unconditional_b = true;
    }
    TerminatorInst *TI = BB->getTerminator();
    if (TI->hasMetadata() and TI->getMetadata(LLVMContext::MD_prof)) {
      GetBranchWeights(BB->getTerminator(), BBWeights);
    }
    else if (ends_in_unconditional_b)
      BBWeights.push_back(0);
    else if (ends_in_conditional_b) {
      BBWeights.push_back(0);
      BBWeights.push_back(0);
    }
}

static MergedFunction
mergeBySequenceAlignmentBackend(Function *F1, Function *F2,
                                std::list<std::pair<Value *, Value *>> &AlignedInsts,
                                double RoughReduction) {
  //optimizeAlignment(AlignedInsts);

  LLVMContext &Context = F1->getContext();
  const DataLayout *DL = &F1->getParent()->getDataLayout();
  Type *IntPtrTy = DL ? DL->getIntPtrType(Context) : NULL;

  MergedFunction ErrorResponse(F1, F2, nullptr);

#ifdef ENABLE_DEBUG_CODE
  if (Verbose) {
    for (auto Pair : AlignedInsts) {

      if (Pair.first != nullptr && Pair.second != nullptr) {

        errs() << "1: ";
        if (isa<BasicBlock>(Pair.first))
          errs() << "BB " << GetValueName(Pair.first) << "\n";
        else
          Pair.first->dump();
        errs() << "2: ";
        if (isa<BasicBlock>(Pair.second))
          errs() << "BB " << GetValueName(Pair.second) << "\n";
        else
          Pair.second->dump();
        errs() << "----\n";

      } else {

        if (Pair.first) {
          errs() << "1: ";
          if (isa<BasicBlock>(Pair.first))
            errs() << "BB " << GetValueName(Pair.first) << "\n";
          else
            Pair.first->dump();
          errs() << "2: -\n";
        } else if (Pair.second) {
          errs() << "1: -\n";
          errs() << "2: ";
          if (isa<BasicBlock>(Pair.second))
            errs() << "BB " << GetValueName(Pair.second) << "\n";
          else
            Pair.second->dump();
        }
        errs() << "----\n";
      }
    }
  }
#endif

  // TODO: tmp
  // int alignmentCount = optimizeAlignment(AlignedInsts);

  /*
  if (alignmentCount==0 && (F1Vec.size() >= 5 || F2Vec.size() >= 5)) {
     if (Verbose) {
       errs() << "Not worthy threshold: very small similarty for a relatively
  large function\n";
     }
     return ErrorResponse;
  }

  if (PredFunc1.PredicatedInsts.size() <= 3 &&
  !canReplaceAllCalls(PredFunc1.getFunction())) { if (Verbose) { errs() << "Not
  worthy threshold: too small and not replaceble: " <<
  GetValueName(PredFunc1.getFunction()) << "\n";
     }
     return ErrorResponse;
  }
  if (PredFunc2.PredicatedInsts.size() <= 3 &&
  !canReplaceAllCalls(PredFunc2.getFunction())) { if (Verbose) { errs() << "Not
  worthy threshold: too small and not replaceble: " <<
  GetValueName(PredFunc2.getFunction()) << "\n";
     }
     return ErrorResponse;
  }
  */

#ifdef TIME_STEPS_DEBUG
  TimeParam.startTimer();
#endif

  // Merging parameters
  std::map<unsigned, unsigned> ParamMap1;
  std::map<unsigned, unsigned> ParamMap2;

  std::vector<Argument *> ArgsList1;
  for (Argument &arg : F1->args()) {
    ArgsList1.push_back(&arg);
  }

  // std::vector<Argument *> ArgsList2;
  // for (Argument &arg : F2->args()) {
  //  ArgsList2.push_back(&arg);
  //}

  std::vector<Type *> args;
  args.push_back(IntegerType::get(Context, 1)); // push the function Id argument
  unsigned paramId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    ParamMap1[paramId] = args.size();
    args.push_back((*I).getType());
    paramId++;
  }

  // merge arguments from Function2 with Function1
  paramId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {

    std::map<unsigned, int> MatchingScore;
    // first try to find an argument with the same name/type
    // otherwise try to match by type only
    for (unsigned i = 0; i < ArgsList1.size(); i++) {
      if (ArgsList1[i]->getType() == (*I).getType()) {

        bool hasConflict = false; // check for conflict from a previous matching
        for (auto ParamPair : ParamMap2) {
          if (ParamPair.second == ParamMap1[i]) {
            hasConflict = true;
            break;
          }
        }
        if (hasConflict)
          continue;

        MatchingScore[i] = 0;

        if (!MaxParamScore)
          break; // if not maximize score, get the first one
      }
    }

    // if ( MaxParamScore && MatchingScore.size() > 0) { //maximize scores
    if (MatchingScore.size() > 0) { // maximize scores
      for (auto Pair : AlignedInsts) {
        if (Pair.first != nullptr && Pair.second != nullptr) {
          auto *I1 = dyn_cast<Instruction>(Pair.first);
          auto *I2 = dyn_cast<Instruction>(Pair.second);
          if (I1 != nullptr && I2 != nullptr) { // test both for sanity
            for (unsigned i = 0; i < I1->getNumOperands(); i++) {
              for (auto KV : MatchingScore) {
                if (I1->getOperand(i) == ArgsList1[KV.first]) {
                  if (i < I2->getNumOperands() && I2->getOperand(i) == &(*I)) {
                    MatchingScore[KV.first]++;
                  }
                }
              }
            }
          }
        }
      }

      int MaxScore = -1;
      int MaxId = 0;

      for (auto KV : MatchingScore) {
        if (KV.second > MaxScore) {
          MaxScore = KV.second;
          MaxId = KV.first;
        }
      }

      // LastMaxParamScore = MaxScore;

      ParamMap2[paramId] = ParamMap1[MaxId];
    } else {
      ParamMap2[paramId] = args.size();
      args.push_back((*I).getType());
    }

    paramId++;
  }

  assert(validMergeTypes(F1, F2) &&
         "Return type must be the same or one of them must be void!");

  Type *RetType1 = F1->getReturnType();
  Type *RetType2 = F2->getReturnType();
  Type *ReturnType = RetType1;
  if (ReturnType->isVoidTy()) {
    ReturnType = RetType2;
  }

  ArrayRef<llvm::Type *> params(args);
  FunctionType *funcType = FunctionType::get(ReturnType, params, false);
  //static int merged_num = 0;
  std::string Name = "";
  //std::string Name = "merged" + std::to_string(merged_num);
  /*std::string Name = std::string("m.") +
                     std::string(PredFunc1.getFunction()->getName().str()) +
                     std::string(".") +
                     std::string(PredFunc2.getFunction()->getName().str());
  */

  Function *MergedFunc =
      Function::Create(funcType, GlobalValue::LinkageTypes::InternalLinkage,
                       Twine(Name), F1->getParent());

  ValueToValueMapTy VMap;

  // SmallVector<AttributeSet, 4> NewArgAttrs(MergedFunc->arg_size());

  std::vector<Argument *> ArgsList;
  for (Argument &arg : MergedFunc->args()) {
    ArgsList.push_back(&arg);
  }
  Value *FuncId = ArgsList[0];
  // AttributeList OldAttrs1 = PredFunc1.getFunction()->getAttributes();
  // AttributeList OldAttrs2 = PredFunc2.getFunction()->getAttributes();

  paramId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap1[paramId]];
    /*
    NewArgAttrs[ParamMap1[paramId]] =
                       OldAttrs1.getParamAttributes(paramId);
    */
    paramId++;
  }
  paramId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap2[paramId]];
    /*
    if (ParamMap2[paramId]>PredFunc1.getFunction()->arg_size()) {
      NewArgAttrs[ParamMap2[paramId]] =
                       OldAttrs2.getParamAttributes(paramId);
    }
    */
    paramId++;
  }

#ifdef TIME_STEPS_DEBUG
  TimeParam.stopTimer();
#endif

  /*
         MergedFunc->setAttributes(
                  AttributeList::get(MergedFunc->getContext(),
     OldAttrs1.getFnAttributes(),
                                     (RetType1==ReturnType)?OldAttrs1.getRetAttributes():OldAttrs2.getRetAttributes(),
                           NewArgAttrs));

   */
  // MergedFunc->setAttributes(PredFunc1.getFunction()->getAttributes());

  unsigned MaxAlignment = std::max(F1->getAlignment(), F2->getAlignment());

  MergedFunc->setAlignment(MaxAlignment);
  //

  // F->setLinkage(GlobalValue::PrivateLinkage);

  if (F1->getAttributes() == F2->getAttributes()) {
    // MergedFunc->setAttributes(PredFunc1.getFunction()->getAttributes());
  }

  if (F1->getCallingConv() == F2->getCallingConv()) {
    MergedFunc->setCallingConv(F1->getCallingConv());
  } else {
    errs() << "ERROR: different calling convention!\n";
  }
  //MergedFunc->setCallingConv(CallingConv::Fast);

  if (F1->getLinkage() == F2->getLinkage()) {
    MergedFunc->setLinkage(F1->getLinkage());
  } else {
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
    errs() << "ERROR: different linkage type!\n";
  }
  //MergedFunc->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);

  if (F1->isDSOLocal() == F2->isDSOLocal()) {
    MergedFunc->setDSOLocal(F1->isDSOLocal());
  } else {
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
    errs() << "ERROR: different DSO local!\n";
  }


  if (F1->getSubprogram() == F2->getSubprogram()) {
    MergedFunc->setSubprogram(F1->getSubprogram());
  } else {
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
    errs() << "ERROR: different subprograms!\n";
  }


  if (F1->getUnnamedAddr() == F2->getUnnamedAddr()) {
    MergedFunc->setUnnamedAddr(F1->getUnnamedAddr());
  } else {
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
    errs() << "ERROR: different unnamed addr!\n";
  }
  //MergedFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  if (F1->getVisibility() == F2->getVisibility()) {
    MergedFunc->setVisibility(F1->getVisibility());
  } else {
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
    errs() << "ERROR: different visibility!\n";
  }

  // Exception Handling requires landing pads to have the same personality
  // function
  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 == PersonalityFn2) {
      MergedFunc->setPersonalityFn(PersonalityFn1);
    } else {
#ifdef ENABLE_DEBUG_CODE
      PersonalityFn1->dump();
      PersonalityFn2->dump();
#endif
      errs() << "ERROR: different personality function!\n";
    }
  } else if (F1->hasPersonalityFn()) {
    errs() << "Only F1 has PersonalityFn\n";
    MergedFunc->setPersonalityFn(F1->getPersonalityFn()); // TODO: check if this
                                                          // is valid: merge
                                                          // function with
                                                          // personality with
                                                          // function without it
  } else if (F2->hasPersonalityFn()) {
    errs() << "Only F2 has PersonalityFn\n";
    MergedFunc->setPersonalityFn(F2->getPersonalityFn()); // TODO: check if this
                                                          // is valid: merge
                                                          // function with
                                                          // personality with
                                                          // function without it
  }

  if (F1->hasComdat() && F2->hasComdat()) {
    auto *Comdat1 = F1->getComdat();
    auto *Comdat2 = F2->getComdat();
    if (Comdat1 == Comdat2) {
      MergedFunc->setComdat(Comdat1);
    } else {
      errs() << "ERROR: different comdats!\n";
    }
  } else if (F1->hasComdat()) {
    errs() << "Only F1 has Comdat\n";
    MergedFunc->setComdat(F1->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
  } else if (F2->hasComdat()) {
    errs() << "Only F2 has Comdat\n";
    MergedFunc->setComdat(F2->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
  }

  if (F1->hasSection())
    MergedFunc->setSection(F1->getSection());

  bool RequiresFuncId = false;

  Value *IsFunc1 = FuncId;

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen1.startTimer();
#endif

  BasicBlock *MergedBB = nullptr;
  BasicBlock *MergedBB1 = nullptr;
  BasicBlock *MergedBB2 = nullptr;

  std::map<BasicBlock *, BasicBlock *> TailBBs;

  std::map<SelectCacheEntry, Value *> SelectCache;
  std::map<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> CacheBBSelect;
  std::vector<Instruction *> ListSelects;

  if (AlignedInsts.front().first == nullptr ||
      AlignedInsts.front().second == nullptr) {
    MergedBB = BasicBlock::Create(Context, "FrontFirstOrSecondNull:", MergedFunc);
  }

  //[Iuli:you've modified this part to take note of the predicted dynamic size of the merged function]
  for (auto Pair : AlignedInsts) {
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<BasicBlock>(Pair.first)) {
        //[Iuli] By looking at the source code it appears that NewBB created here is
        //the first bb in the function
        
        //It shouldn't have any impact on the your own instrumentation
        BasicBlock *BB1 = dyn_cast<BasicBlock>(Pair.first);
        BasicBlock *BB2 = dyn_cast<BasicBlock>(Pair.second);
        
        Function *F1 = BB1->getParent();
        Function *F2 = BB2->getParent();

        double profile_bb1 = FMSALegacyPass::bbProfiles[GetValueName(F1) + "::" + BB1->getName().str()];
        double profile_bb2 = FMSALegacyPass::bbProfiles[GetValueName(F2) + "::" + BB2->getName().str()];
        
        double max_runtime;
        std::string new_bb_name = GetValueName(F1)+"::"+BB1->getName().str()+"::"+GetValueName(F2)+"::"+BB2->getName().str();
        
        //BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        BasicBlock *NewBB = BasicBlock::Create(Context, "MergedLP:" + new_bb_name, MergedFunc);
      
        std::vector<uint32_t> BBWeights1;
        std::vector<uint32_t> BBWeights2;
        std::vector<uint32_t> BBWeights;
    
        //[Iuli] We can assume for now that this path represents either the entry case
        //or the case where the loops share stopping conditions through phi instructions

        if (&(F1->getEntryBlock()) == BB1) {
            if (F1->hasProfileData() and F1->getEntryCount().hasValue())
                BBWeights.push_back(F1->getEntryCount().getCount());
        } else {
            getKindOfTerminator(BBWeights1, BB1);
        }
        if (&(F2->getEntryBlock()) == BB2) {
            if (F2->hasProfileData() and F2->getEntryCount().hasValue())
                BBWeights.push_back(F2->getEntryCount().getCount());
        } else {
            getKindOfTerminator(BBWeights2, BB2);
        }

        if (BBWeights1.size() != 0 and BBWeights2.size() != 0 and BBWeights1.size() == BBWeights2.size())
            for (int i = 0; i < BBWeights1.size(); ++i) {
                BBWeights.push_back(BBWeights1[i] + BBWeights2[i]);
            }
        else if (BBWeights1.size() != BBWeights2.size()) {
            errs() << "There is some missmatch between the sizes of BBWeights1=" << BBWeights1.size();
            errs() << " and BBWeights2=" << BBWeights2.size() << "\n";
            errs() << " missmatch for BB1=" << GetValueName(F1) + "::" + GetValueName(BB1) << " and BB2=" << GetValueName(F2)+"::"+GetValueName(BB2) << "\n";
            //exit(-1);
        }

        //MergingNativeProfiling[NewBB] = MergedBBDynProf(BBWeights, false);
        MergingNativeProfiling[NewBB] = BBWeights;


        FMSALegacyPass::bbProfiles[GetValueName(NewBB->getParent()) + "::" + NewBB->getName().str()] = profile_bb1 + profile_bb2;
        FMSALegacyPass::mergedBBProfiles[NewBB] = profile_bb1 + profile_bb2;

        VMap[BB1] = NewBB;
        VMap[BB2] = NewBB;
        if (BB1->isLandingPad() || BB2->isLandingPad()) {
          LandingPadInst *LP1 = BB1->getLandingPadInst();
          LandingPadInst *LP2 = BB2->getLandingPadInst();
          assert((LP1 != nullptr && LP2 != nullptr) &&
                 "Should be both as per the BasicBlock match!");
          Instruction *NewLP = LP1->clone();
          VMap[LP1] = NewLP;
          VMap[LP2] = NewLP;

          IRBuilder<> Builder(NewBB);
          Builder.Insert(NewLP);
        }
      }
    } else {
      Value *V = nullptr;
      if (Pair.first) {
        V = Pair.first;
      } else {
        V = Pair.second;
      }

      if (isa<BasicBlock>(V)) {
        BasicBlock *BB = dyn_cast<BasicBlock>(V);

        //BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        BasicBlock *NewBB = BasicBlock::Create(Context, "TailMerge:" + GetValueName(BB->getParent()) + BB->getName().str(), MergedFunc);
        std::vector<uint32_t> BBWeights;

        getKindOfTerminator(BBWeights, BB);
        //MergingNativeProfiling[NewBB] = MergedBBDynProf(BBWeights, false);
        MergingNativeProfiling[NewBB] = BBWeights;

        Function *F = BB->getParent();
        if (not FMSALegacyPass::bbProfiles.empty()) {
            double profile_bb = FMSALegacyPass::bbProfiles[GetValueName(F) + "::" + BB->getName().str()];
            FMSALegacyPass::bbProfiles[GetValueName(F) + "::" + NewBB->getName().str()] = profile_bb;
            FMSALegacyPass::mergedBBProfiles[NewBB] = profile_bb;
        }
        VMap[BB] = NewBB;
        TailBBs[dyn_cast<BasicBlock>(V)] = NewBB;

        if (BB->isLandingPad()) {
          LandingPadInst *LP = BB->getLandingPadInst();
          Instruction *NewLP = LP->clone();
          VMap[LP] = NewLP;

          IRBuilder<> Builder(NewBB);
          Builder.Insert(NewLP);
        }
      }
    }
  }

  if (AlignedInsts.front().first == nullptr ||
      AlignedInsts.front().second == nullptr) {
    // MergedBB = BasicBlock::Create(Context, "", MergedFunc);
    BasicBlock *EntryBB1 = dyn_cast<BasicBlock>(&F1->getEntryBlock());
    BasicBlock *EntryBB2 = dyn_cast<BasicBlock>(&F2->getEntryBlock());
    IRBuilder<> Builder(MergedBB);

    std::vector<uint32_t> BBWeights;

    //GetBranchWeights(EntryBB1->getTerminator(), BB1Weights);
    //GetBranchWeights(EntryBB2->getTerminator(), BB2Weights);
    BBWeights.push_back((F1->getEntryCount()).getCount());
    BBWeights.push_back((F2->getEntryCount()).getCount());

    //MergingNativeProfiling[MergedBB] = MergedBBDynProf(BBWeights, false);
    MergingNativeProfiling[MergedBB] = BBWeights;

    setDynamicProfiling(MergedBB, EntryBB1, EntryBB2);

    BranchInst *BI = Builder.CreateCondBr(IsFunc1, dyn_cast<BasicBlock>(VMap[EntryBB1]),
                         dyn_cast<BasicBlock>(VMap[EntryBB2]));
  }
  MergedBB = nullptr;

  for (auto Pair : AlignedInsts) {
    // mergable instructions
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<BasicBlock>(Pair.first)) {
        BasicBlock *NewBB =
            dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(Pair.first)]);
        // VMap[dyn_cast<BasicBlock>(Pair.first)] = NewBB;
        // VMap[dyn_cast<BasicBlock>(Pair.second)] = NewBB;
        // BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        // VMap[dyn_cast<BasicBlock>(Pair.first)] = NewBB;
        // VMap[dyn_cast<BasicBlock>(Pair.second)] = NewBB;

        MergedBB = NewBB;
        MergedBB1 = dyn_cast<BasicBlock>(Pair.first);
        MergedBB2 = dyn_cast<BasicBlock>(Pair.second);

      } else {
        assert(isa<Instruction>(Pair.first) && "Instruction expected!");
        Instruction *I1 = dyn_cast<Instruction>(Pair.first);
        Instruction *I2 = dyn_cast<Instruction>(Pair.second);


        if (MergedBB == nullptr) {

          Function *F1 = (I1->getParent())->getParent();
          Function *F2 = (I2->getParent())->getParent();

          BasicBlock *BB1 = I1->getParent();
          BasicBlock *BB2 = I2->getParent();

          std::string new_bb_name = GetValueName(BB1) + GetValueName(BB2);
          MergedBB = BasicBlock::Create(Context, "Simple1:" + new_bb_name, MergedFunc);


          std::vector<uint32_t> BBWeights;

          getKindOfTerminator(BBWeights, BB1);
          getKindOfTerminator(BBWeights, BB2);
          
          MergingNativeProfiling[MergedBB] = BBWeights;

          setDynamicProfiling(MergedBB, BB1, BB2);

          //BasicBlock *bb_src2 = dyn_cast<BasicBlock>(Pair.second);
          //Function *my_f2 = bb_src2->getParent();

          //double profile_bb = FMSALegacyPass::bbProfiles[my_f2->getName().str() + "::" + bb_src2->getName().str()];
          //FMSALegacyPass::bbProfiles[MergedBB->getName().str() + "::" + MergedBB->getName().str()] = profile_bb;
          //FMSALegacyPass::mergedBBProfiles[MergedBB] = profile_bb;


/*
          I1->dump();
          errs() << "Tail: " <<  GetValueName(dyn_cast<BasicBlock>(I1->getParent())) << " -> " << GetValueName(TailBBs[ dyn_cast<BasicBlock>(I1->getParent()) ]) << "\n";
          I2->dump();
          errs() << "Tail: " <<  GetValueName(dyn_cast<BasicBlock>(I2->getParent())) << " -> " << GetValueName(TailBBs[ dyn_cast<BasicBlock>(I2->getParent()) ]) << "\n";
*/
          {
            IRBuilder<> Builder(TailBBs[ dyn_cast<BasicBlock>(I1->getParent()) ]);
            Builder.CreateBr(MergedBB);
          }
          {
            IRBuilder<> Builder(TailBBs[ dyn_cast<BasicBlock>(I2->getParent()) ]);
            Builder.CreateBr(MergedBB);
          }
        }
        MergedBB1 = dyn_cast<BasicBlock>(I1->getParent());
        MergedBB2 = dyn_cast<BasicBlock>(I2->getParent());

        // if (
        // VMap[I1->getParent()]!=VMap[I2->getParent()]
        // )
        // {
        //   errs() << "Merged instructions not in the same BB\n";
        //}

        Instruction *I = I1;
        if (I1->getOpcode() == Instruction::Ret) {
          if (I1->getNumOperands() >= I2->getNumOperands())
            I = I1;
          else
            I = I2;
        } else {
          assert(I1->getNumOperands() == I2->getNumOperands() &&
                 "Num of Operands SHOULD be EQUAL\n");
        }

        Instruction *NewI = I->clone();
        VMap[I1] = NewI;
        VMap[I2] = NewI;

        IRBuilder<> Builder(MergedBB);
        Builder.Insert(NewI);

        // TODO: temporary removal of metadata
        
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        NewI->getAllMetadata(MDs);
        for (std::pair<unsigned, MDNode *> MDPair : MDs) {
          if (MDPair.first != LLVMContext::MD_prof)
            NewI->setMetadata(MDPair.first, nullptr);
        }

/*        
        for (Instruction &I : *MergedBB) {
          if (isa<LandingPadInst>(&I) && MergedBB->getFirstNonPHI()!=(&I)) {
            NewI->dump();
            errs() << "Broken BB: 1\n";
            MergedBB->dump();
          }
        }
*/

        if (isa<TerminatorInst>(NewI)) {
          MergedBB = nullptr;
          MergedBB1 = nullptr;
          MergedBB2 = nullptr;
        }
      }
    } else {
      RequiresFuncId = true;

      if (MergedBB != nullptr) {

        //in this case it's not clear if there is a 1-to-1 mapping between MergedBB1 and NewBB1. Needs validation
        
        std::string f1Name = GetValueName(MergedBB1->getParent());
        std::string bb1Name = f1Name + "::" + MergedBB1->getName().str();
        std::string f2Name = GetValueName(MergedBB2->getParent());
        std::string bb2Name = f2Name + "::" + MergedBB2->getName().str();
        
        BasicBlock *NewBB1 = BasicBlock::Create(Context, "Nonalig1:" + bb1Name, MergedFunc);
        BasicBlock *NewBB2 = BasicBlock::Create(Context, "Nonalig2:" + bb2Name, MergedFunc);
 
        if (not FMSALegacyPass::bbProfiles.empty()) {
            
            if (FMSALegacyPass::bbProfiles.find(bb1Name) != FMSALegacyPass::bbProfiles.end()) {
                FMSALegacyPass::mergedBBProfiles[NewBB1] = FMSALegacyPass::bbProfiles[bb1Name];
                FMSALegacyPass::bbProfiles[GetValueName(NewBB1->getParent()) + "::" + NewBB1->getName().str()] = FMSALegacyPass::bbProfiles[bb1Name];
            } else
                errs() << "Could not find " + bb1Name + " in bbProfiles for Nonalig1.\n"; 
            if (FMSALegacyPass::bbProfiles.find(bb2Name) != FMSALegacyPass::bbProfiles.end()){
                FMSALegacyPass::mergedBBProfiles[NewBB2] = FMSALegacyPass::bbProfiles[bb2Name];
                FMSALegacyPass::bbProfiles[GetValueName(NewBB2->getParent()) + "::" + NewBB2->getName().str()] = FMSALegacyPass::bbProfiles[bb2Name];
            } else
                errs() << "Could not find " + bb2Name + " in bbProfiles for Nonalig2.\n"; 

        }

        TailBBs[MergedBB1] = NewBB1;
        TailBBs[MergedBB2] = NewBB2;

        IRBuilder<> Builder(MergedBB);
        Builder.CreateCondBr(IsFunc1, NewBB1, NewBB2);

        MergedBB = nullptr;
      }

      Value *V = nullptr;
      if (Pair.first) {
        V = Pair.first;
      } else {
        V = Pair.second;
      }

      if (isa<BasicBlock>(V)) {
        BasicBlock *NewBB = dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(V)]);
        // BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        // VMap[dyn_cast<BasicBlock>(V)] = NewBB;
        TailBBs[dyn_cast<BasicBlock>(V)] = NewBB;
      } else {
        assert(isa<Instruction>(V) && "Instruction expected!");
        Instruction *I = dyn_cast<Instruction>(V);

        // I->dump();

        Instruction *NewI = nullptr;
        if (I->getOpcode() == Instruction::Ret && !ReturnType->isVoidTy() &&
            I->getNumOperands() == 0) {
          NewI = ReturnInst::Create(Context, UndefValue::get(ReturnType));
        } else
          NewI = I->clone();
        VMap[I] = NewI;

        BasicBlock *BBPoint = TailBBs[dyn_cast<BasicBlock>(I->getParent())];
        if (BBPoint == nullptr) {
          BBPoint = TailBBs[dyn_cast<BasicBlock>(I->getParent())] =
              dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(I->getParent())]);
        }

        IRBuilder<> Builder(BBPoint);
        Builder.Insert(NewI);

        // TODO: temporarily removing metadata
        
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        NewI->getAllMetadata(MDs);
        for (std::pair<unsigned, MDNode *> MDPair : MDs) {
          if (MDPair.first != LLVMContext::MD_prof)
            NewI->setMetadata(MDPair.first, nullptr);
        }

/*        
        auto *BB =TailBBs[dyn_cast<BasicBlock>(I->getParent())];
        for (Instruction &I : *BB) {
          if (isa<LandingPadInst>(&I) && BB->getFirstNonPHI()!=(&I)) {
            errs() << "Broken BB: 2\n";
            BB->dump();
          }
        }
*/

      }

    }
  }

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen1.stopTimer();
#endif

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen2.startTimer();
#endif

  for (auto Pair : AlignedInsts) {
    // mergable instructions
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<Instruction>(Pair.first)) {
        Instruction *I1 = dyn_cast<Instruction>(Pair.first);
        Instruction *I2 = dyn_cast<Instruction>(Pair.second);

        Instruction *I = I1;
        if (I1->getOpcode() == Instruction::Ret) {
          if (I1->getNumOperands() >= I2->getNumOperands())
            I = I1;
          else
            I = I2;
        } else {
          assert(I1->getNumOperands() == I2->getNumOperands() &&
                 "Num of Operands SHOULD be EQUAL\n");
        }

        Instruction *NewI = dyn_cast<Instruction>(VMap[I]);

        IRBuilder<> Builder(NewI);

        if (isa<BinaryOperator>(NewI) && I->isCommutative()) {
          CountBinOps++;

          BinaryOperator *BO1 = dyn_cast<BinaryOperator>(I1);
          BinaryOperator *BO2 = dyn_cast<BinaryOperator>(I2);
          Value *VL1 = MapValue(BO1->getOperand(0), VMap);
          Value *VL2 = MapValue(BO2->getOperand(0), VMap);
          Value *VR1 = MapValue(BO1->getOperand(1), VMap);
          Value *VR2 = MapValue(BO2->getOperand(1), VMap);
          if (VL1 == VR2 && VL2 != VR2) {
            Value *TmpV = VR2;
            VR2 = VL2;
            VL2 = TmpV;
            CountOpReorder++;
          }

          std::vector<std::pair<Value *, Value *>> Vs;
          Vs.push_back(std::pair<Value *, Value *>(VL1, VL2));
          Vs.push_back(std::pair<Value *, Value *>(VR1, VR2));

          for (unsigned i = 0; i < Vs.size(); i++) {
            Value *V1 = Vs[i].first;
            Value *V2 = Vs[i].second;

            Value *V = V1; // first assume that V1==V2
            if (V1 != V2) {
              RequiresFuncId = true;
              // create predicated select instruction
              if (V1 == ConstantInt::getTrue(Context) &&
                  V2 == ConstantInt::getFalse(Context)) {
                V = IsFunc1;
              } else if (V1 == ConstantInt::getFalse(Context) &&
                         V2 == ConstantInt::getTrue(Context)) {
                V = Builder.CreateNot(IsFunc1);
              } else {
                Value *SelectI = nullptr;

                SelectCacheEntry SCE(IsFunc1, V1, V2, NewI->getParent());
                if (SelectCache.find(SCE) != SelectCache.end()) {
                  SelectI = SelectCache[SCE];
                } else {
                  Value *CastedV2 =
                      createCastIfNeeded(V2, V1->getType(), Builder, IntPtrTy);
                  SelectI = Builder.CreateSelect(IsFunc1, V1, CastedV2);

                  ListSelects.push_back(dyn_cast<Instruction>(SelectI));

                  SelectCache[SCE] = SelectI;
                }

                V = SelectI;
              }
            }

            Value *CastedV = createCastIfNeeded(
                V, NewI->getOperand(i)->getType(), Builder, IntPtrTy);
            NewI->setOperand(i, CastedV);
          }
        } else {
          for (unsigned i = 0; i < I->getNumOperands(); i++) {
            Value *F1V = nullptr;
            Value *V1 = nullptr;
            if (i < I1->getNumOperands()) {
              F1V = I1->getOperand(i);
              V1 = MapValue(I1->getOperand(i), VMap);
              if (V1 == nullptr) {
                errs() << "ERROR: Null value mapped: V1 = "
                          "MapValue(I1->getOperand(i), "
                          "VMap);\n";
                MergedFunc->eraseFromParent();
                return ErrorResponse;
              }
            } else
              V1 = GetAnyValue(I2->getOperand(i)->getType());

            Value *F2V = nullptr;
            Value *V2 = nullptr;
            if (i < I2->getNumOperands()) {
              F2V = I2->getOperand(i);
              V2 = MapValue(I2->getOperand(i), VMap);
              if (V2 == nullptr) {
                errs() << "ERROR: Null value mapped: V2 = "
                          "MapValue(I2->getOperand(i), "
                          "VMap);\n";
                MergedFunc->eraseFromParent();
                return ErrorResponse;
              }
            } else
              V2 = GetAnyValue(I1->getOperand(i)->getType());

            // if (V1==nullptr) V1 = V2;
            // if (V2==nullptr) V2 = V1;
            assert(V1 != nullptr && "Value should NOT be null!");
            assert(V2 != nullptr && "Value should NOT be null!");

            Value *V = V1; // first assume that V1==V2

            if (V1 != V2) {
              RequiresFuncId = true;
              if (isa<BasicBlock>(V1) && isa<BasicBlock>(V2)) {
                auto CacheKey = std::pair<BasicBlock *, BasicBlock *>(
                    dyn_cast<BasicBlock>(V1), dyn_cast<BasicBlock>(V2));
                BasicBlock *SelectBB = nullptr;
                if (CacheBBSelect.find(CacheKey) != CacheBBSelect.end()) {
                  SelectBB = CacheBBSelect[CacheKey];
                } else {
                  BasicBlock *BB1 = dyn_cast<BasicBlock>(V1);
                  BasicBlock *BB2 = dyn_cast<BasicBlock>(V2);
                  
                  SelectBB = BasicBlock::Create(Context, "SelectBB: " + GetValueName(BB1) + GetValueName(BB2), MergedFunc);
                  IRBuilder<> BuilderBB(SelectBB);
				  uint64_t bb1PredWeights = getPrevWeights(BB1);
				  uint64_t bb2PredWeights = getPrevWeights(BB2);
                  std::vector<uint32_t> BBWeights;
				  BBWeights.push_back(bb1PredWeights);
				  BBWeights.push_back(bb2PredWeights);
                  MergingNativeProfiling[SelectBB] = BBWeights;

                  setDynamicProfiling(SelectBB, BB1, BB2);

                  if (BB1->isLandingPad() || BB2->isLandingPad()) {
                    LandingPadInst *LP1 = BB1->getLandingPadInst();
                    LandingPadInst *LP2 = BB2->getLandingPadInst();
                    // assert ( (LP1!=nullptr && LP2!=nullptr) && "Should be
                    // both as per the BasicBlock match!");
                    if (LP1 == nullptr || LP2 == nullptr) {
                      errs() << "Should have two LandingPadInst as per the "
                                "BasicBlock match!\n";
#ifdef ENABLE_DEBUG_CODE
                      I1->dump();
                      I2->dump();
                      NewI->dump();
#endif
                      MergedFunc->eraseFromParent();
                      return ErrorResponse;
                    }

                    Instruction *NewLP = LP1->clone();
                    BuilderBB.Insert(NewLP);
                    
                    BasicBlock *F1BB = dyn_cast<BasicBlock>(F1V);
                    BasicBlock *F2BB = dyn_cast<BasicBlock>(F2V);

                    VMap[F1BB] = SelectBB;
                    VMap[F2BB] = SelectBB;
                    if (TailBBs[F1BB]==nullptr) TailBBs[F1BB]=BB1;
                    if (TailBBs[F2BB]==nullptr) TailBBs[F2BB]=BB2;
                    VMap[F1BB->getLandingPadInst()] = NewLP;
                    VMap[F2BB->getLandingPadInst()] = NewLP;
                    
                    /*
                    for (auto kv : VMap) {
                      if (kv.second == LP1 || kv.second == LP2) {
                        VMap[kv.first] = NewLP;
                      } else if(kv.second == BB1 || kv.second == B2) {
                        VMap[kv.first] = SelectBB;
                      }
                    }
                    */
                    
                    BB1->replaceAllUsesWith(SelectBB);
                    BB2->replaceAllUsesWith(SelectBB);

                    LP1->replaceAllUsesWith(NewLP);
                    LP1->eraseFromParent();
                    LP2->replaceAllUsesWith(NewLP);
                    LP2->eraseFromParent();
                  }

                  BuilderBB.CreateCondBr(IsFunc1, BB1, BB2);
                  CacheBBSelect[CacheKey] = SelectBB;
                }
                V = SelectBB;
              } else {
                // create predicated select instruction
                if (V1 == ConstantInt::getTrue(Context) &&
                    V2 == ConstantInt::getFalse(Context)) {
                  V = IsFunc1;
                } else if (V1 == ConstantInt::getFalse(Context) &&
                           V2 == ConstantInt::getTrue(Context)) {
                  V = Builder.CreateNot(IsFunc1);
                } else {
                  Value *SelectI = nullptr;

                  SelectCacheEntry SCE(IsFunc1, V1, V2, NewI->getParent());
                  if (SelectCache.find(SCE) != SelectCache.end()) {
                    SelectI = SelectCache[SCE];
                  } else {

                    Value *CastedV2 = createCastIfNeeded(V2, V1->getType(),
                                                         Builder, IntPtrTy);
                    SelectI = Builder.CreateSelect(IsFunc1, V1, CastedV2);

                    ListSelects.push_back(dyn_cast<Instruction>(SelectI));

                    SelectCache[SCE] = SelectI;
                  }

                  V = SelectI;
                }
              }
            }

            Value *CastedV = V;
            if (!isa<BasicBlock>(V))
              CastedV = createCastIfNeeded(V, NewI->getOperand(i)->getType(),
                                           Builder, IntPtrTy);
            NewI->setOperand(i, CastedV);
          }
        } // TODO: end of commutative if-else

/*
        for (Instruction &I : *NewI->getParent()) {
          if (isa<LandingPadInst>(&I) && NewI->getParent()->getFirstNonPHI()!=(&I)) {
            NewI->dump();
            errs() << "Broken BB: 3\n";
            NewI->getParent()->dump();
          }
        }
*/

      }

    } else {
      RequiresFuncId = true;

      Value *V = nullptr;
      if (Pair.first) {
        V = Pair.first;
      } else {
        V = Pair.second;
      }

      if (isa<Instruction>(V)) {
        Instruction *I = dyn_cast<Instruction>(V);

        Instruction *NewI = dyn_cast<Instruction>(VMap[I]);

        IRBuilder<> Builder(NewI);

        for (unsigned i = 0; i < I->getNumOperands(); i++) {
          Value *V = MapValue(I->getOperand(i), VMap);
          if (V == nullptr) {
            errs() << "ERROR: Null value mapped: V = "
                      "MapValue(I->getOperand(i), VMap);\n";
            MergedFunc->eraseFromParent();
            return ErrorResponse;
          }

          Value *CastedV = V;
          if (!isa<BasicBlock>(V))
            CastedV = createCastIfNeeded(V, NewI->getOperand(i)->getType(),
                                         Builder, IntPtrTy);
          NewI->setOperand(i, CastedV);
        }

/*
        for (Instruction &I : *NewI->getParent()) {
          if (isa<LandingPadInst>(&I) && NewI->getParent()->getFirstNonPHI()!=(&I)) {
            NewI->dump();
            errs() << "Broken BB: 4\n";
            NewI->getParent()->dump();
          }
        }
*/

      }
    }
  }

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen2.stopTimer();
#endif

#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.startTimer();
#endif

  {
    DominatorTree DT(*MergedFunc);
    removeRedundantInstructions(MergedFunc, DT, ListSelects);
  }

  {
    DominatorTree DT(*MergedFunc);
    if (!fixNotDominatedUses(MergedFunc, DT)) {
      MergedFunc->eraseFromParent();
      MergedFunc = nullptr;
    }
  }

#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.stopTimer();
#endif

  MergedFunction Result(F1, F2, MergedFunc);
  Result.ParamMap1 = ParamMap1;
  Result.ParamMap2 = ParamMap2;
  Result.hasFuncIdArg = (FuncId != nullptr);
  Result.RoughReduction = RoughReduction;
  return Result;
  /*for (auto &BB : *(Result.MergedFunc)) {
    for (auto &NewI : BB) {
      SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
      NewI.getAllMetadata(MDs);
      for (std::pair<unsigned, MDNode *> MDPair : MDs) {
        if (MDPair.first == LLVMContext::MD_prof)
          NewI.setMetadata(MDPair.first, nullptr);
      }
    }
    if (MergingNativeProfiling.find(&BB) != MergingNativeProfiling.end()) {
      //MergedBBDynProf &bbOutgoingProf = MergingNativeProfiling[&BB];
      std::vector<uint32_t> bbOutgoingProf = MergingNativeProfiling[&BB];
      
      if (bbOutgoingProf.size() != 0) {
        TerminatorInst *TI = BB.getTerminator();
        
        if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
          
          if (BI->isUnconditional() and bbOutgoingProf.size() != 1) {
            errs() << "The branch is unconditional and there are more profiling weights\n";
            errs() << "Potential solution: Attach the sum of the profiling infos.." << 
                       bbOutgoingProf.size() << "\n";
            //exit(-1);
          } else if (BI->isConditional() and bbOutgoingProf.size() != 2) {
            errs() << "The branch is conditional and there are more profiling weights:" << \
                       bbOutgoingProf.size() << "\n";
            errs() << "Potential solution: Attach the sum of the profiling infos..\n";
            //exit(-1);
          }
          SmallVector<uint32_t, 10> BBWeightsSmall;
          for (auto it = bbOutgoingProf.begin(); it != bbOutgoingProf.end(); ++it)
            BBWeightsSmall.push_back(*it);

          MDNode *N = MDBuilder(BB.getContext()).createBranchWeights(BBWeightsSmall);
          BI->setMetadata(LLVMContext::MD_prof, N);
        }
      }
    }

  }*/
}



static MergedFunction
mergeBySequenceAlignmentPrecomputedAlignedInsts(Function *F1, Function *F2,
                         std::list<std::pair<Value *, Value *>> AlignedInsts,
                         double RoughReduction) {


  MergedFunction ErrorResponse(F1, F2, nullptr);
  std::map<BasicBlock *, uint64_t> bbWithWeightsPointing;
  
  getAllUsesInfoMap(bbWithWeightsPointing, F1);
  getAllUsesInfoMap(bbWithWeightsPointing, F2);

  if (!F1->getSection().equals(F2->getSection())) {
    if (Verbose) {
      errs() << "Functions differ in their sections! " << GetValueName(F1)
             << ", " << GetValueName(F2) << "\n";
    }
    return ErrorResponse;
  }


  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 != PersonalityFn2) {
      errs() << "Functions differ in their personality function!\n";
      return ErrorResponse;
    }
  }
#ifdef TIME_STEPS_DEBUG
  TimeAlign.startTimer();
#endif
  
  return mergeBySequenceAlignmentBackend(F1, F2, AlignedInsts, RoughReduction);
}

static MergedFunction
mergeBySequenceAlignment(Function *F1, Function *F2,
                         SmallVectorImpl<Value *> &F1Vec,
                         SmallVectorImpl<Value *> &F2Vec,
                         std::list<BasicBlock *> &OrderedBBs1, 
                         std::list<BasicBlock *> &OrderedBBs2) {


  MergedFunction ErrorResponse(F1, F2, nullptr);
  std::map<BasicBlock *, uint64_t> bbWithWeightsPointing;
  
  getAllUsesInfoMap(bbWithWeightsPointing, F1);
  getAllUsesInfoMap(bbWithWeightsPointing, F2);

  if (!F1->getSection().equals(F2->getSection())) {
    if (Verbose) {
      errs() << "Functions differ in their sections! " << GetValueName(F1)
             << ", " << GetValueName(F2) << "\n";
    }
    return ErrorResponse;
  }


  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 != PersonalityFn2) {
      errs() << "Functions differ in their personality function!\n";
      return ErrorResponse;
    }
  }
#ifdef TIME_STEPS_DEBUG
  TimeAlign.startTimer();
#endif

  std::list<std::pair<Value *, Value *>> AlignedInsts;
  //double RoughReduction = AlignLinearizedCFGs(F1Vec, F2Vec, AlignedInsts);
  std::vector<BasicBlock *> OrderedBBs1Vec(OrderedBBs1.begin(), OrderedBBs1.end());
  std::vector<BasicBlock *> OrderedBBs2Vec(OrderedBBs2.begin(), OrderedBBs2.end());
  double RoughReduction;
  if (UseRandomShufflings)
    RoughReduction = AlignLinearizedCFGsRandomShufflings(F1Vec, F2Vec, OrderedBBs1Vec, OrderedBBs2Vec, AlignedInsts);
  else
    RoughReduction = AlignLinearizedCFGs(F1Vec, F2Vec, AlignedInsts);
#ifdef TIME_STEPS_DEBUG
  TimeAlign.stopTimer();
#endif

  return mergeBySequenceAlignmentBackend(F1, F2, AlignedInsts, RoughReduction);
}

static bool canReplaceAllCalls(Function *F) {
  for (User *U : F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledFunction() != F)
        return false;
    } else
      return false;
  }
  return true;
}

void replaceByCall(Module *M, Function *F, MergedFunction &MergedFunc) {
  LLVMContext &Context = M->getContext();
  const DataLayout *DL = &M->getDataLayout();
  Type *IntPtrTy = DL ? DL->getIntPtrType(Context) : NULL;

  if (Verbose) {
    errs() << "replaceByCall\n";
  }

  Value *FuncId = (MergedFunc.F1 == F)
                      ? ConstantInt::getTrue(MergedFunc.F1->getContext())
                      : ConstantInt::getFalse(MergedFunc.F1->getContext());
  Function *MergedF = MergedFunc.MergedFunc;

  F->deleteBody();
  BasicBlock *NewBB = BasicBlock::Create(MergedFunc.F1->getContext(), "ReplaceByCall", F);
  IRBuilder<> Builder(NewBB);

  // if (Verbose) {
  //   F->dump();
  //}

  std::vector<Value *> args;
  for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
    args.push_back(nullptr);
  }

  if (MergedFunc.hasFuncIdArg) {
    args[0] = FuncId;
    // args[0] = Builder.getInt32(FuncId);
  }

  std::vector<Argument *> ArgsList;
  for (Argument &arg : F->args()) {
    ArgsList.push_back(&arg);
  }

  if (MergedFunc.F1 == F) {
    for (auto Pair : MergedFunc.ParamMap1) {
      args[Pair.second] = ArgsList[Pair.first];
    }
  } else {
    for (auto Pair : MergedFunc.ParamMap2) {
      args[Pair.second] = ArgsList[Pair.first];
    }
  }
  for (unsigned i = 0; i < args.size(); i++) {
    if (args[i] == nullptr) {
      args[i] = GetAnyValue(MergedF->getFunctionType()->getParamType(i));
    }
  }

  CallInst *CI =
      (CallInst *)Builder.CreateCall(MergedF, ArrayRef<Value *>(args));
  CI->setTailCall();
  CI->setCallingConv(MergedF->getCallingConv());
  CI->setAttributes(MergedF->getAttributes());
  CI->setIsNoInline();

  if (F->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Value *CastedV =
        createCastIfNeeded(CI, F->getReturnType(), Builder, IntPtrTy);
    Builder.CreateRet(CastedV);
  }
}

bool replaceCallsWith(Module *M, Function *F, MergedFunction &MergedFunc) {
  //return false;

  LLVMContext &Context = M->getContext();
  const DataLayout *DL = &M->getDataLayout();
  Type *IntPtrTy = DL ? DL->getIntPtrType(Context) : NULL;

  Value *FuncId = (MergedFunc.F1 == F) ? ConstantInt::getTrue(Context)
                                       : ConstantInt::getFalse(Context);
  Function *MergedF = MergedFunc.MergedFunc;

  if (Verbose) {
    errs() << "replaceCallsWith\n";
  }
  std::vector<CallInst *> Calls;
  for (User *U : F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledFunction() == F) {
        CallInst *CI = dyn_cast<CallInst>(U); // CS.getInstruction());
        Calls.push_back(CI);
      } else
        return false;
    } else
      return false;
  }

  for (CallInst *CI : Calls) {
    IRBuilder<> Builder(CI);

    std::vector<Value *> args;
    for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
      args.push_back(nullptr);
    }

    if (MergedFunc.hasFuncIdArg) {
      args[0] = FuncId;
    }

    if (MergedFunc.F1 == F) {
      for (auto Pair : MergedFunc.ParamMap1) {
        args[Pair.second] = CI->getArgOperand(Pair.first);
      }
    } else {
      for (auto Pair : MergedFunc.ParamMap2) {
        args[Pair.second] = CI->getArgOperand(Pair.first);
      }
    }
    for (unsigned i = 0; i < args.size(); i++) {
      if (args[i] == nullptr) {
        args[i] = GetAnyValue(MergedF->getFunctionType()->getParamType(i));
      }
    }

    CallInst *NewCI = (CallInst *)Builder.CreateCall(MergedF->getFunctionType(),
                                                     MergedF, args);
    NewCI->setCallingConv(MergedF->getCallingConv());
    NewCI->setAttributes(MergedF->getAttributes());
    NewCI->setIsNoInline();

    Value *CastedV = NewCI;
    if (!F->getReturnType()->isVoidTy()) {
      CastedV =
          createCastIfNeeded(NewCI, F->getReturnType(), Builder, IntPtrTy);
    }
    // if (F->getReturnType()==MergedF->getReturnType())
    if (CI->getNumUses() > 0) {
      CI->replaceAllUsesWith(CastedV);
    }

    if (CI->getNumUses() == 0) {
      CI->eraseFromParent();
    } else {
      if (CI->getNumUses() > 0) {

        if (Verbose) {
          errs() << "ERROR: Function Call has uses\n";
#ifdef ENABLE_DEBUG_CODE
          CI->dump();
          errs() << "Called type\n";
          F->getReturnType()->dump();
          errs() << "Merged type\n";
          MergedF->getReturnType()->dump();
#endif
        }
      }
    }
  }

  return true;
}

int requiresOriginalInterfaces(MergedFunction &MergedFunc) {
  return (canReplaceAllCalls(MergedFunc.F1) ? 0 : 1) +
         (canReplaceAllCalls(MergedFunc.F2) ? 0 : 1);
}

void static UpdateCallGraph(Module &M, MergedFunction &Result,
                            StringSet<> &AlwaysPreserved) {
  Function *F1 = Result.F1;
  Function *F2 = Result.F2;
  
  uint64_t profilingInfo = 0;
  errs() << "Prof data for merged " << GetValueName(Result.MergedFunc) << "\n";
  if (F1->hasProfileData() and (F1->getEntryCount()).hasValue()) {
    errs() << "Reading prof data for " << GetValueName(F1) << " and it is " << F1->getEntryCount().getCount() << "\n";
    errs() << "the prof data for the other function is " << F2->getEntryCount().getCount() << "\n";
    profilingInfo += (F1->getEntryCount()).getCount();
  }
  
  if (F2->hasProfileData() and (F2->getEntryCount()).hasValue()) {
    errs() << "Reading prof data for " << GetValueName(F2) << " and it is " << F2->getEntryCount().getCount() << "\n";
    errs() << "the prof data for the other function is " << F2->getEntryCount().getCount() << "\n";
    profilingInfo += (F2->getEntryCount()).getCount();
  }
 
  if (profilingInfo != 0)
    (Result.MergedFunc)->setEntryCount(profilingInfo);
  
  replaceByCall(&M, F1, Result);
  replaceByCall(&M, F2, Result);

  bool CanEraseF1 = replaceCallsWith(&M, F1, Result);
  bool CanEraseF2 = replaceCallsWith(&M, F2, Result);

  ////if (F1->getLinkage()==GlobalValue::LinkageTypes::InternalLinkage
  ///|| F1->getLinkage()==GlobalValue::LinkageTypes::PrivateLinkage) {
  // if (!shouldPreserveGV(*F1))
  if (CanEraseF1 && (F1->getNumUses() == 0) && (HasWholeProgram?true:F1->hasLocalLinkage()) &&
      (AlwaysPreserved.find(F1->getName()) == AlwaysPreserved.end())) {
    // CallSiteExtractedLoops.erase(F1);
    F1->eraseFromParent();
  }

  // if (!shouldPreserveGV(*F2))
  if (CanEraseF2 && (F2->getNumUses() == 0) && (HasWholeProgram?true:F2->hasLocalLinkage()) &&
      (AlwaysPreserved.find(F2->getName()) == AlwaysPreserved.end())) {
    // CallSiteExtractedLoops.erase(F2);
    F2->eraseFromParent();
  }
}

bool FMSALegacyPass::shouldPreserveGV(const GlobalValue &GV) {
  // Function must be defined here
  if (GV.isDeclaration())
    return true;

  // Available externally is really just a "declaration with a body".
  if (GV.hasAvailableExternallyLinkage())
    return true;

  // Assume that dllexported symbols are referenced elsewhere
  if (GV.hasDLLExportStorageClass())
    return true;

  // Already local, has nothing to do.
  if (GV.hasLocalLinkage())
    return false;

  // Check some special cases
  if (AlwaysPreserved.find(GV.getName()) != AlwaysPreserved.end())
    return true;

  return false;
}

static bool compareFunctionScores(const std::pair<Function *, unsigned> &F1,
                                  const std::pair<Function *, unsigned> &F2) {
  return F1.second > F2.second;
}

//#define FMSA_USE_JACCARD

class Fingerprint {
public:
  static const size_t MaxOpcode = 65;
  bool builtOnBBs;
  int OpcodeFreq[MaxOpcode];
  // std::map<unsigned, int> OpcodeFreq;
  // size_t NumOfInstructions;
  // size_t NumOfBlocks;

  #ifdef FMSA_USE_JACCARD
  std::set<Type *> Types;
  #else
  std::map<Type*, int> TypeFreq;
  #endif

  Function *F;

  std::vector<BasicBlock *> bbs_in_func;

  Fingerprint(std::vector<BasicBlock *> bbs_in_func) {
    this->builtOnBBs = true;
    this->bbs_in_func = bbs_in_func;

    memset(OpcodeFreq, 0, sizeof(int) * MaxOpcode);
    // for (int i = 0; i<MaxOpcode; i++) OpcodeFreq[i] = 0;

    // NumOfInstructions = 0;
    for (int i = 0; i < bbs_in_func.size(); ++i) {
        BasicBlock *bb = bbs_in_func[i];
        for (BasicBlock::iterator i = bb->begin(); i != bb->end(); ++i) {
          Instruction &I = *i;  
        //for (Instruction &I : instructions(bb)) {
          OpcodeFreq[I.getOpcode()]++;
          /*
                if (OpcodeFreq.find(I.getOpcode()) != OpcodeFreq.end())
                  OpcodeFreq[I.getOpcode()]++;
                else
                  OpcodeFreq[I.getOpcode()] = 1;
          */
          // NumOfInstructions++;

          
          #ifdef FMSA_USE_JACCARD
          Types.insert(I.getType());
          #else
          TypeFreq[I.getType()]++;
          #endif
        }
    // NumOfBlocks = F->size();
    }
  }


  Fingerprint(Function *F) {
    this->builtOnBBs = false;
    this->F = F;

    memset(OpcodeFreq, 0, sizeof(int) * MaxOpcode);
    // for (int i = 0; i<MaxOpcode; i++) OpcodeFreq[i] = 0;

    // NumOfInstructions = 0;
    for (Instruction &I : instructions(F)) {
      OpcodeFreq[I.getOpcode()]++;
      /*
            if (OpcodeFreq.find(I.getOpcode()) != OpcodeFreq.end())
              OpcodeFreq[I.getOpcode()]++;
            else
              OpcodeFreq[I.getOpcode()] = 1;
      */
      // NumOfInstructions++;

      
      #ifdef FMSA_USE_JACCARD
      Types.insert(I.getType());
      #else
      TypeFreq[I.getType()]++;
      #endif
    }
    // NumOfBlocks = F->size();
  }
};

class FingerprintSimilarity {
public:
  Function *F1;
  Function *F2;
  int Similarity;
  int LeftOver;
  int TypesDiff;
  int TypesSim;
  float Score;

  FingerprintSimilarity() : F1(nullptr), F2(nullptr), Score(0.0f) {}

  FingerprintSimilarity(Fingerprint *FP1, Fingerprint *FP2) {
    if (not FP1->builtOnBBs)
        F1 = FP1->F;
    if (not FP2->builtOnBBs)
        F2 = FP2->F;

    Similarity = 0;
    LeftOver = 0;
    TypesDiff = 0;
    TypesSim = 0;

    for (unsigned i = 0; i < Fingerprint::MaxOpcode; i++) {
      int Freq1 = FP1->OpcodeFreq[i];
      int Freq2 = FP2->OpcodeFreq[i];
      int MinFreq = std::min(Freq1, Freq2);
      Similarity += MinFreq;
      LeftOver += std::max(Freq1, Freq2) - MinFreq;
    }
    /*
    for (auto Pair : FP1->OpcodeFreq) {
      if (FP2->OpcodeFreq.find(Pair.first) == FP2->OpcodeFreq.end()) {
        LeftOver += Pair.second;
      } else {
        int MinFreq = std::min(Pair.second, FP2->OpcodeFreq[Pair.first]);
        Similarity += MinFreq;
        LeftOver +=
            std::max(Pair.second, FP2->OpcodeFreq[Pair.first]) - MinFreq;
      }
    }
    for (auto Pair : FP2->OpcodeFreq) {
      if (FP1->OpcodeFreq.find(Pair.first) == FP1->OpcodeFreq.end()) {
        LeftOver += Pair.second;
      }
    }
    */
    
    #ifdef FMSA_USE_JACCARD
    for (auto Ty1 : FP1->Types) {
      if (FP2->Types.find(Ty1) == FP2->Types.end())
        TypesDiff++;
      else
        TypesSim++;
    }
    for (auto Ty2 : FP2->Types) {
      if (FP1->Types.find(Ty2) == FP1->Types.end())
        TypesDiff++;
    }

    float TypeScore = ((float)TypesSim) / ((float)TypesSim + TypesDiff);
    #else
    for (auto Pair : FP1->TypeFreq) {
      if (FP2->TypeFreq.find(Pair.first) == FP2->TypeFreq.end()) {
        TypesDiff += Pair.second;
      } else {
        int MinFreq = std::min(Pair.second, FP2->TypeFreq[Pair.first]);
        TypesSim += MinFreq;
        TypesDiff +=
            std::max(Pair.second, FP2->TypeFreq[Pair.first]) - MinFreq;
      }
    }
    for (auto Pair : FP2->TypeFreq) {
      if (FP1->TypeFreq.find(Pair.first) == FP1->TypeFreq.end()) {
        TypesDiff += Pair.second;
      }
    }
    float TypeScore =
        ((float)TypesSim) / ((float)(TypesSim * 2.0f + TypesDiff));
    #endif
    float UpperBound =
        ((float)Similarity) / ((float)(Similarity * 2.0f + LeftOver));

    #ifdef FMSA_USE_JACCARD
    Score = UpperBound * TypeScore;
    #else
    Score = std::min(UpperBound,TypeScore);
    #endif
  }

  bool operator<(const FingerprintSimilarity &FS) const {
    return Score < FS.Score;
  }

  bool operator>(const FingerprintSimilarity &FS) const {
    return Score > FS.Score;
  }

  bool operator<=(const FingerprintSimilarity &FS) const {
    return Score <= FS.Score;
  }

  bool operator>=(const FingerprintSimilarity &FS) const {
    return Score >= FS.Score;
  }

  bool operator==(const FingerprintSimilarity &FS) const {
    return Score == FS.Score;
  }
};

bool SimilarityHeuristicFilter(const FingerprintSimilarity &Item) {
  if (!ApplySimilarityHeuristic)
    return true;

  if (Item.Similarity < Item.LeftOver)
    return false;

  float TypesDiffRatio = (((float)Item.TypesDiff) / ((float)Item.TypesSim));
  if (TypesDiffRatio > 1.5f)
    return false;

  return true;
}

#ifdef TIME_STEPS_DEBUG
Timer TimePreProcess("Merge::Preprocess", "Merge::Preprocess");
Timer TimeLin("Merge::Lin", "Merge::Lin");
Timer TimeRank("Merge::Rank", "Merge::Rank");
Timer TimeUpdate("Merge::Update", "Merge::Update");
#endif

/*
bool FMSALegacyPass::runOnModule(Module &M) {
  AlwaysPreserved.clear();
  AlwaysPreserved.insert("main");

  std::set< std::string> BlackList;

  srand(time(NULL));

  TargetTransformInfo TTI(M.getDataLayout());

  std::vector<std::pair<Function *, unsigned>> FunctionsToProcess;

  unsigned TotalOpReorder = 0;
  unsigned TotalBinOps = 0;

  std::map<Function *, Fingerprint *> CachedFingerprints;
  std::map<Function *, unsigned> FuncSizes;

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.startTimer();
#endif

  for (auto &F : M) {
    if (F.isDeclaration() || F.isVarArg()) // || F.getSubprogram() != nullptr)
      continue;

    if ( BlackList.count( std::string(F.getName()) ) ) continue;

    FuncSizes[&F] = estimateFunctionSize(F, &TTI); /// TODO

    demoteRegToMem(F);

    FunctionsToProcess.push_back(
      std::pair<Function *, unsigned>(&F, FuncSizes[&F]) );

    CachedFingerprints[&F] = new Fingerprint(&F);
  }

  std::sort(FunctionsToProcess.begin(), FunctionsToProcess.end(),
            compareFunctionScores);

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.stopTimer();
#endif

  std::list<Function *> WorkList;

  std::set<Function *> AvailableCandidates;
  for (std::pair<Function *, unsigned> FuncAndSize1 : FunctionsToProcess) {
    Function *F1 = FuncAndSize1.first;
    WorkList.push_back(F1);
    AvailableCandidates.insert(F1);
  }

  std::vector<FingerprintSimilarity> Rank;
  if (ExplorationThreshold > 1)
    Rank.reserve(FunctionsToProcess.size());

  FunctionsToProcess.clear();

  while (!WorkList.empty()) {
    Function *F1 = WorkList.front();
    WorkList.pop_front();

    AvailableCandidates.erase(F1);

    Rank.clear();

#ifdef TIME_STEPS_DEBUG
    TimeRank.startTimer();
#endif

    Fingerprint *FP1 = CachedFingerprints[F1];

    if (ExplorationThreshold > 1) {
      for (Function *F2 : AvailableCandidates) {
        if (!validMergeTypes(F1, F2) || filterMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (SimilarityHeuristicFilter(PairSim))
          Rank.push_back(PairSim);
      }
      std::make_heap(Rank.begin(), Rank.end());
    } else {

      bool FoundCandidate = false;
      FingerprintSimilarity BestPair;

      for (Function *F2 : AvailableCandidates) {
        if (!validMergeTypes(F1, F2) || filterMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (PairSim > BestPair && SimilarityHeuristicFilter(PairSim)) {
          BestPair = PairSim;
          FoundCandidate = true;
        }
      }
      if (FoundCandidate)
        Rank.push_back(BestPair);
    }

#ifdef TIME_STEPS_DEBUG
    TimeRank.stopTimer();
    TimeLin.startTimer();
#endif

    SmallVector<Value *, 32> F1Vec;
    Linearization(F1, F1Vec, LinearizationKind::LK_Canonical);

#ifdef TIME_STEPS_DEBUG
    TimeLin.stopTimer();
#endif

    unsigned MergingTrialsCount = 0;

    while (!Rank.empty()) {
      auto RankEntry = Rank.front();
      Function *F2 = RankEntry.F2;
      std::pop_heap(Rank.begin(), Rank.end());
      Rank.pop_back();

      CountBinOps = 0;
      CountOpReorder = 0;

      MergingTrialsCount++;

      if (Debug || Verbose) {
        errs() << "Attempting: " << GetValueName(F1) << ", " << GetValueName(F2)
               << "\n";
      }

#ifdef TIME_STEPS_DEBUG
      TimeLin.startTimer();
#endif
      SmallVector<Value *, 32> F2Vec;
      F2Vec.reserve(F1Vec.size());
      Linearization(F2, F2Vec, LinearizationKind::LK_Canonical);

#ifdef TIME_STEPS_DEBUG
      TimeLin.stopTimer();
#endif

      MergedFunction Result = mergeBySequenceAlignment(F1, F2, F1Vec, F2Vec);

      if (Result.MergedFunc != nullptr && verifyFunction(*Result.MergedFunc)) {
        if (Debug || Verbose) {
          errs() << "Invalid Function: " << GetValueName(F1) << ", "
                 << GetValueName(F2) << "\n";
        }
#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          if (Result.MergedFunc != nullptr) {
            Result.MergedFunc->dump();
          }
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
        }
#endif
        Result.MergedFunc->eraseFromParent();
        Result.MergedFunc = nullptr;
      }

      if (Result.MergedFunc) {
        DominatorTree MergedDT(*Result.MergedFunc);
        promoteMemoryToRegister(*Result.MergedFunc, MergedDT);

        unsigned SizeF1 = FuncSizes[F1];
        unsigned SizeF2 = FuncSizes[F2];

        unsigned SizeF12 = requiresOriginalInterfaces(Result) * 3 +
                           estimateFunctionSize(*Result.MergedFunc, &TTI);

#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
          errs() << "F1-F2:\n";
          Result.MergedFunc->dump();
        }
#endif

        if (Debug || Verbose) {
          errs() << "Sizes: " << SizeF1 << " + " << SizeF2 << " <= " << SizeF12 << "?\n";
        }

        if (Debug || Verbose) {
          errs() << "Estimated reduction: "
                 << (int)((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100)
                 << "% ("
                 << (SizeF12 < (SizeF1 + SizeF2) *
                                   ((100.0 + MergingOverheadThreshold) / 100.0))
                 << ") " << MergingTrialsCount << " : " << GetValueName(F1)
                 << "; " << GetValueName(F2) << " | Score " << RankEntry.Score
                 << " | Rough " << Result.RoughReduction << "% ["
                 << (Result.RoughReduction > 1.0) << "]\n";
        }
        // NumOfMergedInsts += maxSimilarity;
        if (SizeF12 <
            (SizeF1 + SizeF2) * ((100.0 + MergingOverheadThreshold) / 100.0)) {

          MergingDistance.push_back(MergingTrialsCount);

          TotalOpReorder += CountOpReorder;
          TotalBinOps += CountBinOps;

          if (Debug || Verbose) {
            errs() << "Merged: " << GetValueName(F1) << ", " << GetValueName(F2)
                   << " = " << GetValueName(Result.MergedFunc) << "\n";
          }

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.startTimer();
#endif

          AvailableCandidates.erase(F2);
          WorkList.remove(F2);

          // update call graph
          UpdateCallGraph(M, Result, AlwaysPreserved);

          // feed new function back into the working lists
          WorkList.push_front(Result.MergedFunc);
          AvailableCandidates.insert(Result.MergedFunc);

          FuncSizes[Result.MergedFunc] =
              estimateFunctionSize(*Result.MergedFunc, &TTI);

          // demote phi instructions
          demoteRegToMem(*Result.MergedFunc);

          CachedFingerprints[Result.MergedFunc] =
              new Fingerprint(Result.MergedFunc);

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.stopTimer();
#endif

          break; // end exploration

        } else {
          if (Result.MergedFunc != nullptr)
            Result.MergedFunc->eraseFromParent();
        }
      }

      if (MergingTrialsCount >= ExplorationThreshold) {
        break;
      }
    }
  }

  WorkList.clear();

  for (auto kv : CachedFingerprints) {
    delete kv.second;
  }
  CachedFingerprints.clear();

  double MergingAverageDistance = 0;
  unsigned MergingMaxDistance = 0;
  for (unsigned Distance : MergingDistance) {
    MergingAverageDistance += Distance;
    if (Distance > MergingMaxDistance)
      MergingMaxDistance = Distance;
  }
  if (MergingDistance.size() > 0) {
    MergingAverageDistance = MergingAverageDistance / MergingDistance.size();
  }

  if (Debug || Verbose) {
    errs() << "Total operand reordering: " << TotalOpReorder << "/"
           << TotalBinOps << " ("
           << 100.0 * (((double)TotalOpReorder) / ((double)TotalBinOps))
           << " %)\n";

    errs() << "Total parameter score: " << TotalParamScore << "\n";

    errs() << "Total number of merges: " << MergingDistance.size() << "\n";
    errs() << "Average number of trials before merging: "
           << MergingAverageDistance << "\n";
    errs() << "Maximum number of trials before merging: " << MergingMaxDistance
           << "\n";
  }

#ifdef TIME_STEPS_DEBUG
  errs() << "Timer:Align: " << TimeAlign.getTotalTime().getWallTime() << "\n";
  TimeAlign.clear();

  errs() << "Timer:Param: " << TimeParam.getTotalTime().getWallTime() << "\n";
  TimeParam.clear();

  errs() << "Timer:CodeGen1: " << TimeCodeGen1.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen1.clear();

  errs() << "Timer:CodeGen2: " << TimeCodeGen2.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen2.clear();

  errs() << "Timer:CodeGenFix: " << TimeCodeGenFix.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGenFix.clear();

  errs() << "Timer:PreProcess: " << TimePreProcess.getTotalTime().getWallTime()
         << "\n";
  TimePreProcess.clear();

  errs() << "Timer:Lin: " << TimeLin.getTotalTime().getWallTime() << "\n";
  TimeLin.clear();

  errs() << "Timer:Rank: " << TimeRank.getTotalTime().getWallTime() << "\n";
  TimeRank.clear();

  errs() << "Timer:Update: " << TimeUpdate.getTotalTime().getWallTime() << "\n";
  TimeUpdate.clear();
#endif

  return true;
}
*/

void FMSALegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {}

char FMSALegacyPass::ID = 0;
// static RegisterPass<FMSALegacyPass> X1("my-func-merge", "My Function
// Merging.", false, false);
INITIALIZE_PASS(FMSALegacyPass, "fmsa", "New Function Merging", false,
                false)


void FMSALegacyPass::LoadRealFuncArea(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: Function merging couldn't load file '" << Filename
           << "'!\n";
    return;
  }
  while (In) {
    std::string FuncName;
	double funcArea;
    In >> FuncName >> funcArea;
    errs() << "Loading func " << FuncName << " with area size " << funcArea << " luts\n";
    if (!FuncName.empty()) {
      FMSALegacyPass::realAreaPerFunc[FuncName] = funcArea;
      //bbProfiles.push_back(BasicBlockDynProfile(FunctionName, BlockName, numInstructions));
    }
  }
}

void FMSALegacyPass::LoadAreaModel(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: Function merging couldn't load file '" << Filename
           << "'!\n";
    return;
  }
  while (In) {
    std::string OpName;
	double lutsPerOp;
    In >> OpName >> lutsPerOp;
    errs() << "Loading op " << OpName << " with " << lutsPerOp << " luts\n";
    if (!OpName.empty()) {
      FMSALegacyPass::opLuts[OpName] = lutsPerOp;
      //bbProfiles.push_back(BasicBlockDynProfile(FunctionName, BlockName, numInstructions));
    }
  }
}

void FMSALegacyPass::LoadFile(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  errs() << "You made it to here 1\n";
  if (!In.good()) {
    errs() << "WARNING: Function merging couldn't load file '" << Filename
           << "'!\n";
    exit(-1);
    return;
  }
  errs() << "You made it to here 2\n";
  while (In) {
    std::string FunctionName, BlockName;
	double percAppInstructions;
    In >> FunctionName;
    In >> BlockName;
    In >> percAppInstructions;
    errs() << "Loading function " << FunctionName << " and bb " << BlockName << " and perc of instructions " << percAppInstructions << "\n";
    if (!BlockName.empty()) {
      errs() << "Reading the profile for bb " + FunctionName + "::" + BlockName << " to be " << percAppInstructions << "\n";
      FMSALegacyPass::bbProfiles["@" + FunctionName + "::" + BlockName] = percAppInstructions;
      //bbProfiles.push_back(BasicBlockDynProfile(FunctionName, BlockName, numInstructions));
    }
  }
}


void FMSALegacyPass::LoadBlacklistFile(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: Function merging couldn't load Blacklistile '" << Filename
           << "'!\n";
    exit(-1);
    return;
  }
  while (In) {
    std::string FunctionName;
    In >> FunctionName;
    if (!FunctionName.empty()) {
      BlackList.insert(FunctionName);
      //errs() << "Reading the profile for bb " + FunctionName + "::" + BlockName << " to be " << percAppInstructions << "\n";
      //FMSALegacyPass::bbProfiles["@" + FunctionName + "::" + BlockName] = percAppInstructions;
      //bbProfiles.push_back(BasicBlockDynProfile(FunctionName, BlockName, numInstructions));
    }
  }
}

static bool canFind(Function &F, std::map<std::string, double> &latAtMergeTime, double &lat) {
    auto it = latAtMergeTime.find(GetValueName(&F));
    if (it != latAtMergeTime.end()) {
        lat = it->second;
        return true;
    } else
        return false;
}

static bool isProfitableAccelMerge(Function &F1, Function &F2, Function &F12) {
    double f1_sw;
    double f1_hw;
    double f2_sw;
    double f2_hw;
    double f12_sw;
    double f12_hw;

    bool profitableAccelMerge = true;
    if (not FMSALegacyPass::bbProfiles.empty() and canFind(F1, hwLatAtMergeTime, f1_hw) and canFind(F1, swLatAtMergeTime, f1_sw) 
                               and canFind(F2, hwLatAtMergeTime, f2_hw) and canFind(F2, swLatAtMergeTime, f2_sw)
                               and canFind(F12, hwLatAtMergeTime, f12_hw) and canFind(F12, swLatAtMergeTime, f12_sw)) {
        if (f1_hw + f2_hw > f12_hw)
            f12_hw = f1_hw + f2_hw;
        if (f1_sw + f2_sw > f12_sw)
            f12_sw = f1_sw + f2_sw;
        profitableAccelMerge = f1_sw + f2_sw - f12_hw - std::max(f1_sw - f1_hw, f2_sw - f2_hw) > 0;
        errs() << "Profitable accel " << profitableAccelMerge << "\n";
        errs() << "f1_sw " << f1_sw << " f1_hw " << f1_hw << "\n";
        errs() << "f2_sw " << f2_sw << " f2_hw " << f2_hw << "\n";
        errs() << "f12_sw " << f12_sw << " f12_hw " << f12_hw << "\n";
        errs() << "Merged benefit " << f1_sw + f2_sw - f12_hw << "\n";
        errs() << "Unmerged benefit " << std::max(f1_sw - f1_hw, f2_sw - f2_hw) << "\n";
    }
    return profitableAccelMerge;

}
static void printBBsWithTheirDynProf(Function *func) {
    errs() << "Printing bbs for " << GetValueName(func) << "\n";
    for (auto &BB : *func) {
        auto bbProfilesKey = GetValueName(func) + "::" + BB.getName().str();
        if (FMSALegacyPass::mergedBBProfiles.find(&BB) != FMSALegacyPass::mergedBBProfiles.end())
            errs() << GetValueName(&BB) << " : " << FMSALegacyPass::mergedBBProfiles[&BB] << " a merged bb\n";
        else if (FMSALegacyPass::bbProfiles.find(bbProfilesKey) != FMSALegacyPass::bbProfiles.end())
            errs() << GetValueName(&BB) << " : " << FMSALegacyPass::bbProfiles[bbProfilesKey] << " a nonmerged bb\n";
        else
            errs() << GetValueName(&BB) << " : not found in bbProfiles or mergedBBProfiles\n";
    }
}




////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

bool FMSALegacyPass::runOnModule(Module &M) {
  errs() << "The received file name is " << std::string(DynamicProfileFile.c_str()) << "\n";
  errs() << "Check1 " << (std::string(DynamicProfileFile.c_str()) != std::string("filename")) << "\n";
  errs() << (std::string(BlacklistFile.c_str()) != std::string("")) << "\n";
  if (std::string(DynamicProfileFile.c_str()) != "filename" and std::string(DynamicProfileFile.c_str()) != "") {
    errs() << "Made it to before the call\n";
    FMSALegacyPass::LoadFile(DynamicProfileFile.c_str());
  }
 
  if (std::string(AreaModelFile.c_str()) != "filename" and std::string(AreaModelFile.c_str()) != "")
    LoadAreaModel(AreaModelFile.c_str());

  if (std::string(RealAreaFile.c_str()) != "filename" and std::string(RealAreaFile.c_str()) != "")
    LoadRealFuncArea(RealAreaFile.c_str());

  if (std::string(BlacklistFile.c_str()) != "filename" and std::string(BlacklistFile.c_str()) != "")
    LoadBlacklistFile(BlacklistFile.c_str());

  AlwaysPreserved.clear();
  
  AlwaysPreserved.insert("main");


  srand(time(NULL));

  TargetTransformInfo TTI(M.getDataLayout());

  std::vector<std::pair<Function *, double>> FunctionsToProcess;

  unsigned TotalOpReorder = 0;
  unsigned TotalBinOps = 0;

  std::map<Function *, Fingerprint *> CachedFingerprints;
  std::map<Function *, double> FuncSizes;

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.startTimer();
#endif


  double total_size = 0;
 // for (auto it = FuncSizes.begin(); it != FuncSizes.end(); ++it) {
 //   total_size += it->second;
 // }
 // errs() << "The total size of the original code is " << total_size << "\n";
    

  double total_initial_area = 0.0;
  for (auto &F : M) {
    total_initial_area += estimateFunctionLatencyOrArea(F, AREA_ACCEL);
    if (F.isDeclaration() || F.isVarArg()) // || F.getSubprogram() != nullptr)
      continue;
    std::string entryBlockName = GetValueName(&F) + "::" + F.getEntryBlock().getName().str();

    if ((not bbProfiles.empty() and bbProfiles.find(entryBlockName) == bbProfiles.end()) ) {
        errs() << "Not considering function " << GetValueName(&F) << "\n";
        errs() << "The key you were looking for " << entryBlockName << "\n";
        if (entryBlockName == "") {
            errs() << "The entry basic block name is empty. Most likely the app wasn't compiled properly saving symbols";
            exit(-1);
        }
        continue;
    }

    //if ( BlackList.count( std::string(F.getName()) ) or GetValueName(&F) == "@" + std::string(BlacklistedFunc.c_str()) ) continue;
    double timeSize = 0.0;


    double func_size = estimateFunctionSize(F, &TTI, timeSize);
    funcTimeSize[&F] = timeSize;
    FuncSizes[&F] = func_size; /// TODO

    errs() << "The size for function " << F.getName().str() << " is " << FuncSizes[&F] << "\n";
    total_size += func_size;
    //Iuli: you commented this
    demoteRegToMem(F);
    

    FunctionsToProcess.push_back(
      std::pair<Function *, double>(&F, FuncSizes[&F]) );

    CachedFingerprints[&F] = new Fingerprint(&F);
  }

  errs() << "The total size is " << total_size << "\n";

  std::sort(FunctionsToProcess.begin(), FunctionsToProcess.end(),
            compareFunctionScores);

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.stopTimer();
#endif

  std::list<Function *> WorkList;

  std::set<Function *> AvailableCandidates;
  for (std::pair<Function *, double> FuncAndSize1 : FunctionsToProcess) {
    Function *F1 = FuncAndSize1.first;
    WorkList.push_back(F1);
    AvailableCandidates.insert(F1);
  }

  std::vector<FingerprintSimilarity> Rank;
  if (ExplorationThreshold > 1)
    Rank.reserve(FunctionsToProcess.size());

  FunctionsToProcess.clear();

  Verbose = Debug = true;

  while (!WorkList.empty()) {
    errs() << "The worklist size is " << WorkList.size() << "\n";
    Function *F1 = WorkList.front();
    WorkList.pop_front();

    AvailableCandidates.erase(F1);

    Rank.clear();

#ifdef TIME_STEPS_DEBUG
    TimeRank.startTimer();
#endif

    Fingerprint *FP1 = CachedFingerprints[F1];

    if (ExplorationThreshold > 1 || RunBruteForceExploration) {
      for (Function *F2 : AvailableCandidates) {
        if (!validMergeTypes(F1, F2) || filterMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (SimilarityHeuristicFilter(PairSim))
          Rank.push_back(PairSim);
      }
      if (!RunBruteForceExploration)
        std::make_heap(Rank.begin(), Rank.end());
    } else {
      bool FoundCandidate = false;
      FingerprintSimilarity BestPair;

      for (Function *F2 : AvailableCandidates) {
        if (!validMergeTypes(F1, F2) || filterMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (PairSim > BestPair && SimilarityHeuristicFilter(PairSim)) {
          BestPair = PairSim;
          FoundCandidate = true;
        }
      }
      if (FoundCandidate)
        Rank.push_back(BestPair);
    }
#ifdef TIME_STEPS_DEBUG
    TimeRank.stopTimer();
    TimeLin.startTimer();
#endif

    SmallVector<Value *, 32> F1Vec;
    std::list<BasicBlock *> OrderedBBs1;
    Linearization(F1, F1Vec, OrderedBBs1, LinearizationKind::LK_Canonical);

#ifdef TIME_STEPS_DEBUG
    TimeLin.stopTimer();
#endif

    unsigned MergingTrialsCount = 0;

    Function *BestCandidate = nullptr;
    int BestReduction = INT_MIN;

    while (!Rank.empty()) {
      //errs() << "The rank size for function " << F1->getName().str() << " is " <<  Rank.size() << "\n";
      Function *F2 = nullptr;
      if (RunBruteForceExploration) {
        auto RankEntry = Rank.back();
        F2 = RankEntry.F2;
        Rank.pop_back();
      } else {
        auto RankEntry = Rank.front();
        F2 = RankEntry.F2;
        std::pop_heap(Rank.begin(), Rank.end());
        Rank.pop_back();
      }

      CountBinOps = 0;
      CountOpReorder = 0;

      MergingTrialsCount++;
      errs() << "Test msg1\n";
      if (Debug || Verbose) {
        errs() << "Attempting: " << GetValueName(F1) << ", " << GetValueName(F2)
               << "\n";
      }

#ifdef TIME_STEPS_DEBUG
      TimeLin.startTimer();
#endif
      SmallVector<Value *, 32> F2Vec;
      F2Vec.reserve(F1Vec.size());

      std::list<BasicBlock *> OrderedBBs2;
      Linearization(F2, F2Vec, OrderedBBs2, LinearizationKind::LK_Canonical);

#ifdef TIME_STEPS_DEBUG
      TimeLin.stopTimer();
#endif

      MergedFunction Result = mergeBySequenceAlignment(F1, F2, F1Vec, F2Vec, OrderedBBs1, OrderedBBs2);

      if (Result.MergedFunc != nullptr && verifyFunction(*Result.MergedFunc)) {
        if (Debug || Verbose) {
          errs() << "Invalid Function: " << GetValueName(F1) << ", "
                 << GetValueName(F2) << "\n";

          //Result.MergedFunc->dump();
          //double SizeF1 = FuncSizes[F1];
          //unsigned SizeF2 = FuncSizes[F2];

          //unsigned SizeF12 = requiresOriginalInterfaces(Result) * 3 +
          //                   estimateFunctionSize(*Result.MergedFunc, &TTI);
          //errs() << "The size of F1 is " << SizeF1 << " the one for F2 is " << SizeF2 << " and the one for F12 is " << SizeF12 << "\n";
        }
#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          if (Result.MergedFunc != nullptr) {
            Result.MergedFunc->dump();
          }
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
        }
#endif
        Result.MergedFunc->eraseFromParent();
        Result.MergedFunc = nullptr;
      }

      //if (Result.MergedFunc and (GetValueName(F1).find("viterbi") == std::string::npos or GetValueName(F2).find("viterbi") == std::string::npos)) 
      if (Result.MergedFunc) {

        
        double SizeF1 = FuncSizes[F1];
        double SizeF2 = FuncSizes[F2];
        double timeSize1 = funcTimeSize[F1];
        double timeSize2 = funcTimeSize[F2];
        double timeSize12 = 0.0;


        std::string smallName, largeName;
        if (F1->getName().str() < F2->getName().str()) {
            smallName = F1->getName().str();
            largeName = F2->getName().str();
        } else {
            smallName = F2->getName().str();
            largeName = F1->getName().str();
        }
            
        //Result.MergedFunc->setName("merged" + std::to_string(merged_func_id));
        Result.MergedFunc->setName("merged_" + smallName + "_" + largeName);
        //need to change name before getting area size because otherwise cache is unusable for result

        //demoteRegToMem(*Result.MergedFunc);
        double SizeF12 = estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize12);
        
        
        if ( BlackList.count( std::string((*Result.MergedFunc).getName().str()) )) {
            SizeF12 = -1;
            SizeF1 = 0;
            SizeF2 = 0;
        } else if (FMSALegacyPass::realAreaPerFunc.find((*Result.MergedFunc).getName().str()) != FMSALegacyPass::realAreaPerFunc.end()) {
            SizeF12 = FMSALegacyPass::realAreaPerFunc[(*Result.MergedFunc).getName().str()];
            SizeF1 = FMSALegacyPass::realAreaPerFunc[F1->getName().str()];
            SizeF2 = FMSALegacyPass::realAreaPerFunc[F2->getName().str()];
        } else if (not opLuts.empty()) {
            double lutDifference = estimateFunctionDifference(*Result.MergedFunc, *F1, *F2);
            SizeF12 = 1;
            SizeF1 = 0.5 + lutDifference/2;
            SizeF2 = 0.5 + lutDifference/2;
        }
        //DominatorTree MergedDT(*Result.MergedFunc);
        //promoteMemoryToRegister(*Result.MergedFunc, MergedDT);


        //double SizeF12 = estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize12);
        errs() << "Time size 1 is " << timeSize1 << " , time size 2 is " << timeSize2 << " and timeSize12 is " << timeSize12 << "\n";
        errs() << "Area size 1 is " << SizeF1 << " , area size 2 is " << SizeF2 << " and areaSize12 is " << SizeF12 << "\n";


        double timeSlowdown = timeSize12/(timeSize1 + timeSize2); //I remember that the predicted summed times didn't account for more than the merged
                                                                     //but that's not correct and you need to fix it
        double areaWin = ((SizeF1+SizeF2)/SizeF12);
        errs() << "Area win is " << areaWin << " and time slowdown is " << timeSlowdown << "\n";

        double max12 = std::max(timeSize1, timeSize2);
        double min12 = std::min(timeSize1, timeSize2);
            

        
        if ((1000*((max12 - min12)/max12) > LoadImbalanceThreshold) or (LoadImbalanceThreshold > 900000.0 and (timeSize1 < 0.0000001 or timeSize2 < 0.0000001 ))) {
            SizeF12 = 99999999;

        } else
            codeAndTimeReduction.push_back(
          std::pair<double, double>(((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100), (max12 - min12)/max12));
        
        errs() << "Load imbalance is " << (max12 - min12)/max12 << "\n";
        funcTimeSize[Result.MergedFunc] = timeSize12;


#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
          errs() << "F1-F2:\n";
          Result.MergedFunc->dump();
        }
#endif

        //if (Debug || Verbose) {
          errs() << "Sizes: " << SizeF1 << " + " << SizeF2 << " <= " << SizeF12 << "?\n";
        //}

        if ((Debug || Verbose) && false) {
          errs() << "Estimated reduction: "
                 << (int)((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100)
                 << "% ("
                 << (SizeF12 < (SizeF1 + SizeF2) *
                                   ((100.0 + MergingOverheadThreshold) / 100.0))
                 << ") " << MergingTrialsCount << " : " << GetValueName(F1)
                 << "; " << GetValueName(F2) << "\n";
                 //<< "; " << GetValueName(F2) << " | Score " << RankEntry.Score
                 //<< " | Rough " << Result.RoughReduction << "% ["
                 //<< (Result.RoughReduction > 1.0) << "]\n";
        }
        // NumOfMergedInsts += maxSimilarity;
        //if (SizeF12 <
        //    (SizeF1 + SizeF2) * ((100.0 + MergingOverheadThreshold) / 100.0)) 

        bool profitableAccelMerge = isProfitableAccelMerge(*F1, *F2, (*Result.MergedFunc));
        profitableAccelMerge = true;
        if ((not bbProfiles.empty() and areaWin/timeSlowdown > AreaWinToSlowdownRatio and profitableAccelMerge and  areaWin > 1.0) or
          (bbProfiles.empty() and areaWin > 1.0)) {
            
            //(SizeF1 + SizeF2) * ((100.0 + MergingOverheadThreshold) / 100.0))

          if (RunBruteForceExploration) {
            int Reduction = (int)(SizeF1 + SizeF2) - ((int)SizeF12);
            if (Reduction > BestReduction) {
               BestReduction = Reduction;
               BestCandidate = F2;
            }
            Result.MergedFunc->eraseFromParent();
            continue;
          }

          MergingDistance.push_back(MergingTrialsCount);

          TotalOpReorder += CountOpReorder;
          TotalBinOps += CountBinOps;

          if (timeSize1 + timeSize2 < timeSize12) 
              swLatAtMergeTime[GetValueName(Result.MergedFunc)] = timeSize1 + timeSize2;

          if (hwLatAtMergeTime[GetValueName(F1)] + hwLatAtMergeTime[GetValueName(F2)] < hwLatAtMergeTime[GetValueName(Result.MergedFunc)])
              hwLatAtMergeTime[GetValueName(Result.MergedFunc)] = hwLatAtMergeTime[GetValueName(F1)] + hwLatAtMergeTime[GetValueName(F2)];
          
          if (Debug || Verbose) {
            merged_func_id++;
            errs() << "Merged: " << GetValueName(F1) << ", " << GetValueName(F2)
                   << " = " << GetValueName(Result.MergedFunc) << "\n";
            errs() << "The merged func bbs \n";
            printBBsWithTheirDynProf(Result.MergedFunc);
            errs() << "The src1 func bbs \n";
            printBBsWithTheirDynProf(F1);
            errs() << "The src2 func bbs \n";
            printBBsWithTheirDynProf(F2);
            if (LatencyMerges and not FMSALegacyPass::bbProfiles.empty()){
                double timeSize = 0.0; 
                double areaOvh = estimateFunctionTimeOvh(*(Result.MergedFunc), &TTI, timeSize);
                errs() << "The time spent in " << GetValueName(Result.MergedFunc) << " is " << timeSize << "\n";
                errs() << "You are outputing this here because you have the info of which bbs were used for the merging.\n";
            }
          }

          mergedFunctions.insert(GetValueName(F1));
          mergedFunctions.insert(GetValueName(F2));


#ifdef TIME_STEPS_DEBUG
          TimeUpdate.startTimer();
#endif

          AvailableCandidates.erase(F2);
          WorkList.remove(F2);

          // update call graph
          UpdateCallGraph(M, Result, AlwaysPreserved);

          // feed new function back into the working lists
          WorkList.push_front(Result.MergedFunc);
          AvailableCandidates.insert(Result.MergedFunc);

          double timeSize = 0.0;
          FuncSizes[Result.MergedFunc] =
              estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize);


          funcTimeSize[Result.MergedFunc] = timeSize;
          // demote phi instructions
          //Iuli: you commented this
          //demoteRegToMem(*Result.MergedFunc);

          CachedFingerprints[Result.MergedFunc] =
              new Fingerprint(Result.MergedFunc);

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.stopTimer();
#endif

          break; // end exploration

        } else {
          if (Result.MergedFunc != nullptr)
            Result.MergedFunc->eraseFromParent();
        }
      }

      if (MergingTrialsCount >= ExplorationThreshold && !RunBruteForceExploration) {
        break;
      }
    }
    if (BestCandidate!=nullptr && RunBruteForceExploration) {
      Function *F2 = BestCandidate;

      SmallVector<Value *, 32> F2Vec;
      F2Vec.reserve(F1Vec.size());

      std::list<BasicBlock *> OrderedBBs2;
      Linearization(F2, F2Vec, OrderedBBs2, LinearizationKind::LK_Canonical);

      MergedFunction Result = mergeBySequenceAlignment(F1, F2, F1Vec, F2Vec, OrderedBBs1, OrderedBBs2);

      if (Result.MergedFunc != nullptr && verifyFunction(*Result.MergedFunc)) {
        if (Debug || Verbose) {
          errs() << "Invalid Function: " << GetValueName(F1) << ", "
                 << GetValueName(F2) << "\n";
        }
#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          if (Result.MergedFunc != nullptr) {
            Result.MergedFunc->dump();
          }
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
        }
#endif
        Result.MergedFunc->eraseFromParent();
        Result.MergedFunc = nullptr;
      }

      //if (Result.MergedFunc and (GetValueName(F1).find("viterbi") == std::string::npos or GetValueName(F2).find("viterbi") == std::string::npos)) 
      if (Result.MergedFunc) {

        double SizeF1 = FuncSizes[F1];
        double SizeF2 = FuncSizes[F2];
        double timeSize1 = funcTimeSize[F1];
        double timeSize2 = funcTimeSize[F2];
        double timeSize12 = 0.0;

        std::string smallName, largeName;
        if (F1->getName().str() < F2->getName().str()) {
            smallName = F1->getName().str();
            largeName = F2->getName().str();
        } else {
            smallName = F2->getName().str();
            largeName = F1->getName().str();
        }

        //Result.MergedFunc->setName("merged" + std::to_string(merged_func_id));
        Result.MergedFunc->setName("merged_" + smallName + "_" + largeName);


        //demoteRegToMem(*Result.MergedFunc);
        //double SizeF12 = requiresOriginalInterfaces(Result) * 3 +
        //                   estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize12);
        double SizeF12 = estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize12);
      
    //0<f1+f2-f12
    //1<(f1+f2)/f2

        
        if (BlackList.count( std::string((*Result.MergedFunc).getName().str()))) {
            SizeF12 = -1;
            SizeF1 = 0;
            SizeF2 = 0;
        } else if (FMSALegacyPass::realAreaPerFunc.find((*Result.MergedFunc).getName().str()) != FMSALegacyPass::realAreaPerFunc.end()) {
            SizeF12 = FMSALegacyPass::realAreaPerFunc[(*Result.MergedFunc).getName().str()];
            SizeF1 = FMSALegacyPass::realAreaPerFunc[F1->getName().str()];
            SizeF2 = FMSALegacyPass::realAreaPerFunc[F2->getName().str()];
        } else if (not opLuts.empty()) { //if we haven't cached the real area numbers, at least we can use the external model if available
            double lutDifference = estimateFunctionDifference(*Result.MergedFunc, *F1, *F2);
            SizeF12 = 1;
            SizeF1 = 0.5 + lutDifference/2;
            SizeF2 = 0.5 + lutDifference/2;
        }
        
        //the most correct course of action here is to measure function sizes with demoted
        //memory for all the functions
        //DominatorTree MergedDT(*Result.MergedFunc);
        //promoteMemoryToRegister(*Result.MergedFunc, MergedDT);

        double timeSlowdown = timeSize12/(timeSize1 + timeSize2); //I remember that the predicted summed times didn't account for more than the merged
                                                                     //but that's not correct and you need to fix it
        double areaWin = ((SizeF1+SizeF2)/SizeF12);
        errs() << "Area win is " << areaWin << "and time slowdown is " << timeSlowdown << "\n";

        //double SizeF12 = estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize12);
        errs() << "Time size 1 is " << timeSize1 << " , time size 2 is " << timeSize2 << " and timeSize12 is " << timeSize12 << "\n";
        double max12 = std::max(timeSize1, timeSize2);
        double min12 = std::min(timeSize1, timeSize2);
        errs() << "Load imbalance is " << (max12 - min12)/max12 << "\n";
        if (1000*(max12 - min12)/max12 > LoadImbalanceThreshold)
            SizeF12 = 99999999;
        else
            codeAndTimeReduction.push_back(
              std::pair<double, double>(((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100), (max12 - min12)/max12));
        funcTimeSize[Result.MergedFunc] = timeSize12;

#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
          errs() << "F1-F2:\n";
          Result.MergedFunc->dump();
        }
#endif

        //if (Debug || Verbose) {
          errs() << "Sizes: " << SizeF1 << " + " << SizeF2 << " <= " << SizeF12 << "?\n";
        //}

        if (Debug || Verbose) {
          errs() << "Estimated reduction: "
                 << (int)((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100)
                 << "% ("
                 << (SizeF12 < (SizeF1 + SizeF2) *
                                   ((100.0 + MergingOverheadThreshold) / 100.0))
                 << ") " << MergingTrialsCount << " : " << GetValueName(F1)
                 << "; " << GetValueName(F2) << "\n";
                 //" | Score " << RankEntry.Score
                 //<< " | Rough " << Result.RoughReduction << "% ["
                 //<< (Result.RoughReduction > 1.0) << "]\n";
        }
        // NumOfMergedInsts += maxSimilarity;
        //if (SizeF12 <
        //    (SizeF1 + SizeF2) * ((100.0 + MergingOverheadThreshold) / 100.0)) 
        

        bool profitableAccelMerge = isProfitableAccelMerge(*F1, *F2, (*Result.MergedFunc));
        profitableAccelMerge = true;

        if ((not bbProfiles.empty() and areaWin/timeSlowdown > AreaWinToSlowdownRatio and areaWin > 1.0 and profitableAccelMerge) or
            (bbProfiles.empty() and areaWin > 1.0)) {

          MergingDistance.push_back(MergingTrialsCount);

          TotalOpReorder += CountOpReorder;
          TotalBinOps += CountBinOps;

          if (timeSize1 + timeSize2 < timeSize12) {
              swLatAtMergeTime[GetValueName(Result.MergedFunc)] = timeSize1 + timeSize2;
          }

          if (hwLatAtMergeTime[GetValueName(F1)] + hwLatAtMergeTime[GetValueName(F2)] < hwLatAtMergeTime[GetValueName(Result.MergedFunc)])
              hwLatAtMergeTime[GetValueName(Result.MergedFunc)] = hwLatAtMergeTime[GetValueName(F1)] + hwLatAtMergeTime[GetValueName(F2)];

          if (numCallsAtMergeTime[GetValueName(F1)] + numCallsAtMergeTime[GetValueName(F2)] < numCallsAtMergeTime[GetValueName(Result.MergedFunc)])
              numCallsAtMergeTime[GetValueName(Result.MergedFunc)] = numCallsAtMergeTime[GetValueName(F1)] + numCallsAtMergeTime[GetValueName(F2)];
          if (Debug || Verbose) {

            merged_func_id++;
            errs() << "Merged: " << GetValueName(F1) << ", " << GetValueName(F2)
                   << " = " << GetValueName(Result.MergedFunc) << "\n";

          }
          mergedFunctions.insert(GetValueName(F1));
          mergedFunctions.insert(GetValueName(F2));

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.startTimer();
#endif

          AvailableCandidates.erase(F2);
          WorkList.remove(F2);

          // update call graph
          UpdateCallGraph(M, Result, AlwaysPreserved);

          // feed new function back into the working lists
          WorkList.push_front(Result.MergedFunc);
          AvailableCandidates.insert(Result.MergedFunc);
          double timeSize = 0.0;
          FuncSizes[Result.MergedFunc] =
              estimateFunctionSize(*Result.MergedFunc, &TTI, timeSize);
          errs() << "Initial size for function " << GetValueName(Result.MergedFunc) << " is " << FuncSizes[Result.MergedFunc] << "\n";

          funcTimeSize[Result.MergedFunc] = timeSize;

          // demote phi instructions
          //Iuli: you commented this
          //demoteRegToMem(*Result.MergedFunc);

          CachedFingerprints[Result.MergedFunc] =
              new Fingerprint(Result.MergedFunc);

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.stopTimer();
#endif

        } else {
          if (Result.MergedFunc != nullptr)
            Result.MergedFunc->eraseFromParent();
        }
      }

    }
  }

  WorkList.clear();

  for (auto kv : CachedFingerprints) {
    delete kv.second;
  }
  CachedFingerprints.clear();

  double MergingAverageDistance = 0;
  unsigned MergingMaxDistance = 0;
  for (unsigned Distance : MergingDistance) {
    MergingAverageDistance += Distance;
    if (Distance > MergingMaxDistance)
      MergingMaxDistance = Distance;
  }
  if (MergingDistance.size() > 0) {
    MergingAverageDistance = MergingAverageDistance / MergingDistance.size();
  }

  if (Debug || Verbose) {
    errs() << "Total operand reordering: " << TotalOpReorder << "/"
           << TotalBinOps << " ("
           << 100.0 * (((double)TotalOpReorder) / ((double)TotalBinOps))
           << " %)\n";

    errs() << "Total parameter score: " << TotalParamScore << "\n";

    errs() << "Total number of merges: " << MergingDistance.size() << "\n";
    errs() << "Average number of trials before merging: "
           << MergingAverageDistance << "\n";
    errs() << "Maximum number of trials before merging: " << MergingMaxDistance
           << "\n";
  }

#ifdef TIME_STEPS_DEBUG
  errs() << "Timer:Align: " << TimeAlign.getTotalTime().getWallTime() << "\n";
  TimeAlign.clear();

  errs() << "Timer:Param: " << TimeParam.getTotalTime().getWallTime() << "\n";
  TimeParam.clear();

  errs() << "Timer:CodeGen1: " << TimeCodeGen1.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen1.clear();

  errs() << "Timer:CodeGen2: " << TimeCodeGen2.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen2.clear();

  errs() << "Timer:CodeGenFix: " << TimeCodeGenFix.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGenFix.clear();

  errs() << "Timer:PreProcess: " << TimePreProcess.getTotalTime().getWallTime()
         << "\n";
  TimePreProcess.clear();

  errs() << "Timer:Lin: " << TimeLin.getTotalTime().getWallTime() << "\n";
  TimeLin.clear();

  errs() << "Timer:Rank: " << TimeRank.getTotalTime().getWallTime() << "\n";
  TimeRank.clear();

  errs() << "Timer:Update: " << TimeUpdate.getTotalTime().getWallTime() << "\n";
  TimeUpdate.clear();
#endif

  double total_final_area = 0;
  double total_old_final_area = 0;

  for (auto &F : M) {

    if (not F.isDeclaration()){
        if (areaAtMergeTime.find(GetValueName(&F)) != areaAtMergeTime.end()) {
            errs() << "The accelSeeker area size for " <<  GetValueName(&F) << " is " << std::to_string(areaAtMergeTime[GetValueName(&F)]) << "\n";
            errs() << "The accelSeeker hw latency for " <<  GetValueName(&F) << " is " << std::to_string(hwLatAtMergeTime[GetValueName(&F)]) << "\n";
            errs() << "The accelSeeker sw latency for " <<  GetValueName(&F) << " is " << std::to_string(swLatAtMergeTime[GetValueName(&F)]) << "\n";
            errs() << "The accelSeeker code size for " <<  GetValueName(&F) << " is " << std::to_string(funcCodeSizeAtMergeTime[GetValueName(&F)]) << "\n";
            //errs() << "The number of calls for " <<  GetValueName(&F) << " is " << std::to_string(numCallsAtMergeTime[GetValueName(&F)]) << "\n";
          total_final_area += areaAtMergeTime[GetValueName(&F)];
        } else {

            double estimated_area = estimateFunctionLatencyOrArea(F, AREA_ACCEL);
            errs() << "The accelSeeker area size for " <<  GetValueName(&F) << " is " << std::to_string(estimated_area) << "\n";
            errs() << "The accelSeeker hw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, HW_LATENCY_ACCEL)) << "\n";
            errs() << "The accelSeeker sw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, SW_LATENCY_ACCEL)) << "\n";
            errs() << "The accelSeeker code size for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionCodeSize(F, &TTI)) << "\n";
            //errs() << "The number of calls for " <<  GetValueName(&F) << " is " << std::to_string(estimateNumberOfCallsForFunc(F)) << "\n";
            estimateNumberOfCallsForFunc(F); 
            total_final_area += estimated_area;
        }
        
        double estimated_area = estimateFunctionLatencyOrArea(F, AREA_ACCEL);
        total_old_final_area += estimated_area;
        //errs() << "The old area size for " <<  GetValueName(&F) << " is " << std::to_string(estimated_area) << "\n";
        //errs() << "The old hw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, HW_LATENCY_ACCEL)) << "\n";
        //errs() << "The old sw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, SW_LATENCY_ACCEL)) << "\n";
    }
  }
  errs() << "The total initial area is " << total_initial_area << " and the total final area is " << total_final_area << " and the old one is " << total_old_final_area << "\n";

  
  for (auto F : mergedFunctions) { //these shouldn't appear in the final binary
        errs() << "The merging operand area size for " << F << " is " << std::to_string(areaAtMergeTime[F]) << "\n";
        errs() << "The merging operand hw latency for " << F << " is " << std::to_string(hwLatAtMergeTime[F]) << "\n";
        errs() << "The merging operand sw latency for " << F << " is " << std::to_string(swLatAtMergeTime[F]) << "\n";
        errs() << "The merging operand code size for " <<  F << " is " << std::to_string(funcCodeSizeAtMergeTime[F]) << "\n";
        //errs() << "The merging operand number of calls " << F << " is " << std::to_string(numCallsAtMergeTime[F]) << "\n";
  }
  for (auto itCalls = global_num_calls.begin(); itCalls != global_num_calls.end(); ++itCalls) {
    errs() << "The number of calls for " << itCalls->first << " is " << itCalls->second << "\n";
  }
  //errs() << "you go past this1\n";
  //for (auto &F : mergedFunctions) {
  //  double timeSize = 0.0;
  //  unsigned func_size = estimateFunctionSize(*F, &TTI, timeSize);
  //  errs() << "The size for function " << GetValueName(F) << " is " << FuncSizes[F] << "\n";
  //  errs() << "the one you read again for " << GetValueName(F) << " is " << func_size << "\n";
  //  errs() << "The timesize for function " << GetValueName(F) << " is " << timeSize << "\n";

  //  errs() << "The accelSeeker area size for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, AREA_ACCEL)) << "\n";
  //  errs() << "The accelSeeker hw latency for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, HW_LATENCY_ACCEL)) << "\n";
  //  errs() << "The accelSeeker sw latency for " <<  GetValueName(F) << " is " << std::to_string(estimateFunctionLatencyOrArea(*F, SW_LATENCY_ACCEL)) << "\n";
  //}
  //errs() << "you go past this2\n";
  //for (auto &F : M) {
  //  double timeSize = 0.0;
  //  unsigned func_size = estimateFunctionSize(F, &TTI, timeSize);
  //  errs() << "The size for function " << GetValueName(&F) << " is " << FuncSizes[&F] << "\n";
  //  errs() << "the one you read again for " << GetValueName(&F) << " is " << func_size << "\n";
  //  errs() << "The timesize for function " << GetValueName(&F) << " is " << timeSize << "\n";

  //  errs() << "The accelSeeker area size for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, AREA_ACCEL)) << "\n";
  //  errs() << "The accelSeeker hw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, HW_LATENCY_ACCEL)) << "\n";
  //  errs() << "The accelSeeker sw latency for " <<  GetValueName(&F) << " is " << std::to_string(estimateFunctionLatencyOrArea(F, SW_LATENCY_ACCEL)) << "\n";
  //}



  for (int i = 0; i < codeAndTimeReduction.size(); ++i)
    errs() << "Code reduction " << codeAndTimeReduction[i].first << " " << codeAndTimeReduction[i].second << "\n";

  return true;
}

//static void registerMyPass(const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
                               //PM.add(new FMSALegacyPass());
//                               }
//static RegisterStandardPasses
//    RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
//                       registerMyPass);


static cl::opt<std::string>
FileWithFunctionsToMerge("fmsa-func-to-merge-file", cl::value_desc("funcname"),
          cl::desc("A file containing the the pairs of functions to be merged per line"),
          cl::Hidden);

struct MergeSpecificFunctions : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  std::vector<std::pair<std::string, std::string> > listOfFuncNamesToMerge;
  std::vector<std::pair<std::string, std::string> > alignedBBNames;
  MergeSpecificFunctions() : ModulePass(ID) {}


  void LoadFile(const char *Filename) {
    // Load the BlockFile...
    std::ifstream In(Filename);
    if (!In.good()) {
      errs() << "WARNING: Function merging couldn't load file '" << Filename
             << "'!\n";
      exit(-1);
      return;
    }
    while (In) {
      std::string FunctionToMerge1, FunctionToMerge2;
      In >> FunctionToMerge1;
      In >> FunctionToMerge2;
      if (FunctionToMerge1 != "" and FunctionToMerge2 != "") {
        FunctionToMerge1 = "@"+FunctionToMerge1;
        FunctionToMerge2 = "@"+FunctionToMerge2;
        errs() << "Have to merge " + FunctionToMerge1 + " and " + FunctionToMerge2 << "\n";
        listOfFuncNamesToMerge.push_back(std::pair<std::string, std::string>(FunctionToMerge1, FunctionToMerge2));
      }
        //bbProfiles.push_back(BasicBlockDynProfile(FunctionName, BlockName, numInstructions));
    }
  }


  void LoadBBNames(const char *Filename) {
    // Load the BlockFile...
    std::ifstream In(Filename);
    alignedBBNames.clear();
    if (!In.good()) {
      errs() << "The bb alignment will be computed in this pass as opposed to externally\n";
      errs() << Filename << "'!\n";
      return;
    }
    while (In) {
      std::string bbName1, bbName2, bbSimil;
      In >> bbName1;
      In >> bbName2;
      In >> bbSimil;
      if (bbName1 != "" and bbName2 != "") {
        errs() << "Have to merge " + bbName1 + " and " + bbName2 << "\n";
        alignedBBNames.push_back(std::pair<std::string, std::string>(bbName1, bbName2));
      }
    }
  }


  //some code you might need but if you don't understand it, delete it.
  /*for (Function::iterator BB1 = F.begin(), BE = F.end(); BB1 != BE; ++BB1) {
    Function::iterator BB2 = BB1;
    BB2++; //+1 can't be used apparently on llvm iterators
    for (Function::iterator BB2 = F->begin(); BB2 != BE1; ++BB2) {
      
    }
  }*/

  void CreateFVecFromSingleBB(SmallVectorImpl<Value *> &FVec, BasicBlock * bb) {
      std::list<BasicBlock *> OrderedBBs;
      OrderedBBs.push_back(bb);
      CreateFVec(OrderedBBs, FVec, 10);
  }

  void MoveNonAlignedInsts(std::set<BasicBlock* > &nonalig_bbsf, 
                           std::list<std::pair<Value *, Value *> > &AlignedInstsGlobal,
                           bool isFirstFunc) {
      for (auto nonalig_bbf_it = nonalig_bbsf.begin(); nonalig_bbf_it != nonalig_bbsf.end(); nonalig_bbf_it++) {
          BasicBlock *nonalig_bbf = *nonalig_bbf_it;

          SmallVector<Value *, 10> FVec;
          CreateFVecFromSingleBB(FVec, nonalig_bbf);
          for (int i = 0; i < FVec.size(); i++)
              if (isFirstFunc)
                  AlignedInstsGlobal.push_back(std::pair<Value*, Value* >(FVec[i], nullptr));
              else
                  AlignedInstsGlobal.push_back(std::pair<Value*, Value* >(nullptr, FVec[i]));
      }
  }


  void PrintInstrsInOrdBBs(std::list<BasicBlock *> &OrderedBBs) {
    for (auto it = OrderedBBs.begin(); it != OrderedBBs.end(); ++it) {
      BasicBlock *bb = *it;
      int numInstr = 0;
      for (auto &II : *bb) {
        numInstr++;
      }
      errs() << "BB " << bb->getParent()->getName() << "::" << bb->getName() << " contains " << numInstr << " instructions\n";
    }
  }

  //Pass used for merging manually two functions
  bool runOnModule(Module &M) override {
    LoadFile(FileWithFunctionsToMerge.c_str());
    
    std::vector<Function *> mergingInput1;
    std::vector<Function *> mergingInput2;
    int i = 0;
    for (auto funcNames : listOfFuncNamesToMerge) {
        std::list<std::pair<Value *, Value *>> AlignedInstsGlobal;
        double RoughReduction = 0;
        int numRoughReductions = 0;

        std::string f1_name_noat = funcNames.first.substr(1);
        std::string f2_name_noat = funcNames.second.substr(1);
        std::map<std::string, BasicBlock* > bbNameToBB;
        
        if (not FuncConcatMode)
          LoadBBNames(("../matchings/"+f1_name_noat+"_"+f2_name_noat+"_matching.txt").c_str());

        std::set<BasicBlock *> nonalig_bbsf1;
        std::set<BasicBlock *> nonalig_bbsf2;

        for (Function &F : M) {
            if (GetValueName(&F) == funcNames.first) {

                demoteRegToMem(F);
                mergingInput1.push_back(&F);
                for (auto &bb1 : F) {
                  nonalig_bbsf1.insert(&bb1);
                  errs() << "The nonalig_bbsf1 adds bb " << GetValueName(&F) << "::";
                  errs() << bb1.getName() << "\n";
                }
            }
            else if (GetValueName(&F) == funcNames.second) {
                demoteRegToMem(F);
                mergingInput2.push_back(&F);
                for (auto &bb2 : F) {
                  nonalig_bbsf2.insert(&bb2);
                  errs() << "The nonalig_bbsf1 adds bb " << GetValueName(&F) << "::";
                  errs() << bb2.getName() << "\n";
                }
            } else
                continue;
     
            std::set<BasicBlock* > setOfBBsToAlign;
            for (auto &BB : F) { 
                //Before changing the following line consider that bb names without
                //prefix might overlap
                std::string bbName = F.getName().str() + "::" + BB.getName().str();
                bbNameToBB[bbName] = &BB;
            }
        }
        
        //Begin DBG
        std::list<BasicBlock *> OrderedBBs1;
        std::list<BasicBlock *> OrderedBBs2;
        //End DBG

        for (int i = 0; i < alignedBBNames.size(); ++i) {
          std::string bb1Name = alignedBBNames[i].first;
          std::string bb2Name = alignedBBNames[i].second;
          BasicBlock *bb1_tmp = bbNameToBB[bb1Name];
          BasicBlock *bb2_tmp = bbNameToBB[bb2Name];
          
          BasicBlock *bb1, *bb2;
          if (bb1_tmp->getParent()->getName() == funcNames.first.substr(1)) {
            bb1 = bb1_tmp;
            bb2 = bb2_tmp;
            assert(bb2_tmp->getParent()->getName() == funcNames.second.substr(1));
            if (bb2_tmp->getParent()->getName() != funcNames.second.substr(1)) {
              errs() << "a)" << bb1_tmp->getParent()->getName() << " " << funcNames.first << "\n";
              errs() << "b)" << bb2_tmp->getParent()->getName() << " " << funcNames.second << "\n";
              errs() << "1\n";
              exit(-1);
            }
          } else {
            bb1 = bb2_tmp;
            bb2 = bb1_tmp;
            if (bb2_tmp->getParent()->getName() != funcNames.first.substr(1)) {
              errs() << "a)" << bb1_tmp->getParent()->getName() << " " << funcNames.first << "\n";
              errs() << "b)" << bb2_tmp->getParent()->getName() << " " << funcNames.second << "\n";
              errs() << "2\n";
              exit(-1);
            }
            assert(bb2_tmp->getParent()->getName() == funcNames.first.substr(1));          
          }

          //Begin DBG
          OrderedBBs1.push_back(bb1);
          OrderedBBs2.push_back(bb2);
          //End DBG


          errs() << "The nonalig_bbsf1 removes " << bb1->getParent()->getName() << "::";
          errs() << bb1->getName() << "\n";


          errs() << "The nonalig_bbsf2 removes " << bb1->getParent()->getName() << "::";
          errs() << bb2->getName() << "\n";

          nonalig_bbsf1.erase(bb1);
          nonalig_bbsf2.erase(bb2);
          
          /* DBGC
          SmallVector<Value *, 10> FVec1;
          CreateFVecFromSingleBB(FVec1, bb1);

          SmallVector<Value *, 10> FVec2;
          CreateFVecFromSingleBB(FVec2, bb2);*/
          
          //std::list<std::pair<Value *, Value *>> AlignedInstsLocal; //DBGC
          //RoughReduction += AlignLinearizedCFGs(FVec1, FVec2, AlignedInstsLocal); DBGC
          numRoughReductions += 1;
          //AlignedInstsGlobal.splice(AlignedInstsGlobal.end(), AlignedInstsLocal); DBGC
        }
        
        assert(OrderedBBs1.size() > 0);
        assert(OrderedBBs2.size() > 0);
        //Begin DBG
        for (auto it = nonalig_bbsf1.begin(); it != nonalig_bbsf1.begin(); ++it)
          OrderedBBs1.push_back(*it);
        for (auto it = nonalig_bbsf2.begin(); it != nonalig_bbsf2.begin(); ++it)
          OrderedBBs2.push_back(*it);

        SmallVector<Value *, 100> F1Vec;
        SmallVector<Value *, 100> F2Vec;

        //unsigned numInstsAndBBs = F1Vec.size();
        //unsigned numBBs = OrderedBBs1.size();
        //CreateFVec(OrderedBBs1, F1Vec, 100);
        
        //numInstsAndBBs = F2Vec.size();
        //numBBs = OrderedBBs2.size();
        //CreateFVec(OrderedBBs2, F2Vec, 100);
        assert(F1Vec.size() > 0);
        assert(F2Vec.size() > 0);


        errs() << "F1VecSize:" <<  F1Vec.size() << "; F2VecSize:" << F2Vec.size();
        errs() << "mergingInput1Size:" <<  mergingInput1.size() << "; mergi2Size:" << mergingInput2.size();


        errs() << "F1 bbs info\n";
        PrintInstrsInOrdBBs(OrderedBBs1);
        errs() << "F2 bbs info\n";
        PrintInstrsInOrdBBs(OrderedBBs2);
        
        if (numRoughReductions != 0)
          RoughReduction += AlignLinearizedCFGs(F1Vec, F2Vec, AlignedInstsGlobal);
        //numRoughReductions = 1;
        //End DBG
        //MoveNonAlignedInsts(nonalig_bbsf1, AlignedInstsGlobal, true); DBGC
        //MoveNonAlignedInsts(nonalig_bbsf2, AlignedInstsGlobal, false); DBGC

        //At this point the two functions you were looking for, were detected in the src code (hopefully)

        if (i == mergingInput1.size() - 1 and i == mergingInput2.size() - 1)
          ++i;
        else {
            errs() << "The functions provided are not found in the src code\n";
            errs() << "A typical cause is that you applied an application suffix or prefix to func names\n";
            exit(-1);
        }

        //This could be failing
        Function *F1 = mergingInput1[i - 1];
        Function *F2 = mergingInput2[i - 1];

        if (numRoughReductions == 0) {
            OrderedBBs1.clear();
            OrderedBBs2.clear();
            Linearization(mergingInput1[i-1], F1Vec, OrderedBBs1, LinearizationKind::LK_Canonical);
            F2Vec.reserve(F1Vec.size());
            Linearization(mergingInput2[i-1], F2Vec, OrderedBBs2, LinearizationKind::LK_Canonical);
            errs() << "F1VecSize:" <<  F1Vec.size() << "; F2VecSize:" << F2Vec.size() << "\n";
            errs() << "mergingInput1Size:" <<  mergingInput1.size() << "; mergi2Size:" << mergingInput2.size() << "\n";
            errs() << "F1 bbs info\n";
            PrintInstrsInOrdBBs(OrderedBBs1);
            errs() << "F2 bbs info\n";
            PrintInstrsInOrdBBs(OrderedBBs2);

            //demoteRegToMem(*(mergingInput1[i-1])); //you can suppose this has already been done
            //demoteRegToMem(*(mergingInput2[i-1]));
            errs() << "Merging functions 1:" << GetValueName(mergingInput1[i-1]) << " and 2:" << GetValueName(mergingInput2[i-1]) << "\n";
            MergedFunction Result = mergeBySequenceAlignment(mergingInput1[i-1], mergingInput2[i-1], F1Vec, F2Vec, OrderedBBs1, OrderedBBs2);
            StringSet<> AlwaysPreserved;
            Result.MergedFunc->setName("merged_" + listOfFuncNamesToMerge[i-1].first.substr(1) + "_" + listOfFuncNamesToMerge[i-1].second.substr(1));
            errs() << "Merged: " << GetValueName(mergingInput1[i-1]) 
                   << ", " << GetValueName(mergingInput2[i-1])
                   << " = " << GetValueName(Result.MergedFunc) << "\n";
            UpdateCallGraph(M, Result, AlwaysPreserved);
            Result.MergedFunc->setName("merged_" + listOfFuncNamesToMerge[i-1].first.substr(1) + "_" + listOfFuncNamesToMerge[i-1].second.substr(1));

            merged_func_id++;

            RoughReduction = -1;
        
        } else {
          RoughReduction = RoughReduction/numRoughReductions;

          //demoteRegToMem(*F1); //you can suppose this has already been done
          //demoteRegToMem(*F2);
          //START DBG
          /*std::list<BasicBlock *> OrderedBBs1;
          std::list<BasicBlock *> OrderedBBs2;

          SmallVector<Value *, 32> F1Vec;
          Linearization(mergingInput1[i-1], F1Vec, OrderedBBs1, 
                        LinearizationKind::LK_Canonical);
          SmallVector<Value *, 32> F2Vec;
          F2Vec.reserve(F1Vec.size());
          Linearization(mergingInput2[i-1], F2Vec, OrderedBBs2, 
                        LinearizationKind::LK_Canonical);*/
          //END DBG
          errs() << "F1VecSize:" <<  F1Vec.size() << "; F2VecSize:" << F2Vec.size();
          errs() << "mergingInput1Size:" <<  mergingInput1.size() << "; mergi2Size:" << mergingInput2.size();


          errs() << "F1 bbs info\n";
          PrintInstrsInOrdBBs(OrderedBBs1);
          errs() << "F2 bbs info\n";
          PrintInstrsInOrdBBs(OrderedBBs2);
          
          MergedFunction Result = mergeBySequenceAlignment(mergingInput1[i-1], mergingInput2[i-1], F1Vec, F2Vec, OrderedBBs1, OrderedBBs2);
          StringSet<> AlwaysPreserved;
          errs() << "Merged: " << GetValueName(mergingInput1[i-1]) << ", " 
                 << GetValueName(mergingInput2[i-1])
                 << " = " << GetValueName(Result.MergedFunc) << "\n";
          UpdateCallGraph(M, Result, AlwaysPreserved);
          Result.MergedFunc->setName("merged_" + listOfFuncNamesToMerge[i-1].first.substr(1) + "_" + listOfFuncNamesToMerge[i-1].second.substr(1));
          /*MergedFunction mf = mergeBySequenceAlignmentPrecomputedAlignedInsts(F1, F2,
                                   AlignedInstsGlobal,
                                   RoughReduction);*/

        }

    }
    return true; //one possible cause of errors is that sometimes maybe you don't merge anything?
  } //before returning true check whether there has been any merging
    
    
    //for (auto funcNames : listOfFuncNamesToMerge) {
    //    Function *F1 = NULL;
    //    Function *F2 = NULL;
    //    for (Function &F : M) {
    //        if (not F.isDeclaration()) {
    //            if (GetValueName(&F) == funcNames.first)
    //                F1 = &F;
    //            else if (GetValueName(&F) == funcNames.second)
    //                F2 = &F;
    //        }
    //        if (F1 != (Function *)NULL and F2 != (Function *)NULL) {
    //            mergingInput1.push_back(&F);
    //            mergingInput2.push_back(&F);
    //        }
    //    }
    //}


};

char MergeSpecificFunctions::ID = 0;
static RegisterPass<MergeSpecificFunctions> MLF("fmsa-merge-list-of-funcs", "Merge only the list of function name pairs present in the file FileWithFunctionsToMerge.");
