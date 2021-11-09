//===- Transforms/FunctionMerging.h - function merging passes  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file

//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_FUNCTION_MERGING_H
#define LLVM_TRANSFORMS_FUNCTION_MERGING_H

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include <llvm/Analysis/DependenceAnalysis.h>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/PassManager.h"

#include "llvm/Transforms/Instrumentation.h"

namespace llvm{


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


bool matchIntrinsicCalls(Intrinsic::ID ID, const CallInst *CI1,
                                const CallInst *CI2);

bool matchLandingPad(LandingPadInst *LP1, LandingPadInst *LP2);

bool match(Instruction *I1, Instruction *I2);

bool match(Value *V1, Value *V2);

bool match(std::vector<Value *> &F1, std::vector<Value *> &F2,
                  unsigned i, unsigned j);

bool match(SmallVectorImpl<Value *> &F1, SmallVectorImpl<Value *> &F2,
                  unsigned i, unsigned j);

struct BasicBlockDynProfile {
    std::string funcName;
    std::string bbName;
    double dynPercInstr;
    BasicBlockDynProfile(std::string funcName, std::string bbName, int numInstr): funcName(funcName), bbName(bbName), dynPercInstr(dynPercInstr){}
};

class FunctionMerging : public ModulePass {
  StringSet<> AlwaysPreserved;

  //std::vector< BasicBlockDynProfile > bbProfiles;

  void LoadRealFuncArea(const char *Filename);
  void LoadAreaModel(const char *Filename);
  void LoadFile(const char *Filename);
  void LoadBlacklistFile(const char *Filename);
  bool shouldPreserveGV(const GlobalValue &GV);

public:
   static std::map< std::string, double > realAreaPerFunc;
  static std::map< std::string, double > opLuts;
  static std::set< std::string> BlackList;
  static std::map< std::string, double > bbProfiles;
  static std::map< BasicBlock *, double > mergedBBProfiles;
  static std::map< BasicBlock *, double > mergedBBTotal;
  static std::map< Function *, double > funcTimeSize;
  static char ID;
  FunctionMerging() : ModulePass(ID) {
     initializeFunctionMergingPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace
#endif
