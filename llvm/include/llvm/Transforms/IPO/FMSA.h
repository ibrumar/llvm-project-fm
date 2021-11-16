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

#include "llvm/InitializePasses.h"

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

struct BasicBlockDynProfile {
    std::string funcName;
    std::string bbName;
    double dynPercInstr;
    BasicBlockDynProfile(std::string funcName, std::string bbName, int numInstr): funcName(funcName), bbName(bbName), dynPercInstr(dynPercInstr){}
};

class FMSALegacyPass : public ModulePass {
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
  FMSALegacyPass() : ModulePass(ID) {
     initializeFMSALegacyPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace
#endif

