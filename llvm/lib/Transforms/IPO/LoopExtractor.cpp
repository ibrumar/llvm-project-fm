//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/IR/Type.h"
#include <fstream>
#include <set>
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "loop-extract"

STATISTIC(NumExtracted, "Number of loops extracted");

namespace {
  struct LoopExtractor : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    unsigned NumLoops;
    std::map<std::pair<const Type *, bool>, Function * > typeToFuncPtr;
    Function *CreateTbXmlFunc;



    bool doInitialization(Module &M) {
        /*START LEI*/
        Type *VoidTy = Type::getVoidTy(M.getContext());
        Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
        Type *CharPtrTy = Type::getInt8PtrTy(M.getContext());
        Type *ShortPtrTy = Type::getInt16PtrTy(M.getContext());
        Type *IntPtrTy = Type::getInt32PtrTy(M.getContext());
        Type *FloatPtrTy = Type::getFloatPtrTy(M.getContext());
        Type *DoublePtrTy = Type::getDoublePtrTy(M.getContext());
        
        Type *FloatTy = Type::getFloatTy(M.getContext());
        Type *DoubleTy = Type::getDoubleTy(M.getContext());
        Type *ParamIdxTy = IntegerType::getInt64Ty(M.getContext());
        Type *SizeTy = IntegerType::getInt64Ty(M.getContext());
        Type *IntegerTy = IntegerType::getInt32Ty(M.getContext());
        Type *ShortTy = IntegerType::getInt16Ty(M.getContext());
        Type *CharTy = IntegerType::getInt8Ty(M.getContext());
        
        M.getOrInsertFunction("__captureOriginalIntPtrVal", VoidTy, IntPtrTy, ParamIdxTy, NULL);
        M.getOrInsertFunction("__captureOriginalShortPtrVal", VoidTy, ShortPtrTy, ParamIdxTy, NULL);
        M.getOrInsertFunction("__captureOriginalCharPtrVal", VoidTy, CharPtrTy, ParamIdxTy, NULL);
        M.getOrInsertFunction("__captureOriginalFloatPtrVal", VoidTy, FloatPtrTy, ParamIdxTy, NULL);
        M.getOrInsertFunction("__captureOriginalDoublePtrVal", VoidTy, DoublePtrTy, ParamIdxTy, NULL);

        typeToFuncPtr[std::pair<const Type *, bool>(IntPtrTy, false)] = M.getFunction("__captureOriginalIntPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(ShortPtrTy, false)]  = M.getFunction("__captureOriginalShortPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(CharPtrTy, false)] = M.getFunction("__captureOriginalCharPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(FloatPtrTy, false)]  = M.getFunction("__captureOriginalFloatPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(DoublePtrTy, false)] = M.getFunction("__captureOriginalDoublePtrVal");
        
        //For allocas and scalars that are output from the extracted region
        /*
        M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, IntPtrTy, ParamIdxTy, SizeTy, NULL);
        M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, ShortPtrTy, ParamIdxTy, SizeTy, NULL);
        M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, CharPtrTy, ParamIdxTy, SizeTy, NULL);
        M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, FloatPtrTy, ParamIdxTy, SizeTy, NULL);
        M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, DoublePtrTy, ParamIdxTy, SizeTy, NULL);*/


         M.getOrInsertFunction("__captureOriginalFloatVal", VoidTy, FloatTy, ParamIdxTy, NULL);
         M.getOrInsertFunction("__captureOriginalDoubleVal", VoidTy, DoubleTy, ParamIdxTy, NULL);
         M.getOrInsertFunction("__captureOriginalIntegerVal", VoidTy, IntegerTy, ParamIdxTy, NULL);
         M.getOrInsertFunction("__captureOriginalShortVal", VoidTy, ShortTy, ParamIdxTy, NULL);
         M.getOrInsertFunction("__captureOriginalCharVal", VoidTy, CharTy, ParamIdxTy, NULL);
         
         typeToFuncPtr[std::pair<const Type *, bool>(FloatTy, false)] = M.getFunction("__captureOriginalFloatVal");
         typeToFuncPtr[std::pair<const Type *, bool>(DoubleTy, false)] = M.getFunction("__captureOriginalDoubleVal"); 
         typeToFuncPtr[std::pair<const Type *, bool>(IntegerTy, false)] = M.getFunction("__captureOriginalIntegerVal");     
         typeToFuncPtr[std::pair<const Type *, bool>(ShortTy, false)] = M.getFunction("__captureOriginalShortVal");
         typeToFuncPtr[std::pair<const Type *, bool>(CharTy, false)] = M.getFunction("__captureOriginalCharVal");
         
         M.getOrInsertFunction("__createTbXml", VoidTy, NULL);
         CreateTbXmlFunc = M.getFunction("__createTbXml");
        //END LEI
    
    }

    explicit LoopExtractor(unsigned numLoops = ~0)
      : LoopPass(ID), NumLoops(numLoops) {
        initializeLoopExtractorPass(*PassRegistry::getPassRegistry());

      }

    bool runOnLoop(Loop *L, LPPassManager &) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
    }
  };
}

char LoopExtractor::ID = 0;
INITIALIZE_PASS_BEGIN(LoopExtractor, "loop-extract",
                      "Extract loops into new functions", false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(LoopExtractor, "loop-extract",
                    "Extract loops into new functions", false, false)

namespace {
  /// SingleLoopExtractor - For bugpoint.
  struct SingleLoopExtractor : public LoopExtractor {
    static char ID; // Pass identification, replacement for typeid
    SingleLoopExtractor() : LoopExtractor(1) {}
  };
} // End anonymous namespace

char SingleLoopExtractor::ID = 0;
INITIALIZE_PASS(SingleLoopExtractor, "loop-extract-single",
                "Extract at most one loop into a new function", false, false)

// createLoopExtractorPass - This pass extracts all natural loops from the
// program into a function if it can.
//
Pass *llvm::createLoopExtractorPass() { return new LoopExtractor(); }

bool LoopExtractor::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipLoop(L)) {
    errs() << "1.Loop " << L->getName() << " is failing here\n";
    return false;
  }

  // Only visit top-level loops.
  if (L->getParentLoop()) {
    errs() << "2.Loop " << L->getName() << " is failing here\n";
    return false;
  }

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm()) {
    errs() << "3.Loop " << L->getName() << " is failing here\n";
    return false;
  }

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  bool Changed = false;

  // If there is more than one top-level loop in this function, extract all of
  // the loops. Otherwise there is exactly one top-level loop; in this case if
  // this function is more than a minimal wrapper around the loop, extract
  // the loop.
  bool ShouldExtractLoop = false;

  // Extract the loop if the entry block doesn't branch to the loop header.
  TerminatorInst *EntryTI =
    L->getHeader()->getParent()->getEntryBlock().getTerminator();
  if (!isa<BranchInst>(EntryTI) ||
      !cast<BranchInst>(EntryTI)->isUnconditional() ||
      EntryTI->getSuccessor(0) != L->getHeader()) {
    ShouldExtractLoop = true;
  } else {
    // Check to see if any exits from the loop are more than just return
    // blocks.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (!isa<ReturnInst>(ExitBlocks[i]->getTerminator())) {
        ShouldExtractLoop = true;
        break;
      }
  }

  if (ShouldExtractLoop) {
    // We must omit EH pads. EH pads must accompany the invoke
    // instruction. But this would result in a loop in the extracted
    // function. An infinite cycle occurs when it tries to extract that loop as
    // well.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (ExitBlocks[i]->isEHPad()) {
        ShouldExtractLoop = false;
        break;
      }
  }

  if (ShouldExtractLoop) {
    if (NumLoops == 0) return Changed;
    --NumLoops;
    CodeExtractor Extractor(DT, *L);
    //LEI modification
    std::pair<Function *, CallInst *> ExtractedFuncAndCI = Extractor.extractCodeRegion(); //NEWLINE LEI
    if (ExtractedFuncAndCI.first != nullptr) { //LEI modification end
      Changed = true;
      // After extraction, the loop is replaced by a function call, so
      // we shouldn't try to run any more loop passes on it.
      LPM.markLoopAsDeleted(*L);
      LI.erase(L);
      //LEI modification
      
      //ExtractedFuncAndCI.second->insertBefore;
      IRBuilder<> Builder(ExtractedFuncAndCI.second); //adding the call instruction as an insertion point

      Function *CF = ExtractedFuncAndCI.second->getCalledFunction();
      
      //for (auto& A : CF->getArgumentList()) {
      for (auto *A : ExtractedFuncAndCI.second->args()) {
        //Not sure if we need to distinguish arguments for being pointers or scalars
        Builder.CreateCall(typeToFuncPtr[std::pair<Type*, bool>(A->getType(), false)], A);
      }
      
      //LEI end
    }
    ++NumExtracted;
  }

  if (ShouldExtractLoop) 
    errs() << "returns Changed=" << Changed << "\n";
  return Changed;
}

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleLoopExtractorPass() {
  return new SingleLoopExtractor();
}
