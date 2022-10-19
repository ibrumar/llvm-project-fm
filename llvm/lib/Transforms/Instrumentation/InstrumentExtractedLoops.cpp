//===- InstrumentMemoryAccesses.cpp - Insert load/store checks ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments loads, stores, and other memory intrinsics with
// load/store checks by inserting the relevant __loadcheck and/or
// __storecheck calls before the them.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instrument-memory-accesses"
#include "llvm/Support/CommandLine.h"

//#include "CommonMemorySafetyPasses.h"
#include<iostream>
#include<string>
#include<set>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

//TODO: Uncomment after Saketh's fix
//#include "LoopShapePass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
//#include "llvm/Support/raw_os_ostream.h"
//#include "llvm/Support/Debug.h"

//#include "GroundTruth.h"
//#include "LoopShapePass.h"

using namespace llvm;



namespace {
  class InstrumentExtractedLoopsLegacyPass : public FunctionPass,
                                   public InstVisitor<InstrumentExtractedLoopsLegacyPass> {
    
    std::map<std::pair<const Type *, bool>, Function * > typeToFuncPtr;
    Function *CreateTbXmlFunc;
    std::set<std::string> leFuncNames;
    Type *CharPtrTy;
    Type *SizeTy;
  public:
    static char ID;
    InstrumentExtractedLoopsLegacyPass(): FunctionPass(ID) { }
    //virtual bool doInitialization(Module &M);
    //virtual bool runOnFunction(Function &F);

    void getAnalysisUsage(AnalysisUsage &AU) const {
//      AU.addRequired<DataLayoutPass>();
      //AU.addRequired<MyModulePass>(); //we need it so that all the functions have their names in global variables
      //AU.addRequired<Sita::LoopShapePass>();
      //AU.setPreservesCFG();
    }



  bool doInitialization(Module &M) {
        /*START LEI*/
        Type *VoidTy = Type::getVoidTy(M.getContext());
        Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
        CharPtrTy = Type::getInt8PtrTy(M.getContext());
        Type *ShortPtrTy = Type::getInt16PtrTy(M.getContext());
        Type *IntPtrTy = Type::getInt32PtrTy(M.getContext());
        Type *FloatPtrTy = Type::getFloatPtrTy(M.getContext());
        Type *DoublePtrTy = Type::getDoublePtrTy(M.getContext());
        
        Type *FloatTy = Type::getFloatTy(M.getContext());
        Type *DoubleTy = Type::getDoubleTy(M.getContext());
        Type *ParamIdxTy = IntegerType::getInt64Ty(M.getContext());
        SizeTy = IntegerType::getInt64Ty(M.getContext());
        Type *IntegerTy = IntegerType::getInt32Ty(M.getContext());
        Type *ShortTy = IntegerType::getInt16Ty(M.getContext());
        Type *CharTy = IntegerType::getInt8Ty(M.getContext());
        
        M.getOrInsertFunction("__captureOriginalIntPtrVal", VoidTy, IntPtrTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalShortPtrVal", VoidTy, ShortPtrTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalCharPtrVal", VoidTy, CharPtrTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalFloatPtrVal", VoidTy, FloatPtrTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalDoublePtrVal", VoidTy, DoublePtrTy, ParamIdxTy);

        typeToFuncPtr[std::pair<const Type *, bool>(IntPtrTy, false)] = M.getFunction("__captureOriginalIntPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(ShortPtrTy, false)]  = M.getFunction("__captureOriginalShortPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(CharPtrTy, false)] = M.getFunction("__captureOriginalCharPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(FloatPtrTy, false)]  = M.getFunction("__captureOriginalFloatPtrVal");
        typeToFuncPtr[std::pair<const Type *, bool>(DoublePtrTy, false)] = M.getFunction("__captureOriginalDoublePtrVal");
        
        /*
        //For allocas and scalars that are output from the extracted region
        
        M.getOrInsertFunction("__captureIntPtrValWithSize", VoidTy, IntPtrTy, ParamIdxTy, SizeTy);
        M.getOrInsertFunction("__captureShortPtrValWithSize", VoidTy, ShortPtrTy, ParamIdxTy, SizeTy);
        M.getOrInsertFunction("__captureCharPtrValWithSize", VoidTy, CharPtrTy, ParamIdxTy, SizeTy);
        M.getOrInsertFunction("__captureFloatPtrValWithSize", VoidTy, FloatPtrTy, ParamIdxTy, SizeTy);
        M.getOrInsertFunction("__captureDoublePtrValWithSize", VoidTy, DoublePtrTy, ParamIdxTy, SizeTy);


        typeToFuncPtr[std::pair<const Type *, bool>(FloatTy, true)] = M.getFunction("__captureFloatVal");
        typeToFuncPtr[std::pair<const Type *, bool>(DoubleTy, true)] = M.getFunction("__captureDoubleVal"); 
        typeToFuncPtr[std::pair<const Type *, bool>(IntegerTy, true)] = M.getFunction("__captureIntegerVal");     
        typeToFuncPtr[std::pair<const Type *, bool>(ShortTy, true)] = M.getFunction("__captureShortVal");
        typeToFuncPtr[std::pair<const Type *, bool>(CharTy, true)] = M.getFunction("__captureCharVal");
        */

        M.getOrInsertFunction("__captureOriginalFloatVal", VoidTy, FloatTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalDoubleVal", VoidTy, DoubleTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalIntegerVal", VoidTy, IntegerTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalShortVal", VoidTy, ShortTy, ParamIdxTy);
        M.getOrInsertFunction("__captureOriginalCharVal", VoidTy, CharTy, ParamIdxTy);
        typeToFuncPtr[std::pair<const Type *, bool>(FloatTy, false)] = M.getFunction("__captureOriginalFloatVal");
        typeToFuncPtr[std::pair<const Type *, bool>(DoubleTy, false)] = M.getFunction("__captureOriginalDoubleVal"); 
        typeToFuncPtr[std::pair<const Type *, bool>(IntegerTy, false)] = M.getFunction("__captureOriginalIntegerVal");     
        typeToFuncPtr[std::pair<const Type *, bool>(ShortTy, false)] = M.getFunction("__captureOriginalShortVal");
        typeToFuncPtr[std::pair<const Type *, bool>(CharTy, false)] = M.getFunction("__captureOriginalCharVal");
        for (auto it = typeToFuncPtr.begin(); it != typeToFuncPtr.end(); ++it) {
          it->second->setLinkage(Function::ExternalLinkage);
        }

        //passes called function name and the number of arguments
        M.getOrInsertFunction("__createTbXml", VoidTy, CharPtrTy, SizeTy);
        CreateTbXmlFunc = M.getFunction("__createTbXml");
        CreateTbXmlFunc->setLinkage(Function::ExternalLinkage);
        M.getOrInsertFunction("__setPtrSize", VoidTy, VoidPtrTy, SizeTy);
        Function *SetPtrFunc = M.getFunction("__setPtrSize");
        SetPtrFunc->setLinkage(Function::ExternalLinkage);
        //END LEI
  }

  
  std::string GetValueName(const Value *V) {
    if (V) {
      std::string name;
      raw_string_ostream namestream(name);
      V->printAsOperand(namestream, false);
      return namestream.str();
    }
    return "[null]";
  }

  bool runOnFunction(Function &F) { 
                                    
    if (F.isDeclaration())
      return false;
    leFuncNames.insert(GetValueName(&F)+"_"); //naming contention in release
    errs() << "Running on function " << F.getName() << "\n";

    for (auto &BB : F) {
      for (auto &II : BB) {
        CallInst *CI = dyn_cast<CallInst>(&II); //II.dyn_cast?
        if (CI == nullptr)
          continue;
        Function *CF = CI->getCalledFunction();
        bool foundForCond = GetValueName(CF).find("for.cond") != std::string::npos; 
        bool foundForBody = GetValueName(CF).find("for.body") != std::string::npos; 
        bool foundWhileCond = GetValueName(CF).find("while.cond") != std::string::npos; 
        bool foundWhileBody = GetValueName(CF).find("while.body") != std::string::npos; 
        bool foundUnderscore = leFuncNames.find(GetValueName(CF)) != leFuncNames.end(); 
        if (foundUnderscore or foundForCond or foundForBody or foundWhileCond or foundWhileBody) {
          //we insert instrumentation before the call to the extracted loop
          //for (auto& A : CF->getArgumentList()) {
          //  Builder->CreateCall(captureOriginalValuesFunction, A);
          //}

          IRBuilder<> Builder(F.getContext()); //adding the call instruction as an insertion point
          Builder.SetInsertPoint(CI);
          //for (auto& A : CF->getArgumentList()) {
          //for (auto *A : ExtractedFuncAndCI.second->args()) {
          //  //Not sure if we need to distinguish arguments for being pointers or scalars
          //}
          errs() << "Reaching this point\n";

          //Type *CharPtrTy = Type::getInt8PtrTy(TheContext);
          Value *functionNameParam = Builder.CreatePointerCast(Builder.CreateGlobalStringPtr(CI->getCalledFunction()->getName()), CharPtrTy);
          //Value *functionNameParam = Builder.CreateGlobalStringPtr(CI->getCalledFunction()->getName());

          std::vector<Value *> createXmlArgsV;
          createXmlArgsV.push_back(functionNameParam);
          createXmlArgsV.push_back(Builder.getInt64(CI->getNumArgOperands()));
          ArrayRef<Value *> createXmlArgs(createXmlArgsV);

          Builder.CreateCall(CreateTbXmlFunc, createXmlArgs);
          
          for (int i = 0; i<CI->getNumArgOperands(); i++) {
              Value *Arg = CI->getArgOperand(i);
              errs() << "Creating a call to instr func " << typeToFuncPtr[std::pair<Type*, bool>(Arg->getType(), false)]->getName() << "\n";
              
              
              std::vector<Value *> instrArgsV;
              instrArgsV.push_back(Arg);
              instrArgsV.push_back(Builder.getInt64(i));
              ArrayRef<Value *> instrArgs(instrArgsV);
              
              Builder.CreateCall(typeToFuncPtr[std::pair<Type*, bool>(Arg->getType(), false)], instrArgs);
              //Builder.CreateCall(typeToFuncPtr[std::pair<Type*, bool>(Arg->getType(), false)], Arg, Builder.getInt64(i));
              //IRB.getInt32(0)
          }
        }
      }
    }
    
    return true;
  }

  };
} // end anon namespace

char InstrumentExtractedLoopsLegacyPass::ID = 0;


FunctionPass *llvm::createInstrumentExtractedLoopsPass() {
  return new InstrumentExtractedLoopsLegacyPass();
}

INITIALIZE_PASS(InstrumentExtractedLoopsLegacyPass, "instrument-extracted-loops", "Merge only the list of function name pairs present in the file FileWithFunctionsToMerge.", false, false)



////char MyModulePass::ID = 0;
//static RegisterPass<InstrumentExtractedLoops> Z("instrumentPar", "Getting loads and stores to be instrumented");
//
//// Register for Clang
//static void registerInstrumentExtractedLoopsPass(const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
//    // TODO: Restrict based on specified optimization levels.
//    PM.add(new InstrumentExtractedLoops());
//}
//
//static RegisterStandardPasses RegisterSitaInstrumentExtractedLoopsPass(PassManagerBuilder::EP_EnabledOnOptLevel0, registerInstrumentExtractedLoopsPass);

