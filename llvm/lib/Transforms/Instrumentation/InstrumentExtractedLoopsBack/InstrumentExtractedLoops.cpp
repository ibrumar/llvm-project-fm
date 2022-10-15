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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
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
//    const DataLayout *TD;
    IRBuilder<> *Builder;

    PointerType *VoidPtrTy;
    IntegerType *SizeTy;
    IntegerType *IntegerTy;

    PointerType *FloatTy = Type::getFloatTy(M.getContext());
    PointerType *DoubleTy = Type::getDoubleTy(M.getContext());
    IntegerType *ShortTy = IntegerType::getInt16Ty(M.getContext());
    IntegerType *CharTy = IntegerType::getInt8Ty(M.getContext());

    Function *captureOriginalValuesFunction;
    Function *LoadCheckFunction;
    Function *StoreCheckFunction;
    Function *BasicBlockInstCountFunction;
    Function *ReferenceFunction;
    
    Function *startCallFunction;                         
    Function *startFunction;                         
    Function *startIntrinsic;
    Function *endCallFunction;
    Function *confirmInternalFunction;
    Function *reportStatisticsFunction;
    Function *mainFunction;
    Function *exitFunction;
    Module *currentModule;
    Value *currentFunctionValue;
    int bbCounter;

    std::set<Function *> functionPtrs;
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


  void detectFunctionAnnotation(Module &M) {
    if(GlobalVariable* GA = M.getGlobalVariable("llvm.global.annotations")) {
      // the first operand holds the metadata
      for (Value *AOp : GA->operands()) {
        // all metadata are stored in an array of struct of metadata
        if (ConstantArray *CA = dyn_cast<ConstantArray>(AOp)) {
          // so iterate over the operands
          for (Value *CAOp : CA->operands()) {
            // get the struct, which holds a pointer to the annotated function
            // as first field, and the annotation as second field
            if (ConstantStruct *CS = dyn_cast<ConstantStruct>(CAOp)) {
              if (CS->getNumOperands() >= 2) {
                Function* AnnotatedFunction = cast<Function>(CS->getOperand(0)->getOperand(0));
								errs() << "The function's name is " << AnnotatedFunction->getName() << "\n";
                // the second field is a pointer to a global constant Array that holds the string
                if (GlobalVariable *GAnn =
                  dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0))) {
                  if (ConstantDataArray *A =
                    dyn_cast<ConstantDataArray>(GAnn->getOperand(0))) {
                    // we have the annotation! Check it's an epona annotation and process
                    StringRef AS = A->getAsString();
                    errs() << "The function is annotated with " << AS << "\n";
                    if (AS.startswith("referenceFunction")) {
                    //  FuncAnnotations[AnnotatedFunction].emplace_back(AS);
                        ReferenceFunction = AnnotatedFunction;
                    }
                  }
                }
              }
            }
          }
        }
      }
		}
  }

  bool doInitialization(Module &M) {
    Type *VoidTy = Type::getVoidTy(M.getContext());
    VoidPtrTy = Type::getInt8PtrTy(M.getContext());
    Type *CharPtrTy = Type::getInt8PtrTy(TheContext);
    Type *ShortPtrTy = Type::getInt16PtrTy(TheContext);
    Type *IntPtrTy = Type::getInt32PtrTy(TheContext);
    SizeTy = IntegerType::getInt64Ty(M.getContext());
    
    FloatTy = Type::getFloatTy(M.getContext());
    DoubleTy = Type::getDoubleTy(M.getContext());
    IntegerTy = IntegerType::getInt32Ty(M.getContext());
    ShortTy = IntegerType::getInt16Ty(M.getContext());
    CharTy = IntegerType::getInt8Ty(M.getContext());
    currentModule = &M;
	  detectFunctionAnnotation(M);

    //create a map from Pointer or value type to function and then call the appropriate
    //function
    errs() << "Source file " + M.getSourceFileName() << "\n";
    M.getOrInsertFunction("__captureOriginalPtrVal", VoidTy, IntPtrTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrVal", VoidTy, ShortPtrTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrVal", VoidTy, CharPtrTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrVal", VoidTy, FloatPtrTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrVal", VoidTy, DoublePtrTy, SizeTy, NULL);

    //For allocas and scalars that are output from the extracted region
    M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, IntPtrTy, SizeTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, ShortPtrTy, SizeTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, CharPtrTy, SizeTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, FloatPtrTy, SizeTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalPtrValWithSize", VoidTy, DoublePtrTy, SizeTy, SizeTy, NULL);

    M.getOrInsertFunction("__captureOriginalVal", VoidTy, FloatTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalVal", VoidTy, DoubleTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalVal", VoidTy, IntegerTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalVal", VoidTy, ShortTy, SizeTy, NULL);
    M.getOrInsertFunction("__captureOriginalVal", VoidTy, CharTy, SizeTy, NULL);
    M.getOrInsertFunction("__createTbXml", VoidTy, NULL);
    errs() << "got to this point 2a \n";

    return true;
  }

  void checkFunctionsStillExist(Function &F) {
    captureValuesFunction = F.getParent()->getFunction("__captureOriginalPtrVal");
    functionPtrs.insert(captureValuesFunction);
    assert(captureValuesFunction && "__captureOriginalPtrVal function has disappeared!\n");
  }
  

  bool runOnFunction(Function &F) { 
                                    
    if (F.isDeclaration())
      return false;
    
    errs() << "Running on function " << F.getName() << "\n";
    checkFunctionsStillExist(F);
    if (functionPtrs.find(&F) != functionPtrs.end())
      return false; //we are instrumenting an instrumentation function

    //We use this for all the functions otherwise we would use the indirect calls
    IRBuilder<> TheBuilder(&(*F.getEntryBlock().getFirstInsertionPt()));
    Builder = &TheBuilder;

    Value *NumInstInBB = ConstantInt::get(IntegerTy, BB.getInstList().size());
    checkCallArgsV.push_back(NumInstInBB);
    ArrayRef<Value *> bbArgsArr(checkCallArgsV);
    Builder->CreateCall(BasicBlockInstCountFunction, bbArgsArr);

    for (auto BB : F) {
      for (auto II : BB) {
        CallInstr *CI = dyn_cast<CallInstr>(II); //II.dyn_cast?
        Function *CF = CI->getCalledFunction();
        bool foundForCond = GetValueName(CF).find("for.cond") != std::string::npos; 
        bool foundForBody = GetValueName(CF).find("for.body") != std::string::npos; 
        bool foundWhileCond = GetValueName(CF).find("while.cond") != std::string::npos; 
        bool foundWhileBody = GetValueName(CF).find("while.body") != std::string::npos; 
        if (foundForCond or foundForBody or foundWhileCond or foundWhileBody) {
          //we insert instrumentation before the call to the extracted loop
          for (auto& A : CF->getArgumentList()) {
            //Type *argTy;

            //Not sure if we need to distinguish arguments for being pointers or scalars
            
            Builder->CreateCall(captureOriginalValuesFunction, A);
            
            //if (argTy = A.getParamByValType()) {
            //  

            //  PointerType *ptrToArg = argTy.getPoinerTo();
            //  Builder->CreateCall(captureOriginalValuesFunction, argTy);
            //}
            //else if (argTy = A.getParamByRefType()) {
            //  
            //}
          }
          
        }
      }
    }

    return true;
  }




  std::string getBBName(BasicBlock &BB) {
    char bbNameChrPt[500];
    if (BB.hasName()) {
      sprintf(bbNameChrPt, "%s::%s", BB.getParent()->getName().str().c_str(), \
                                     BB.getName().str().c_str());
    }
    else {
      sprintf(bbNameChrPt, "BBwithNoName%i", bbCounter);
      ++bbCounter;
    }
    std::string bbName(bbNameChrPt);
    return bbName;
  }



  void visitCallInst (CallInst &CI) {
    if (exitFunction and CI.getCalledFunction() == exitFunction) {
      Builder->SetInsertPoint(&CI);
      Builder->CreateCall(reportStatisticsFunction);
    } else if (exitFunction and CI.getCalledValue()->stripPointerCasts() == exitFunction) {
      Builder->SetInsertPoint(&CI);
      Builder->CreateCall(reportStatisticsFunction);
    } else { //we also tolerate indirect calls
      if (not CI.getCalledFunction() or not CI.getCalledFunction()->isIntrinsic()) {
        if (functionPtrs.find(CI.getCalledFunction()) == functionPtrs.end()) {
          Builder->SetInsertPoint(CI.getNextNode());
          CallInst *CI2 = Builder->CreateCall(endCallFunction);
          Builder->SetInsertPoint(&CI);
          CallInst *CI3 = Builder->CreateCall(startCallFunction); //the CreateCall routine receives Value objects


           if (MDNode *MD = CI.getMetadata("dbg")) {
             CI2->setMetadata("dbg", MD);
             CI3->setMetadata("dbg", MD);
           }
        }
      } else { //intrinsic
        Type *CharPtrTy = Type::getInt8PtrTy(TheContext);
        Value *functionNameParam = Builder->CreatePointerCast(Builder->CreateGlobalStringPtr(CI.getCalledFunction()->getName()), CharPtrTy);
        CallInst *CI2 = Builder->CreateCall(startIntrinsic, functionNameParam);
        if (MDNode *MD = CI.getMetadata("dbg"))
          CI2->setMetadata("dbg", MD);
      }
    }

  }
   // void visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
   // void visitAtomicRMWInst(AtomicRMWInst &I);
   // void visitMemIntrinsic(MemIntrinsic &MI);
  };
} // end anon namespace

char InstrumentExtractedLoopsLegacyPass::ID = 0;


FunctionPass *llvm::createInstrumentExtractedLoopsPass() {
  return new InstrumentExtractedLoopsLegacyPass();
}

char InstrumentExtractedLoopsLegacyPass::ID = 0;
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

