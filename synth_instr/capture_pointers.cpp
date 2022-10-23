#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <memory>
using namespace std;

//extern C is necessary for linking against llvm that didn't declare these functions
extern "C" void __captureOriginalDoublePtrVal(double *arr, long long int paramIdx);
extern "C" void __captureOriginalFloatPtrVal(float *arr, long long int paramIdx);
extern "C" void __captureOriginalIntPtrVal(int *arr, long long int paramIdx);
extern "C" void __captureOriginalShortPtrVal(short *arr, long long int paramIdx);
extern "C" void __captureOriginalCharPtrVal(char *arr, long long int paramIdx);

extern "C" void __captureOriginalDoubleVal(double val, long long int paramIdx);
extern "C" void __captureOriginalFloatVal(float val, long long int paramIdx);
extern "C" void __captureOriginalIntegerVal(int val, long long int paramIdx);
extern "C" void __captureOriginalShortVal(short val, long long int paramIdx);
extern "C" void __captureOriginalCharVal(char val, long long int paramIdx);


extern "C" void __createTbXml(const char *funcName, long long int numArgs);
extern void __setPtrSize(void *ptr, long long int size);

std::map<long long int, long long int> SizesCache;

ofstream xmlfile;

int initialParamIdInVerilog = 0;
long long int numArgsGl;

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}



//TODO: parse initial param id in verilog.
//TODO: use templates to cover all the types with the same implementation.

//for each pointer we will also register the original name
void __setPtrSize(void *ptr, long long int size) {
  SizesCache[(long long int) ptr] = size;
}


template <typename T>
void __captureOriginalPtrTemp(T *arr, long long int paramIdx) { 
  std::string param_id; 
  xmlfile << string_format("P%lli=\"{", paramIdx + initialParamIdInVerilog);
  long long int paramSize = SizesCache[(long long int) arr];
  for (int i = 0; i < paramSize; ++i) {
    if (i == paramSize - 1)
      xmlfile << arr[i];
    else
      xmlfile << arr[i] << ", ";
  }
  xmlfile << "}\" ";

  if (paramIdx == numArgsGl - 1) {
    xmlfile << "/>\n</function>";
    xmlfile.close();
  }
}

//the only difference with the previous method
template <typename T>
void __captureOriginalValTemp(T val, long long int paramIdx) { 
  std::string param_id; 
  xmlfile << string_format("P%lli=\"", paramIdx + initialParamIdInVerilog);
  xmlfile << val;
  xmlfile << "\" ";

  if (paramIdx == numArgsGl - 1) {
    xmlfile << "/>\n</function>";
    xmlfile.close();
  }
}


void __captureOriginalDoublePtrVal(double *arr, long long int paramIdx) {
  __captureOriginalPtrTemp(arr, paramIdx);
}


void __captureOriginalIntPtrVal(int *arr, long long int paramIdx) {
  __captureOriginalPtrTemp(arr, paramIdx);
}


void __captureOriginalFloatPtrVal(float *arr, long long int paramIdx) {
  __captureOriginalPtrTemp(arr, paramIdx);
}


void __captureOriginalCharPtrVal(char *arr, long long int paramIdx) {
  __captureOriginalPtrTemp(arr, paramIdx);
}



void __captureOriginalShortPtrVal(short *arr, long long int paramIdx) {
  __captureOriginalPtrTemp(arr, paramIdx);
}

//Scalars implementation


void __captureOriginalDoubleVal(double val, long long int paramIdx) {
  __captureOriginalValTemp(val, paramIdx);
}


void __captureOriginalIntegerVal(int val, long long int paramIdx) {
  __captureOriginalValTemp(val, paramIdx);
}


void __captureOriginalFloatVal(float val, long long int paramIdx) {
  __captureOriginalValTemp(val, paramIdx);
}


void __captureOriginalCharVal(char val, long long int paramIdx) {
  __captureOriginalValTemp(val, paramIdx);
}



void __captureOriginalShortVal(short val, long long int paramIdx) {
  __captureOriginalValTemp(val, paramIdx);
}

//this happens before any capture function
void __createTbXml(const char *funcName, long long int numArgs) {
  std::cout << "Writting from here1\n";
  numArgsGl = numArgs;

  //create file with params
  xmlfile.open(string_format("file_with_values_%s.xml", funcName), ios::out);
  xmlfile << "<?xml version=\"1.0\"?>\n<function>\n<testbench ";

}
