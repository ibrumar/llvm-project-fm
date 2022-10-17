#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <memory>
using namespace std;

std::map<long long int, long long int> SizesCache;

ofstream xmlfile;

int initialParamIdInVerilog = 5;
int numArgsGl;

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
void setPtrSize(void *ptr, long long int size) {
  SizesCache[(long long int) ptr] = size;
}

void __captureOriginalDoublePtrVal(double *arr, long long int paramIdx) {
  std::string param_id; 
  xmlfile << string_format("Pd%lli=\"{", paramIdx + initialParamIdInVerilog);
  long long int paramSize = SizesCache[(long long int) arr];
  for (int i = 0; i < paramSize; ++i) {
    if (i == paramSize - 1)
      xmlfile << string_format("%f}", arr[i]);
    else
      xmlfile << string_format("%f, ", arr[i]);
  }
  xmlfile << "}\" ";

  if (paramIdx == numArgsGl - 1) {
    xmlfile << "/>\n</function>";
    xmlfile.close();
  }
}

//this happens before any capture function
void __createTbXml(const char *funcName, int numArgs) {
  numArgsGl = numArgs;

  //create file with params
  xmlfile.open(string_format("file_with_values_%s.xml", funcName), ios::out);
  xmlfile << "<?xml version=\"1.0\"?>\n<function>\n<testbench ";

}
