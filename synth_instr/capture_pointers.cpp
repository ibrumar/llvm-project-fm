#include <iostream>
using namespace std;

std::map<unsigned int, long long int> SizesCache;

ofstream xmlfile;

int initialParamIdInVerilog = 5;

//TODO: parse initial param id in verilog.
//TODO: use templates to cover all the types with the same implementation.

//for each pointer we will also register the original name
void setPtrSize(void *ptr, long long int size) {
  SizesCache[(int) ptr] = size;
}

void __captureOriginalDoublePtrVal(double *arr, long long int paramIdx) {
  if (paramIdx == 0) {
    //create file with params
    xmlfile.open("file_with_values.xml", ios::out);
    xmlfile.write("<?xml version="1.0"?>\n<function>\n<testbench ");
  }
  
  xmlfile.write("Pd%i=\"{".format(paramIdx + initialParamIdInVerilog));
  long long int paramSize = SizesCache[(int) arr];
  for (int i = 0; i < paramSize; ++i) {
    if (i == paramSize - 1)
      xmlfile.write("%d}".format(arr[i]));
    else
      xmlfile.write("%d, ".format(arr[i]));
  }
  xmlfile.write("}\" ");
}

void __createTbXm() {
  xmlfile.write("/>\n</function>");
  xmlfile.close();
}
