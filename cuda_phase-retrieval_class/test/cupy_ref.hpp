#pragma once

using namespace std;
namespace py = pybind11;

template<typename TData, int TDim = 2>
class Custom_Cupy_Ref {
  public :
    TData * ptr;
    size_t size;
    string dtype;
    vector<size_t> shape;

    Custom_Cupy_Ref(){}

    Custom_Cupy_Ref(py::object obj)
    {
      ptr = reinterpret_cast<TData *>(obj.attr("data").attr("ptr").cast<size_t>());
      size = obj.attr("size").cast<size_t>();
      dtype = py::module::import("builtins").attr("str")(obj.attr("dtype")).cast<string>();
      shape = obj.attr("shape").cast<vector<size_t>>();
    }
};