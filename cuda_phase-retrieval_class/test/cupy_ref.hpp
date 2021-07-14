#pragma once

using namespace std;
namespace py = pybind11;

//change to 0
template<typename TData, int TDim = 0>
class Custom_Cupy_Ref {
  public :
    TData * ptr;
    size_t size;
    string dtype;
    vector<size_t> shape;

    Custom_Cupy_Ref(){}

    static Custom_Cupy_Ref getCustomCupyRef(py::object obj)
    {
        if(py::module::import("builtins").attr("str")(obj.attr("__class__")).cast<string>() != "<class 'cupy.core.core.ndarray'>")
        {
            throw std::runtime_error("Exception : Python object must be a cupy array");   
        }

        return Custom_Cupy_Ref(obj);
    }

  private:

    Custom_Cupy_Ref(py::object obj)
    {
      ptr = reinterpret_cast<TData *>(obj.attr("data").attr("ptr").cast<size_t>());
      size = obj.attr("size").cast<size_t>();
      dtype = py::module::import("builtins").attr("str")(obj.attr("dtype")).cast<string>();
      shape = obj.attr("shape").cast<vector<size_t>>();
    }
};