#pragma once

template<typename TData, int TDim = 0>
class Cupy_Ref {
  public :
    TData * ptr;
    std::string dtype, typestr;
    std::vector<unsigned int> shape;

    Cupy_Ref(){}

    static Cupy_Ref getCupyRef(pybind11::object obj)
    {
        if(pybind11::module::import("builtins").attr("str")(obj.attr("__class__")).cast<std::string>() != "<class 'cupy._core.core.ndarray'>")
        {
            throw std::runtime_error("Exception : Python object must be a cupy array");   
        }

        return Cupy_Ref(obj);
    }

  private:

    Cupy_Ref(pybind11::object obj)
    {
      ptr = reinterpret_cast<TData *>(obj.attr("data").attr("ptr").cast<std::size_t>());
      dtype = pybind11::module::import("builtins").attr("str")(obj.attr("dtype")).cast<std::string>();
      shape = obj.attr("shape").cast<std::vector<unsigned int>>();
      typestr = pybind11::module::import("builtins").attr("str")(obj.attr("dtype").attr("str")).cast<std::string>();
    }

    
};