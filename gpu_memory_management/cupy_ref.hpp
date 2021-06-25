#pragma once

template<typename TData>
class Custom_Cupy_Ref {
  public :
    TData * ptr;
    std::size_t size;
    std::string dtype;
};