#pragma once

template<typename TData, int TDim = 0>
class Hip_Ref {
  public :
    TData * ptr;
    std::string dtype;
    std::vector<unsigned int> shape;

    Hip_Ref(){}
};