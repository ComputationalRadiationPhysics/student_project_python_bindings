#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

/*To avoid every unrelated problem, easiest version of C++ struct
to pass ptr from Python to C++.*/
struct Simple_Numpy_Ref {
  std::uint32_t * ptr;
};

/* more realistic data container */
struct Advanced_Numpy_Ref {
  std::uint32_t * ptr;
  std::size_t size;

  std::uint32_t & operator[](std::size_t idx){
    if(idx >= size){
      throw std::out_of_range("Advanced_Numpy_Ref: "
			      + std::to_string(idx)
			      + " is out of range "
			      + std::to_string(size) );
    }

    return ptr[idx];
  }

  /* begin() and end() is required for ranged based loops */
  std::uint32_t * begin(){
    return ptr;
  }

  std::uint32_t * end(){
    return ptr + size;
  }
};
