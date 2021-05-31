#pragma once

/*To avoid every unrelated problem, easiest version of C++ struct
to pass ptr from Python to C++.*/
struct simple_ptr {
  int * ptr;
};

/* more realistic data container */
struct real_container {
  int * ptr;
  int size;
};
