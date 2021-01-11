#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstdio>

namespace py = pybind11;

// double* fienup_phase_retrieval(double* mag, int steps, bool verbose)
// {
//     double* x;
//     int* mask = NULL;
//     double beta = 0.8;

//     //mode = hybrid
//     int mode = 3;


//     return x;
// }

py::array_t<double> fienup_phase_retrieval(py::array_t<double> mag) {
   /* read input arrays buffer_info */
   py::buffer_info bufMag = mag.request();

   /* allocate the output buffer */
   py::array_t<double> result = py::array_t<double>(bufMag.size);
   py::buffer_info bufres = result.request();
   double *ptr1 = (double *) bufMag.ptr, *ptrres = (double *)bufres.ptr;
   size_t X = bufMag.shape[0];
   size_t Y = bufMag.shape[1];

   /* Add both arrays */
   for (size_t idx = 0; idx < X; idx++)
       for (size_t idy = 0; idy < Y; idy++)
       {
           ptrres[idx*Y + idy] = ptr1[idx*Y+ idy];
       }

   /* Reshape result to have same shape as input */
   result.resize({X,Y});

   return result;
}
