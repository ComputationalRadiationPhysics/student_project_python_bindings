import pytest
from typing import List

import numpy as np

import pointerCaster  # Python bindings to C++ function


def test_modify_numpy_array_as_arg():
    """Test, if a numpy array can be passed as reference to a function
    void change_first_element_numpy_array(py::array_t<int> &input);
    """
    # create a numpy array and set first element to a value (0)
    np_array = np.arange(1)
    assert np_array[0] == 0

    # function changes the first value of an numpy array to 42
    pointerCaster.change_first_element_numpy_array(np_array)
    # passing as reference does not work
    assert np_array[0] != 42
    # check, if first value is still the same an not undefined
    assert np_array[0] == 0


def test_access_numpy_ptr_cpp():
    """The test compares the memory pointer of a numpy array on the
    Python and C++ side.
    """
    np_array = np.arange(1)

    # get memory address of the numpy array data
    python_np_ptr = hex(np_array.__array_interface__["data"][0])
    cpp_np_ptr = pointerCaster.return_numpy_ptr_as_string(np_array)
    cpp_np_ptr2 = pointerCaster.return_numpy_ptr_as_string(np_array)

    # if the C++ side uses the same numpy array, as on the Python side, the memory address
    # should be the same
    # Memory addresses are not the same -> numpy array is passed as value and copied
    assert python_np_ptr != cpp_np_ptr, "python address: {} | cpp address {}".format(
        python_np_ptr, cpp_np_ptr
    )

    # Verify if numpy array is passed as value (copied). If the numpy array is copied each time,
    # the memory address of the data pointer should be different if the function
    # return_numpy_ptr_as_string() is called twice.
    assert cpp_np_ptr != cpp_np_ptr2, "cpp address 1: {} | cpp address 1 {}".format(
        cpp_np_ptr, cpp_np_ptr2
    )
