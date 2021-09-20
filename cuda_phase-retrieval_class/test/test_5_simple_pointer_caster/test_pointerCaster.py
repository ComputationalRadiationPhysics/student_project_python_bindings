import pytest
from typing import List

import numpy as np

import pointerCaster  # Python bindings to C++ function
import numpy_ref


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


def test_passing_pointer_with_numpy():
    """The test demonstrate, how to pass the pointer of a single nunpy value via
    pybind11 to a C++ function.

    The test test_passing_pointer_with_numpy() and test_passing_numpy_pointer_and_increment_values()
    demonstrate, how to pass pointers from Python to C++. Numpy is only used, because it is a
    easy way, to allocate memory and get the pointer of it in Python. It should be also possible,
    with nativ Python types, if the structure of the type is know and how to get the memory pointer.
    """
    i = np.uint32(3)

    print("[python] memory adress: {}".format(i.__array_interface__["data"][0]))
    assert (
        pointerCaster.dereference_pointer_numpy_uint32(i.__array_interface__["data"][0])
        == i
    )
    # int i = 3 and pointerCaster.dereference_pointer_numpy_uint32(id(i)) didn't work
    # Python int is not equal to C++ int


def test_passing_numpy_pointer_and_increment_values():
    """The test demonstrate, how to pass the pointer of a nunpy array via
    pybind11 to a C++ function. This allows the C++ function to modify the
    content of the numpy array.
    """
    input = np.arange(4, dtype=np.uint32)
    result = np.copy(input)
    offset = np.uint32(4)

    for i in range(len(result)):
        result[i] += offset

    pointerCaster.add_to_numpy_array_uint32(
        input.__array_interface__["data"][0], len(input), offset
    )

    assert np.array_equal(
        input,
        result,
    )


def test_add_to_simple_numpy_ref():
    """Instead using direct a numpy Python ptr, use a wrapper class to allow
    a better C++ interface.

    add_to_simple_numpy_ref() adds the value of offset to each element of
    the numpy array.
    """
    input = np.arange(4, dtype=np.uint32)
    result = np.copy(input)
    offset = np.uint32(10)

    for i in range(len(result)):
        result[i] += offset

    simple_numpy_ref = numpy_ref.Simple_Numpy_Ref(
        ptr=input.__array_interface__["data"][0]
    )

    pointerCaster.add_to_simple_numpy_ref(simple_numpy_ref, len(input), offset)

    assert np.array_equal(
        input,
        result,
    )


def test_advanced_numpy_ref():
    """Advanced_Numpy_Ref is a data struct, which allows easily to iterate over
    the data.

    init_advanced_numpy_ref() does the same like arange without memory allocation.
    """
    input = np.zeros(4, dtype=np.uint32)
    result = np.arange(4, dtype=np.uint32)

    advanced_numpy_ref = numpy_ref.Advanced_Numpy_Ref(
        ptr=input.__array_interface__["data"][0], size=len(input)
    )

    pointerCaster.init_advanced_numpy_ref(advanced_numpy_ref)

    assert np.array_equal(
        input,
        result,
    )


def test_add_to_advanced_numpy_ref():
    """add_to_advanced_numpy_ref() adds the value of offset to each element of
    the numpy array.
    """
    input = np.arange(4, dtype=np.uint32)
    result = np.copy(input)
    offset = np.uint32(8)

    for i in range(len(result)):
        result[i] += offset

    advanced_numpy_ref = numpy_ref.Advanced_Numpy_Ref(
        ptr=input.__array_interface__["data"][0], size=len(input)
    )

    pointerCaster.add_to_advanced_numpy_ref(advanced_numpy_ref, offset)

    assert np.array_equal(
        input,
        result,
    )
