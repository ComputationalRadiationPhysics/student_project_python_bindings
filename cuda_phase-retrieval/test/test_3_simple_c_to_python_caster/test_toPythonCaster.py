import pytest

import toPythonCaster

import toPython


def test_toPython_type():
    """check, if the correct Python type is returned"""
    return_value_binding = toPythonCaster.test_toPython()
    assert isinstance(
        return_value_binding, toPython.toPython
    ), "type of the return object: {}".format(type(return_value_binding))


def test_toPython_value():
    """check, if the correct data in the member variables are set"""
    return_value_binding = toPythonCaster.test_toPython()
    assert return_value_binding.number == 3.5
    assert return_value_binding.data == [1, 2, 3, 4]
