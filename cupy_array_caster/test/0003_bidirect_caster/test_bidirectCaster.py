import pytest
from typing import List

import bidirectCaster

import bidirect


def test_data_container_type():
    """check, if the correct Python type is returned"""
    return_value_binding = bidirectCaster.test_simple_return()
    assert isinstance(
        return_value_binding, bidirect.data_container
    ), "type of the return object: {}".format(type(return_value_binding))


def test_data_container_value():
    """check, if the correct data in the member variables are set"""
    return_value_binding = bidirectCaster.test_simple_return()
    assert return_value_binding.data == [1.0, 2.0, 3.0, 4.0]
    assert return_value_binding.scalar == 1.5


def test_copy_container():
    """Copy a datacontainer."""
    input = bidirect.data_container(data=[1.0, 2.0, 3.0, 4.0], scalar=3.0)
    expected_result = bidirect.data_container(data=[1.0, 2.0, 3.0, 4.0], scalar=3.0)
    result = bidirectCaster.copy_data(input)

    # change input after copy_data, to verify, that the result is a copy and no reference
    input.data[2] = -10.0
    input.scalar = -1.0

    assert result.data == expected_result.data
    assert result.scalar == expected_result.scalar


def test_add_to_data_container_with_scalar():
    """Test a complex arithmetic function, with data_container's as
    input and ouput.
    """

    input1 = bidirect.data_container(data=[1.0, 2.0, 3.0, 4.0], scalar=3.0)
    input2 = bidirect.data_container(data=[4.0, 12.0, -7.0, 6.5], scalar=1.5)

    expected_result = bidirect.data_container(data=[0.0, 0.0, 0.0, 0.0], scalar=1.5)

    for i in range(len(input1.data)):
        expected_result.data[i] = (
            # the same mathematical operation is done in add_scale()
            input1.scalar * input1.data[i]
            + input2.scalar * input2.data[i]
        )

    result = bidirectCaster.add_scale(input1, input2)
    assert result.data == expected_result.data
    assert result.scalar == expected_result.scalar


def test_datacontainer_wrong_args1():
    """The test checks, if an error is thrown, the attribute scale
    is missing.
    """

    class wrong_datacontainer1:
        def __init__(self, data: List[float]):
            self.data = data

    wdc1 = wrong_datacontainer1([42.0, 12.0])

    with pytest.raises(AttributeError):
        bidirectCaster.copy_data(wdc1)


def test_datacontainer_wrong_args2():
    """The test checks, if an error is thrown, the attribute data
    is missing.
    """

    class wrong_datacontainer2:
        def __init__(self, scalar: float):
            self.scalar = scalar

    wdc2 = wrong_datacontainer2(1.4)

    with pytest.raises(AttributeError):
        bidirectCaster.copy_data(wdc2)


def test_datacontainer_wrong_args3():
    """The test checks, if an error is thrown, the attributes date
    and scale are missing.
    """

    class wrong_datacontainer3:
        def __init__(self):
            pass

    wdc3 = wrong_datacontainer3()

    with pytest.raises(Exception):
        bidirectCaster.copy_data(wdc3)


def test_datacontainer_wrong_args4():
    """The test checks, if an error is thrown, when totally wrong
    attributes are passed.
    """
    with pytest.raises(TypeError):
        bidirectCaster.copy_data(3)

    with pytest.raises(TypeError):
        bidirectCaster.copy_data(3, "test")


def test_datacontainer_wrong_args5():
    """The test checks, if an error is thrown, if the attributes
    data and scale have the wrong type.
    """
    dc1 = bidirect.data_container(data=["d", "e"], scalar=1.5)

    with pytest.raises(RuntimeError):
        bidirectCaster.copy_data(dc1)

    dc2 = bidirect.data_container(data=[1.6, 2.2], scalar="foo")

    with pytest.raises(RuntimeError):
        bidirectCaster.copy_data(dc2)


def test_add_scale_different_lenght():
    """Test, if self written binding function throws the correct
    error, if input1 and input2 has not the same size.
    """
    input1 = bidirect.data_container(data=[1.0, 2.0, 3.0, 4.0], scalar=3.0)
    input2 = bidirect.data_container(data=[4.0, 12.0, 6.5], scalar=1.5)

    with pytest.raises(ValueError):
        bidirectCaster.add_scale(input1, input2)
