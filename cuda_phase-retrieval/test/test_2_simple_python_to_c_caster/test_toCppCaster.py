import pytest
import toCppCaster
import toCpp


def test_toCpp():
    """The test requires, that the toCpp object is correct casted,
    if it is passed to the binding function test_toCpp().
    """

    tc = toCpp.toCpp(version=42, text="the answer is")

    assert toCppCaster.test_toCpp(tc) == "the answer is 42"


def test_toCpp_wrong_args1():
    """The test checks, if an error is thrown, the attribute text
    is missing.
    """

    class wrongToCpp1:
        def __init__(self, version: int):
            self.version = version

    wtc1 = wrongToCpp1(42)

    with pytest.raises(AttributeError):
        toCppCaster.test_toCpp(wtc1)


def test_toCpp_wrong_args2():
    """The test checks, if an error is thrown, the attribute version
    is missing.
    """

    class wrongToCpp2:
        def __init__(self, text: str):
            self.text = text

    wtc2 = wrongToCpp2("the answer is")

    with pytest.raises(AttributeError):
        toCppCaster.test_toCpp(wtc2)


def test_toCpp_wrong_args3():
    """The test checks, if an error is thrown, the attributes text
    and version are missing.
    """

    class wrongToCpp3:
        def __init__(self):
            pass

    wtc3 = wrongToCpp3()

    with pytest.raises(Exception):
        toCppCaster.test_toCpp(wtc3)


def test_toCpp_wrong_args4():
    """The test checks, if an error is thrown, totally wrong
    attributes are passed.
    """
    with pytest.raises(TypeError):
        toCppCaster.test_toCpp(3)

    with pytest.raises(TypeError):
        toCppCaster.test_toCpp(3, "test")


def test_toCpp_wrong_args5():
    """The test checks, if an error is thrown, if the attributes
    text and version have the wrong type.
    """
    tc1 = toCpp.toCpp(version="version string", text=4.2)

    with pytest.raises(RuntimeError):
        toCppCaster.test_toCpp(tc1)

    tc2 = toCpp.toCpp(version=1, text=4.2)

    with pytest.raises(RuntimeError):
        toCppCaster.test_toCpp(tc2)
