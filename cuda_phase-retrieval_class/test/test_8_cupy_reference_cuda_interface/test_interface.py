import cupy as cp
import cupy_ref
import pytest

# Problem : altoutgh "a" and "b.as_cupy()" has the same value and cupy_ref "b" is coming from cupy "a", 
# "a" and "b.as_cupy()" are 2 different cupy object
def test_return_as_cupy():
    a = cp.array([4,5,6])
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))

    assert(cp.array_equal(b.as_cupy(), a))

