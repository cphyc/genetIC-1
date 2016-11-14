import numpy as np
import scipy.fftpack
import functools
import math

def unitary_fft(x):
    return scipy.fftpack.rfft(x)/math.sqrt(float(len(x)))

def unitary_inverse_fft(x):
    return scipy.fftpack.irfft(x)*math.sqrt(float(len(x)))

class FFTArray(np.ndarray):
    def __new__(subtype, data, **kwargs):
        X = np.asarray(data,**kwargs).view(subtype)
        X.fourier = False
        return X

    def __array_finalize__(self, obj):
        self.fourier = getattr(obj, 'fourier', False)

    def __init__(self, *args, **kwargs):
        assert hasattr(self, 'fourier')

    def in_fourier_space(self):
        if not self.fourier:
            self[:] = unitary_fft(self)
            self.fourier = True
            return self

        return self

    def in_real_space(self):
        if self.fourier:
            self[:] = unitary_inverse_fft(self)
            self.fourier = False

        return self

def _converter(fn, call='in_fourier_space'):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        for a in args:
            if hasattr(a, call):
                new_args.append(getattr(a, call)())
            else:
                new_args.append(a)

        for k, v in kwargs.iteritems():
            if hasattr(v, call):
                new_kwargs[k] = getattr(v, call)()
            else:
                new_kwargs[k] = v

        return fn(*args, **kwargs)

    return wrapped

def in_fourier_space(fn):
    """Function decorator to ensure all FFTArray inputs are given in Fourier space"""
    return _converter(fn, 'in_fourier_space')

def in_real_space(fn):
    """Function decorator to ensure all FFTArray inputs are given in real space"""
    return _converter(fn, 'in_real_space')
