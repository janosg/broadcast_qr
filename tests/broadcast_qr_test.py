from numpy.testing import assert_array_almost_equal as aaae
from broadcast_qr import r_from_qr
import numpy as np
from numpy.core.umath_tests import matrix_multiply


def a_prime_a(a):
    return matrix_multiply(np.transpose(a, axes=(0, 2, 1)), a)


class TestRFromQR:
    def setup(self):
        pass

    def test_non_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 7, 3)
        self.expected_prod = a_prime_a(self.some_array)
        r_from_qr(self.some_array, np.ones(1))
        aaae(a_prime_a(self.some_array), self.expected_prod)

    def test_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 12, 12)
        self.expected_prod = a_prime_a(self.some_array)
        r_from_qr(self.some_array, np.ones(1))
        aaae(a_prime_a(self.some_array), self.expected_prod)

if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
