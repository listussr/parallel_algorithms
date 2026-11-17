"""Regression tests for the GAXPY panel; only a C compiler is required.

Run from the repository root: python3 tests/test_stage1.py
Build products live in a temporary directory, not in bin/ or results/.
"""
import ctypes
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GaxpyPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        wrapper = Path(cls.tmp.name) / "stage1_test.c"
        wrapper.write_text(
            '#define main stage1_program_main\n'
            f'#include "{(ROOT / "stage1_seq.c").as_posix()}"\n'
            '#undef main\n'
            'void test_solve(int n, const double *a, double *x) '
            '{ lu_solve(n, a, x); }\n'
        )
        library = Path(cls.tmp.name) / "stage1.so"
        subprocess.run([
            *shlex.split(os.environ.get("CC", "cc")), "-std=gnu99", "-O2",
            "-Wall", "-Wextra", "-Werror", "-fno-tree-vectorize",
            "-ffp-contract=off", "-shared", "-fPIC", str(wrapper),
            "-lm", "-o", str(library),
        ], check=True)
        cls.lib = ctypes.CDLL(str(library))
        ptr = ctypes.POINTER(ctypes.c_double)
        cls.lib.block_lu.argtypes = [ctypes.c_int, ctypes.c_int, ptr]
        cls.lib.block_lu.restype = None
        cls.lib.test_solve.argtypes = [ctypes.c_int, ptr, ptr]
        cls.lib.test_solve.restype = None

    def check_factors(self, lower, upper, alpha):
        n = len(lower)
        original = [
            sum(lower[i][k] * upper[k][j] for k in range(n))
            for i in range(n) for j in range(n)
        ]
        a = (ctypes.c_double * (n * n))(*original)
        self.lib.block_lu(n, alpha, a)
        # Check the factors against an independent, prescribed decomposition.
        for i in range(n):
            for j in range(n):
                expected = lower[i][j] if i > j else upper[i][j]
                self.assertTrue(math.isfinite(a[i * n + j]))
                self.assertAlmostEqual(a[i * n + j], expected, delta=1e-10)
        # Reconstruct A and check a relative row-sum residual.
        error = 0.0
        for i in range(n):
            row_error = 0.0
            for j in range(n):
                product = sum(
                    (1.0 if i == k else a[i * n + k]) * a[k * n + j]
                    for k in range(min(i, j) + 1)
                )
                row_error += abs(product - original[i * n + j])
            error = max(error, row_error)
        norm = max(sum(abs(x) for x in original[i*n:(i+1)*n]) for i in range(n))
        self.assertLessEqual(error / norm, 1e-12)
        # Several right-hand sides, including one unlike the program's all-ones x.
        for solution in ([1.0] * n, [(-1.0)**i * (i + 1) / n for i in range(n)]):
            rhs = (ctypes.c_double * n)(*[
                sum(original[i*n+j] * solution[j] for j in range(n))
                for i in range(n)
            ])
            self.lib.test_solve(n, a, rhs)
            self.assertLess(max(abs(rhs[i] - solution[i]) for i in range(n)), 1e-10)

    def test_panel_boundaries_and_multiple_panels(self):
        for n in [1, 2, 5, 31, 32, 33, 64, 65, 97]:
            rng = random.Random(n)
            lower = [[1.0 if i == j else rng.uniform(-0.125, 0.125)
                      if i > j else 0.0 for j in range(n)] for i in range(n)]
            upper = [[2.0 + i/n if i == j else rng.uniform(-0.125, 0.125)
                      if i < j else 0.0 for j in range(n)] for i in range(n)]
            for alpha in [1, 2, 7, 32, n + 1]:
                with self.subTest(n=n, alpha=alpha):
                    self.check_factors(lower, upper, alpha)

    def test_integer_example_from_walkthrough(self):
        lower = [[1,0,0,0,0,0], [2,1,0,0,0,0], [-1,1,1,0,0,0],
                 [1,0,2,1,0,0], [0,2,-1,1,1,0], [1,-1,1,0,2,1]]
        upper = [[2,1,-1,0,1,2], [0,3,1,2,0,-1], [0,0,2,1,-1,0],
                 [0,0,0,4,1,1], [0,0,0,0,3,-2], [0,0,0,0,0,2]]
        for alpha in [1, 2, 4, 6, 8]:
            with self.subTest(alpha=alpha):
                self.check_factors(lower, upper, alpha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
