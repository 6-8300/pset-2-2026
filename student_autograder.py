#!/usr/bin/env python3
"""
Combined Local Test Suite for All Parts
6.8300 - Advances in Computer Vision

Run this to verify your implementations before submitting to Gradescope.
Note: These tests use different values than the Gradescope autograder,
so passing these tests indicates correct implementation, not memorized answers.

Usage:
    python test_all_local.py
"""

import sys
import argparse
import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import nbformat
from sqlalchemy import func

# Try to import torch (needed for parts 1 and 3)
try:
    import torch
    from scipy.spatial.transform import Rotation as R
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ============================================================================
# Common utilities
# ============================================================================

def extract_functions_from_notebook(notebook_path: str, namespace: dict) -> dict:
    """Extract and execute code cells from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            code = cell.source
            if code.strip().startswith('%') or code.strip().startswith('!'):
                continue

            # Check if this cell defines important functions we need
            important_functions = ['gaussian_filter', '_conv', '_gaussian_filter_1d',
                                 'astronaut', 'hubble', 'imshow']
            has_important_func = any(f'def {func}' in code for func in important_functions)

            # Skip plotting cells unless they contain important function definitions
            if not has_important_func:
                if 'plt.show()' in code or 'plt.figure' in code:
                    continue
                if ('fig,' in code or 'fig=' in code) and 'def ' not in code:
                    continue

            if 'from helper_code' in code or 'from src.' in code:
                continue
            if 'load_video' in code or 'save_video' in code:
                continue
            if 'load_image(' in code and 'def load_image' not in code:
                continue
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    exec(code, namespace)
            except Exception:
                pass

    return namespace


def find_notebook(pattern: str) -> str:
    """Find a notebook matching the pattern."""
    search_path = Path(".")
    notebooks = list(search_path.glob(f"*{pattern}*.ipynb"))
    if notebooks:
        for nb in notebooks:
            if "solution" in nb.name.lower():
                return str(nb)
        return str(notebooks[0])
    return None


PART1_NAMESPACE = {}


def load_functions():
    """Load functions from Part 1 notebook."""
    global PART1_NAMESPACE
    if PART1_NAMESPACE:
        return

    if not HAS_TORCH:
        return

    notebook_path = find_notebook("fourier_and_gauss")
    if notebook_path:
        print(f"Loading from {notebook_path}")
        namespace = {
            'torch': torch,
            'np': np,
            'numpy': np,
            'pi': np.pi,
        }
        PART1_NAMESPACE = extract_functions_from_notebook(notebook_path, namespace)
    else:
        print("No notebook found")


def get_part1_func(name):
    return PART1_NAMESPACE.get(name)


def f32(x):
    return torch.tensor(x, dtype=torch.float32)


class TestPart1Identity(unittest.TestCase):
    """Part 1: Tests for identity function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_identity_basic(self):
        func = get_part1_func('identity')
        coeffs_to_sine = get_part1_func('coeffs_to_sine')
        if func is None or coeffs_to_sine is None:
            self.skipTest("identity or coeffs_to_sine not implemented")

        a = f32([3.0])
        b = f32([4.0])

        expected_R = f32([5.0])
        expected_theta = f32([0.643501])
        result_R, result_theta = func(a, b)

        self.assertTrue(torch.allclose(result_R, expected_R, atol=1e-5))
        self.assertTrue(torch.allclose(result_theta, expected_theta, atol=1e-5))

        # Verify that a*cos(x) + b*sin(x) = R*sin(x + theta)
        x = torch.linspace(0, 2 * torch.pi, 100)
        lhs = coeffs_to_sine(a, b, x)
        rhs = result_R * torch.sin(x + result_theta)
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-5),
                       "a*cos(x) + b*sin(x) should equal R*sin(x + theta)")

    def test_identity_batch(self):
        func = get_part1_func('identity')
        coeffs_to_sine = get_part1_func('coeffs_to_sine')

        if func is None or coeffs_to_sine is None:
            self.skipTest("identity or coeffs_to_sine not implemented")

        a = f32([[3.0, 6.0], [4.0, 8.0]])
        b = f32([[4.0, 8.0], [3.0, 6.0]])

        R, theta = func(a, b)

        x = torch.linspace(0, 2 * torch.pi, 100)
        lhs = coeffs_to_sine(a.unsqueeze(-1), b.unsqueeze(-1), x)
        rhs = R.unsqueeze(-1) * torch.sin(x + theta.unsqueeze(-1))
        self.assertTrue(
            torch.allclose(lhs, rhs, atol=1e-5),
            "a*cos(x) + b*sin(x) should equal R*sin(x + theta)"
        )

class TestPart1CoeffsToSine(unittest.TestCase):
    """Part 1.2: Tests for coeffs_to_sine function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_coeffs_to_sine_cosine_only(self):
        func = get_part1_func('coeffs_to_sine')
        if func is None:
            self.skipTest("coeffs_to_sine not implemented")

        a = f32(1.0)
        b = f32(0.0)
        x = f32([0.0, torch.pi / 2, torch.pi])
        result = func(a, b, x)
        expected = f32([1.0, 0.0, -1.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_coeffs_to_sine_sine_only(self):
        func = get_part1_func('coeffs_to_sine')
        if func is None:
            self.skipTest("coeffs_to_sine not implemented")

        a = f32(0.0)
        b = f32(1.0)
        x = f32([0.0, torch.pi / 2, torch.pi])
        result = func(a, b, x)
        expected = f32([0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_coeffs_to_sine_combination(self):
        func = get_part1_func('coeffs_to_sine')
        if func is None:
            self.skipTest("coeffs_to_sine not implemented")

        a = f32(0.6)
        b = f32(0.8)
        x = f32([0.0, torch.pi / 2, torch.pi])
        result = func(a, b, x)
        expected = f32([0.6, 0.8, -0.6])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

class TestPart1AngleToCoeffs(unittest.TestCase):
    """Part 1.2: Tests for angle_to_coeffs function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_angle_to_coeffs_zero(self):
        func = get_part1_func('angle_to_coeffs')
        if func is None:
            self.skipTest("angle_to_coeffs not implemented")

        angle = f32(0.0)
        cos_a, sin_a = func(angle)
        self.assertTrue(torch.isclose(cos_a, f32(1.0), atol=1e-5))
        self.assertTrue(torch.isclose(sin_a, f32(0.0), atol=1e-5))

    def test_angle_to_coeffs_pi_half(self):
        func = get_part1_func('angle_to_coeffs')
        if func is None:
            self.skipTest("angle_to_coeffs not implemented")

        angle = f32(torch.pi / 2)
        cos_a, sin_a = func(angle)
        self.assertTrue(torch.isclose(cos_a, f32(0.0), atol=1e-5))
        self.assertTrue(torch.isclose(sin_a, f32(1.0), atol=1e-5))

    def test_angle_to_coeffs_pi_quarter(self):
        func = get_part1_func('angle_to_coeffs')
        if func is None:
            self.skipTest("angle_to_coeffs not implemented")

        angle = f32(torch.pi / 4)
        cos_a, sin_a = func(angle)
        expected_val = f32(0.7071068)
        self.assertTrue(torch.isclose(cos_a, expected_val, atol=1e-5))
        self.assertTrue(torch.isclose(sin_a, expected_val, atol=1e-5))

    def test_angle_to_coeffs_unit_circle(self):
        """Verify coefficients lie on unit circle."""
        func = get_part1_func('angle_to_coeffs')
        if func is None:
            self.skipTest("angle_to_coeffs not implemented")

        angles = torch.linspace(0, 2 * torch.pi, 20)
        for angle in angles:
            cos_a, sin_a = func(angle)
            # Should lie on unit circle: cos²(θ) + sin²(θ) = 1
            radius_squared = cos_a**2 + sin_a**2
            self.assertTrue(torch.isclose(radius_squared, f32(1.0), atol=1e-5),
                           f"Point should be on unit circle, got radius² = {radius_squared}")

class TestPart2ShiftOperator(unittest.TestCase):
    """Part 2.1: Tests for shift_operator function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_shift_operator_horizontal(self):
        func = get_part1_func('shift_operator')
        if func is None:
            self.skipTest("shift_operator not implemented")

        img_shape = (3, 3)
        shift_x = 1
        shift_y = 0
        S = func(img_shape, shift_x, shift_y)

        img = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        shifted = (S @ img.flatten()).reshape(img_shape)
        expected = torch.tensor([[2.0, 3.0, 1.0], [5.0, 6.0, 4.0], [8.0, 9.0, 7.0]])
        self.assertTrue(torch.allclose(shifted, expected, atol=1e-5))

    def test_shift_operator_vertical(self):
        func = get_part1_func('shift_operator')
        if func is None:
            self.skipTest("shift_operator not implemented")

        img_shape = (3, 3)
        shift_x = 0
        shift_y = 1
        S = func(img_shape, shift_x, shift_y)

        img = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        shifted = (S @ img.flatten()).reshape(img_shape)
        expected = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(shifted, expected, atol=1e-5))

    def test_shift_operator_both(self):
        func = get_part1_func('shift_operator')
        if func is None:
            self.skipTest("shift_operator not implemented")

        img_shape = (4, 4)
        shift_x = 2
        shift_y = 1
        S = func(img_shape, shift_x, shift_y)

        img = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]])
        shifted = (S @ img.flatten()).reshape(img_shape)
        expected = torch.tensor([[7.0, 8.0, 5.0, 6.0],
                                [11.0, 12.0, 9.0, 10.0],
                                [15.0, 16.0, 13.0, 14.0],
                                [3.0, 4.0, 1.0, 2.0]])
        self.assertTrue(torch.allclose(shifted, expected, atol=1e-5))

class TestPart2MatrixFromConv(unittest.TestCase):
    """Part 2.2: Tests for matrix_from_convolution_kernel function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_matrix_from_conv_shape(self):
        func = get_part1_func('matrix_from_convolution_kernel')
        if func is None:
            self.skipTest("matrix_from_convolution_kernel not implemented")

        kernel = torch.tensor([1.0, 2.0, 1.0])
        n = 5
        M = func(kernel, n)
        self.assertEqual(M.shape, (n, n))

    def test_matrix_from_conv_circulant(self):
        func = get_part1_func('matrix_from_convolution_kernel')
        if func is None:
            self.skipTest("matrix_from_convolution_kernel not implemented")

        kernel = torch.tensor([1.0, 2.0, 1.0])
        n = 6
        M = func(kernel, n)

        # Check circulant property
        for i in range(1, n):
            self.assertTrue(torch.allclose(M[i], torch.roll(M[0], i), atol=1e-5))

    def test_matrix_from_conv_symmetric(self):
        func = get_part1_func('matrix_from_convolution_kernel')
        if func is None:
            self.skipTest("matrix_from_convolution_kernel not implemented")

        kernel = torch.tensor([1.0, 2.0, 1.0])
        n = 7
        M = func(kernel, n)

        # Symmetric kernel should produce symmetric matrix
        self.assertTrue(torch.allclose(M, M.T, atol=1e-5))

    def test_matrix_from_conv_correct_values(self):
        """Verify the circulant matrix produces correct convolution."""
        func = get_part1_func('matrix_from_convolution_kernel')
        if func is None:
            self.skipTest("matrix_from_convolution_kernel not implemented")

        kernel = torch.tensor([0.25, 0.5, 0.25])
        n = 5
        M = func(kernel, n)

        signal = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        result = M @ signal
        expected = torch.tensor([0.0, 0.25, 0.5, 0.25, 0.0])

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-5),
            f"Convolution result incorrect. Got {result}, expected {expected}"
        )

class TestPart2ImageOpFromMatrix(unittest.TestCase):
    """Part 2.2: Tests for image_operator_from_sep_kernels function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_image_op_shape(self):
        func = get_part1_func('image_operator_from_sep_kernels')
        if func is None:
            self.skipTest("image_operator_from_sep_kernels not implemented")

        img_shape = (3, 4)
        kernel = torch.tensor([1.0, 1.0, 1.0]) / 3
        M = func(img_shape, kernel, kernel)

        h, w = img_shape
        self.assertEqual(M.shape, (h * w, h * w))

    def test_image_op_symmetric(self):
        func = get_part1_func('image_operator_from_sep_kernels')
        if func is None:
            self.skipTest("image_operator_from_sep_kernels not implemented")

        img_shape = (4, 4)
        kernel = torch.tensor([1.0, 2.0, 1.0]) / 4
        M = func(img_shape, kernel, kernel)

        # Symmetric kernels should produce symmetric operator
        self.assertTrue(torch.allclose(M, M.T, atol=1e-5))

    def test_image_op_commutes_with_shift(self):
        func_conv = get_part1_func('image_operator_from_sep_kernels')
        func_shift = get_part1_func('shift_operator')
        if func_conv is None or func_shift is None:
            self.skipTest("Required functions not implemented")

        img_shape = (3, 3)
        kernel = torch.tensor([1.0, 1.0, 1.0]) / 3
        M = func_conv(img_shape, kernel, kernel)
        S = func_shift(img_shape, 1, 1)

        # Shift and convolution should commute
        self.assertTrue(torch.allclose(S @ M, M @ S, atol=1e-4))

class TestPart2EigenDecomposition(unittest.TestCase):
    """Part 2.3: Tests for eigendecomposition function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_eigendecomp_descending(self):
        func = get_part1_func('eigendecomposition')
        if func is None:
            self.skipTest("eigendecomposition not implemented")

        A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        eigenvalues, _ = func(A, descending=True)

        # Check descending order
        self.assertTrue(eigenvalues[0] >= eigenvalues[1])

    def test_eigendecomp_reconstruction(self):
        func = get_part1_func('eigendecomposition')
        if func is None:
            self.skipTest("eigendecomposition not implemented")

        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        eigenvalues, eigenvectors = func(A, descending=True)

        # Reconstruct matrix
        reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        self.assertTrue(torch.allclose(reconstructed, A, atol=1e-4))

    def test_eigendecomp_larger_matrix(self):
        func = get_part1_func('eigendecomposition')
        if func is None:
            self.skipTest("eigendecomposition not implemented")

        torch.manual_seed(123)
        A = torch.randn(5, 5)
        A = (A + A.T) / 2  # Make symmetric

        eigenvalues, eigenvectors = func(A, descending=True)

        # Verify eigendecomposition
        reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        self.assertTrue(torch.allclose(reconstructed, A, atol=1e-4))

    def test_eigendecomp_eigenvector_equation(self):
        """Verify that Av = λv for each eigenpair."""
        func = get_part1_func('eigendecomposition')
        if func is None:
            self.skipTest("eigendecomposition not implemented")

        A = torch.tensor([[6.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
        eigenvalues, eigenvectors = func(A, descending=True)

        # Check eigenvalue equation for each eigenvector
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lhs = A @ v
            rhs = eigenvalues[i] * v
            self.assertTrue(torch.allclose(lhs, rhs, atol=1e-4),
                           f"Eigenvalue equation Av=λv failed for eigenpair {i}")

class TestPart2FourierTransformOperator(unittest.TestCase):
    """Part 2.4: Tests for fourier_transform_operator function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_fourier_transform_operator_shape(self):
        func = get_part1_func('fourier_transform_operator')
        if func is None:
            self.skipTest("fourier_transform_operator not implemented")

        N = 4
        torch.manual_seed(456)
        operator = torch.randn(N, N)
        operator = (operator + operator.T) / 2

        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        transformed = func(operator, basis)
        self.assertEqual(transformed.shape, (N, N))

    def test_fourier_transform_operator_formula(self):
        func = get_part1_func('fourier_transform_operator')
        if func is None:
            self.skipTest("fourier_transform_operator not implemented")

        N = 6
        torch.manual_seed(789)
        operator = torch.randn(N, N)
        operator = (operator + operator.T) / 2

        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        transformed = func(operator, basis)
        expected = torch.tensor([[-0.2592,  1.1666,  0.3643, -0.4518, -0.2542, -0.7292],
                                [ 1.1666,  0.1754,  0.1303, -0.9125, -0.6118,  1.6759],
                                [ 0.3643,  0.1303, -2.0134,  0.4625,  0.5942,  0.6608],
                                [-0.4518, -0.9125,  0.4625,  0.7963, -0.8041, -0.0808],
                                [-0.2542, -0.6118,  0.5942, -0.8041, -0.3173, -0.3770],
                                [-0.7292,  1.6759,  0.6608, -0.0808, -0.3770,  0.2918]])
        self.assertTrue(torch.allclose(transformed, expected, atol=1e-4))

class TestPart2FourierTransform(unittest.TestCase):
    """Part 2.4: Tests for fourier_transform function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_fourier_transform_shape(self):
        func = get_part1_func('fourier_transform')
        if func is None:
            self.skipTest("fourier_transform not implemented")

        N = 8
        torch.manual_seed(111)
        img = torch.randn(N)
        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        fourier_img = func(img, basis)
        self.assertEqual(fourier_img.shape, img.shape)

    def test_fourier_transform_formula(self):
        func = get_part1_func('fourier_transform')
        if func is None:
            self.skipTest("fourier_transform not implemented")

        N = 10
        torch.manual_seed(222)
        img = torch.randn(N)
        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        fourier_img = func(img, basis)
        expected = torch.tensor([ 0.5464,  1.7740,  2.3506, -1.9930,  0.9759, -0.3871, -0.0506,  0.3182,
        -1.5878,  0.1907])
        self.assertTrue(torch.allclose(fourier_img, expected, atol=1e-4))

class TestPart2InverseFourierTransform(unittest.TestCase):
    """Part 2.4: Tests for inv_fourier_transform function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_inv_fourier_transform_roundtrip(self):
        func_fwd = get_part1_func('fourier_transform')
        func_inv = get_part1_func('inv_fourier_transform')
        if func_fwd is None or func_inv is None:
            self.skipTest("fourier_transform or inv_fourier_transform not implemented")

        N = 12
        torch.manual_seed(333)
        img = torch.randn(N)
        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        fourier_img = func_fwd(img, basis)
        reconstructed = func_inv(fourier_img, basis)

        self.assertTrue(torch.allclose(reconstructed, img, atol=1e-4))

    def test_inv_fourier_transform_formula(self):
        func = get_part1_func('inv_fourier_transform')
        if func is None:
            self.skipTest("inv_fourier_transform not implemented")

        N = 8
        torch.manual_seed(444)
        fourier_img = torch.randn(N)
        basis = torch.randn(N, N)
        basis, _ = torch.linalg.qr(basis)

        img = func(fourier_img, basis)
        expected = torch.tensor([ 0.9394,  1.5180, -1.1375,  0.4073, -0.3359,  0.5491,  1.0024, -1.1601])
        self.assertTrue(torch.allclose(img, expected, atol=1e-4))

class TestPart31DGaussianFilter(unittest.TestCase):
    """Part 3.1: Tests for _gaussian_filter_1d function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_gaussian_1d_zeroth_order(self):
        func = get_part1_func('_gaussian_filter_1d')
        if func is None:
            self.skipTest("_gaussian_filter_1d not implemented")

        sigma = 1.5
        kernel = func(sigma, order=0)

        # Zeroth order should sum to approximately 1 (conserves mass)
        self.assertTrue(kernel.sum() > 0.99 and kernel.sum() < 1.01,
                       f"Kernel sum should be ~1, got {kernel.sum()}")

        # Should be symmetric
        self.assertTrue(torch.allclose(kernel, kernel.flip(0), atol=1e-6),
                       "Gaussian kernel should be symmetric")

        # Maximum should be at center
        center_idx = len(kernel) // 2
        self.assertEqual(kernel.argmax().item(), center_idx,
                        "Gaussian maximum should be at center")

    def test_gaussian_1d_first_order_mean(self):
        func = get_part1_func('_gaussian_filter_1d')
        if func is None:
            self.skipTest("_gaussian_filter_1d not implemented")

        sigma = 2.0
        kernel = func(sigma, order=1)

        # First order derivative mean should be 0
        self.assertTrue(abs(kernel.mean()) < 1e-6,
                       f"First derivative mean should be 0, got {kernel.mean()}")

        # Polynomial response condition: G'(1) * 1 = 0
        response_const = kernel.sum()
        self.assertTrue(abs(response_const) < 1e-6,
                       f"Response to constant function should be 0, got {response_const}")

        # Polynomial response condition: G'(1) * x = -1
        radius = (len(kernel) - 1) // 2
        positions = torch.arange(-radius, radius + 1, dtype=kernel.dtype)
        response_linear = (kernel * positions).sum()
        self.assertTrue(torch.isclose(response_linear, f32(-1.0), atol=1e-5),
                       f"Response to linear function should be -1, got {response_linear}")

    def test_gaussian_1d_second_order_mean(self):
        func = get_part1_func('_gaussian_filter_1d')
        if func is None:
            self.skipTest("_gaussian_filter_1d not implemented")

        sigma = 1.0
        kernel = func(sigma, order=2)

        # Second order derivative mean should be 0
        self.assertTrue(abs(kernel.mean()) < 1e-6,
                       f"Second derivative mean should be 0, got {kernel.mean()}")

        # Polynomial response conditions
        radius = (len(kernel) - 1) // 2
        positions = torch.arange(-radius, radius + 1, dtype=kernel.dtype)

        # G''(2) * 1 = 0
        response_const = kernel.sum()
        self.assertTrue(abs(response_const) < 1e-6,
                       f"Response to constant should be 0, got {response_const}")

        # G''(2) * x = 0
        response_linear = (kernel * positions).sum()
        self.assertTrue(abs(response_linear) < 1e-6,
                       f"Response to linear function should be 0, got {response_linear}")

        # G''(2) * x^2 = 2
        response_quadratic = (kernel * positions.pow(2)).sum()
        self.assertTrue(torch.isclose(response_quadratic, f32(2.0), atol=1e-5),
                       f"Response to quadratic should be 2, got {response_quadratic}")

class TestPart32DGaussian(unittest.TestCase):
    """Part 3.2: Tests for 2D Gaussian derivative functions."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_zeroth_order_shape(self):
        func = get_part1_func('zeroth_order')
        if func is None:
            self.skipTest("zeroth_order not implemented")

        img = torch.randn(1, 1, 10, 10)
        result = func(img, sigma=1.0)
        self.assertEqual(result.shape, img.shape)

        # Smoothing should reduce variance (remove high frequencies)
        self.assertTrue(result.var() <= img.var(),
                       "Smoothing should reduce variance")

        # Constant image should remain constant
        const_img = torch.ones(1, 1, 10, 10) * 5.0
        const_result = func(const_img, sigma=1.0)
        self.assertTrue(torch.allclose(const_result, const_img, atol=1e-5),
                       "Constant image should remain constant after smoothing")
    
    def test_first_order_x_gradient(self):
        func = get_part1_func('first_order_x')
        if func is None:
            self.skipTest("first_order_x not implemented")

        x = torch.linspace(0, 1, 10)
        img = x.repeat(10, 1)
        img = img.unsqueeze(0).unsqueeze(0)
        dx = func(img, sigma=0.5)

        # Compare correlation, to center
        ref = img[..., :, 2:] - img[..., :, :-2]
        ref = ref / 2.0
        dx_c = dx[..., :, 1:-1]
        corr = (dx_c * ref).mean()
        self.assertTrue(corr.abs() > 1e-3)

    def test_first_order_y_gradient(self):
        func = get_part1_func('first_order_y')
        if func is None:
            self.skipTest("first_order_y not implemented")

        y = torch.linspace(0, 1, 10)
        img = y.repeat(10, 1).T
        img = img.unsqueeze(0).unsqueeze(0)
        dy = func(img, sigma=0.5)

        # Compare correlation, to center
        ref = img[..., 2:, :] - img[..., :-2, :]
        ref = ref / 2.0
        dy_c = dy[..., 1:-1, :]
        corr = (dy_c * ref).mean()
        self.assertTrue(corr.abs() > 1e-3)

    def test_second_order_linear_function(self):
        func = get_part1_func('second_order_yy')
        if func is None:
            self.skipTest("second_order_yy not implemented")

        # Linear function should have zero second derivative
        img = torch.zeros(1, 1, 31, 31, dtype=torch.float64)
        img[..., 7:24] += torch.arange(1, 18)
        img[..., 24:] += 17

        dyy = func(img, sigma=0.5)
        # Should be very close to zero
        self.assertTrue(dyy.abs().max() < 1e-10,
                       f"Second derivative of linear function should be ~0, got {dyy.abs().max()}")

class TestPart3LOG(unittest.TestCase):
    """Part 3.3: Tests for log (Laplacian of Gaussian) function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_log_shape(self):
        func = get_part1_func('log')
        if func is None:
            self.skipTest("log not implemented")

        img = torch.randn(1, 1, 20, 20)
        result = func(img, sigma=1.0)
        self.assertEqual(result.shape, img.shape)

    def test_log_zero_crossings(self):
        func = get_part1_func('log')
        if func is None:
            self.skipTest("log not implemented")

        img = torch.randn(1, 1, 30, 30)
        result = func(img, sigma=1.5)

        # LoG should have both positive and negative values
        self.assertTrue(result.min() < 0 and result.max() > 0,
                       "LoG should have zero crossings (both positive and negative values)")

    def test_log_is_sum_of_second_derivatives(self):
        """Verify LoG = d²/dx² + d²/dy²"""
        func_log = get_part1_func('log')
        func_xx = get_part1_func('second_order_xx')
        func_yy = get_part1_func('second_order_yy')

        if func_log is None or func_xx is None or func_yy is None:
            self.skipTest("Required functions not implemented")

        torch.manual_seed(42)
        img = torch.randn(1, 1, 25, 25)
        sigma = 1.2

        log_result = func_log(img, sigma)
        manual_log = func_xx(img, sigma) + func_yy(img, sigma)

        self.assertTrue(torch.allclose(log_result, manual_log, atol=1e-5),
                       "LoG should equal d²/dx² + d²/dy²")

class TestPart3OrientedFilter(unittest.TestCase):
    """Part 3.4: Tests for oriented_filter function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_oriented_filter_shape(self):
        func = get_part1_func('oriented_filter')
        if func is None:
            self.skipTest("oriented_filter not implemented")

        theta = torch.pi / 4
        sigma = 1.0
        filt = func(theta, sigma)

        # Should be square
        self.assertEqual(filt.shape[0], filt.shape[1])

    def test_oriented_filter_nonzero(self):
        func = get_part1_func('oriented_filter')
        if func is None:
            self.skipTest("oriented_filter not implemented")

        theta = torch.pi / 3
        sigma = 2.0
        filt = func(theta, sigma)

        # Should not be all zeros
        self.assertTrue(filt.abs().max() > 0)

    def test_oriented_filter_angles(self):
        """Verify that 0° and 90° filters are orthogonal basis."""
        func = get_part1_func('oriented_filter')
        if func is None:
            self.skipTest("oriented_filter not implemented")

        sigma = 1.5
        filter_0 = func(0.0, sigma)  # Horizontal gradient
        filter_90 = func(torch.pi / 2, sigma)  # Vertical gradient
        filter_45 = func(torch.pi / 4, sigma)

        # 0° and 90° should be different
        self.assertFalse(torch.allclose(filter_0, filter_90, atol=1e-3),
                        "0° and 90° filters should be different")

        # 45° filter should be somewhere between 0° and 90°
        # Verify that filter values at different angles produce different results
        self.assertFalse(torch.allclose(filter_45, filter_0, atol=1e-3),
                        "45° filter should differ from 0° filter")
        self.assertFalse(torch.allclose(filter_45, filter_90, atol=1e-3),
                        "45° filter should differ from 90° filter")

class TestPart3Convolution(unittest.TestCase):
    """Part 3.4: Tests for conv function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_conv_shape_preservation(self):
        func = get_part1_func('conv')
        if func is None:
            self.skipTest("conv not implemented")

        img = torch.randn(1, 1, 15, 15)
        kernel = torch.randn(5, 5)
        result = func(img, kernel)

        # Shape should be preserved with proper padding
        self.assertEqual(result.shape, img.shape)

    def test_conv_identity_kernel(self):
        func = get_part1_func('conv')
        if func is None:
            self.skipTest("conv not implemented")

        img = torch.randn(1, 1, 10, 10)
        # Create identity kernel (delta function)
        kernel = torch.zeros(3, 3)
        kernel[1, 1] = 1.0

        result = func(img, kernel)

        # Identity kernel should not change image
        self.assertTrue(torch.allclose(result, img, atol=1e-5))

    def test_conv_box_blur(self):
        func = get_part1_func('conv')
        if func is None:
            self.skipTest("conv not implemented")

        # Create simple test image
        img = torch.zeros(1, 1, 5, 5)
        img[0, 0, 2, 2] = 9.0  # Single bright pixel in center

        # Box blur kernel
        kernel = torch.ones(3, 3) / 9.0

        result = func(img, kernel)

        # Center pixel should be 1.0, neighbors should be non-zero
        self.assertTrue(torch.isclose(result[0, 0, 2, 2], f32(1.0), atol=1e-5),
                       "Box blur should average the 3x3 neighborhood")

class TestPart3Steering(unittest.TestCase):
    """Part 3.4: Tests for steerable filters."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_steer_the_filter_shape(self):
        func = get_part1_func('steer_the_filter')
        if func is None:
            self.skipTest("steer_the_filter not implemented")

        img = torch.randn(1, 1, 20, 20)
        theta = torch.pi / 4
        sigma = 1.0
        result = func(img, theta, sigma)

        self.assertEqual(result.shape, img.shape)

    def test_steer_the_images_shape(self):
        func = get_part1_func('steer_the_images')
        if func is None:
            self.skipTest("steer_the_images not implemented")

        img = torch.randn(1, 1, 20, 20)
        theta = torch.pi / 4
        sigma = 1.0
        result = func(img, theta, sigma)

        self.assertEqual(result.shape, img.shape)

    def test_steer_equivalence(self):
        """Test that steer_the_filter and steer_the_images produce the same result."""
        func1 = get_part1_func('steer_the_filter')
        func2 = get_part1_func('steer_the_images')
        if func1 is None or func2 is None:
            self.skipTest("steer_the_filter or steer_the_images not implemented")

        torch.manual_seed(555)
        img = torch.randn(1, 1, 25, 25)
        theta = torch.pi / 3
        sigma = 1.5

        result1 = func1(img, theta, sigma)
        result2 = func2(img, theta, sigma)

        self.assertTrue(torch.allclose(result1, result2, atol=1e-4),
                       "Steered filter and steered images should produce the same result")

class TestPart3OrientationMeasurement(unittest.TestCase):
    """Part 3.5: Tests for measure_orientation function."""

    @classmethod
    def setUpClass(cls):
        load_functions()

    def test_measure_orientation_shape(self):
        func = get_part1_func('measure_orientation')
        if func is None:
            self.skipTest("measure_orientation not implemented")

        img = torch.randn(1, 1, 30, 30)
        sigma = 1.0
        orientations = func(img, sigma)

        # Shape should be preserved
        self.assertEqual(orientations.shape, img.shape)

    def test_measure_orientation_bounded(self):
        func = get_part1_func('measure_orientation')
        if func is None:
            self.skipTest("measure_orientation not implemented")

        img = torch.randn(1, 1, 40, 40)
        sigma = 1.5
        orientations = func(img, sigma)

        # Orientation magnitudes should be bounded by pi
        self.assertTrue(orientations.abs().max() <= torch.pi + 0.1,
                       f"Orientation magnitudes should be bounded, got max {orientations.abs().max()}")



def create_suite():
    """Create a test suite for all parts."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if HAS_TORCH:
        print("-" * 70)
        print("TEST CASES")
        print("-" * 70)
        suite.addTests(loader.loadTestsFromTestCase(TestPart1Identity))
        suite.addTests(loader.loadTestsFromTestCase(TestPart1CoeffsToSine))
        suite.addTests(loader.loadTestsFromTestCase(TestPart1AngleToCoeffs))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2ShiftOperator))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2MatrixFromConv))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2ImageOpFromMatrix))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2EigenDecomposition))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2FourierTransformOperator))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2FourierTransform))
        suite.addTests(loader.loadTestsFromTestCase(TestPart2InverseFourierTransform))
        suite.addTests(loader.loadTestsFromTestCase(TestPart31DGaussianFilter))
        suite.addTests(loader.loadTestsFromTestCase(TestPart32DGaussian))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3LOG))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3OrientedFilter))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3Convolution))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3Steering))
        suite.addTests(loader.loadTestsFromTestCase(TestPart3OrientationMeasurement))
    else:
        print("Skipped (torch not available)")

    return suite

# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run tests for specified parts."""
    print("=" * 70)
    print("Test Suite - All Parts")
    print("6.8300 - Advances in Computer Vision")
    print("=" * 70)
    print("\nNote: These tests use DIFFERENT values than Gradescope.")
    print("Passing here means your implementation is likely correct!\n")

    suite = create_suite()

    print()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"TOTAL: {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors, "
          f"{len(result.skipped)} skipped")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n All tests passed!")
    else:
        print("\n Some tests failed. Review the errors above.")

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(
        description='Combined test suite for all parts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python test_all_local.py
"""
    )
    success = run_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
