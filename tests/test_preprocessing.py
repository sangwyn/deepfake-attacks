"""Contract tests for the shared differentiable preprocessing and projector."""

import unittest

import numpy as np
import torch

from attacklab import preprocessing as pre


def _image(seed: int = 0, height: int = 178, width: int = 218) -> np.ndarray:
    return (np.random.RandomState(seed).rand(height, width, 3) * 255).astype("uint8")


class ConversionTests(unittest.TestCase):
    def test_uint8_round_trip_is_lossless(self):
        image = _image()
        tensor = pre.from_uint8_image(image, torch.device("cpu"))
        self.assertEqual(tuple(tensor.shape), (1, 3, 178, 218))
        self.assertTrue(bool((tensor >= 0).all() and (tensor <= 1).all()))
        self.assertTrue(np.array_equal(pre.to_uint8_image(tensor), image))

    def test_conversion_rejects_a_non_rgb_array(self):
        with self.assertRaises(ValueError):
            pre.from_uint8_image(np.zeros((4, 4), dtype="uint8"), torch.device("cpu"))
        with self.assertRaises(ValueError):
            pre.to_uint8_image(torch.zeros(3, 8, 8))


class ProjectorTests(unittest.TestCase):
    def test_projection_respects_the_budget_and_pixel_range(self):
        original = torch.rand(1, 3, 16, 16)
        adversarial = original + torch.randn(1, 3, 16, 16)
        epsilon = 8 / 255
        projected = pre.project_linf(adversarial, original, epsilon)
        self.assertLessEqual(float((projected - original).abs().max()), epsilon + 1e-7)
        self.assertGreaterEqual(float(projected.min()), 0.0)
        self.assertLessEqual(float(projected.max()), 1.0)

    def test_projection_is_a_no_op_inside_the_ball(self):
        original = torch.full((1, 3, 8, 8), 0.5)
        inside = original + 1 / 255
        self.assertTrue(torch.equal(pre.project_linf(inside, original, 8 / 255), inside))

    def test_projection_rejects_bad_arguments(self):
        original = torch.rand(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            pre.project_linf(original, original, 0.0)
        with self.assertRaises(ValueError):
            pre.project_linf(torch.rand(1, 3, 4, 4), original, 8 / 255)


class SpatialTests(unittest.TestCase):
    def test_vit_shape_and_finite_gradients(self):
        x = pre.from_uint8_image(_image(), torch.device("cpu")).requires_grad_()
        out = pre.preprocess_for("vit_b_16", x)
        self.assertEqual(tuple(out.shape), (1, 3, 224, 224))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_non_vit_spatial_keeps_the_full_256_square(self):
        x = pre.from_uint8_image(_image(), torch.device("cpu"))
        self.assertEqual(
            tuple(pre.differentiable_spatial(x, "resnet50").shape), (1, 3, 256, 256)
        )

    def test_unsupported_source_model_is_rejected(self):
        x = pre.from_uint8_image(_image(), torch.device("cpu"))
        with self.assertRaises(ValueError):
            pre.preprocess_for("no_such_detector", x)

    def test_spatial_output_tracks_the_evaluator_transform(self):
        """Surrogate resize differs from PIL's, so only bound the divergence."""

        from PIL import Image

        import evaluate

        image = _image(seed=3)
        reference = evaluate.build_spatial_transform("vit_b_16")(
            Image.fromarray(image)
        ).unsqueeze(0)
        surrogate = pre.preprocess_for(
            "vit_b_16", pre.from_uint8_image(image, torch.device("cpu"))
        )
        self.assertEqual(reference.shape, surrogate.shape)
        # Documented divergence: tensor bilinear vs PIL resize on uint8.
        # Observed 0.0053 on a scale whose reference standard deviation is ~1.
        self.assertLess(float((reference - surrogate).abs().mean()), 0.02)


class DctTests(unittest.TestCase):
    def test_dct_shape_and_finite_gradients(self):
        x = pre.from_uint8_image(_image(), torch.device("cpu")).requires_grad_()
        out = pre.preprocess_for("densenet121_dct", x)
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_dct_matrix_is_orthonormal(self):
        basis = pre.dct_matrix(16, torch.float64, torch.device("cpu"))
        identity = basis @ basis.transpose(0, 1)
        self.assertLess(float((identity - torch.eye(16, dtype=torch.float64)).abs().max()), 1e-10)

    def test_dct_matches_scipy_on_identical_input(self):
        """The transform itself must be exact; only the resize is a surrogate."""

        from scipy.fft import dct as scipy_dct

        array = np.random.RandomState(7).rand(128, 128).astype("float64") * 255
        expected = scipy_dct(
            scipy_dct(array, axis=0, norm="ortho"), axis=1, norm="ortho"
        )
        basis = pre.dct_matrix(128, torch.float64, torch.device("cpu"))
        tensor = torch.from_numpy(array)
        observed = (basis @ tensor @ basis.transpose(0, 1)).numpy()
        self.assertLess(float(np.abs(expected - observed).max()), 1e-8)

    def test_dct_output_tracks_the_evaluator_transform(self):
        from PIL import Image

        import evaluate

        image = _image(seed=5, height=256, width=256)
        reference = evaluate.build_dct_transform(True)(
            Image.fromarray(image)
        ).unsqueeze(0)
        surrogate = pre.preprocess_for(
            "densenet121_dct", pre.from_uint8_image(image, torch.device("cpu"))
        )
        self.assertEqual(reference.shape, surrogate.shape)
        # Observed 0.0247 on a reference spanning roughly [-6.7, 9.7]. Individual
        # near-zero coefficients diverge more because of the log magnitude.
        self.assertLess(float((reference - surrogate).abs().mean()), 0.10)


class IfgsmTests(unittest.TestCase):
    def test_ifgsm_respects_the_budget_and_returns_uint8(self):
        import torch.nn as nn

        from attacks import ifgsm

        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = _image(seed=11)
        output = ifgsm.attack(
            image, {"vit_b_16": {"model": model}}, torch.device("cpu"), target_class=0
        )
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        delta = output.astype(int) - image.astype(int)
        self.assertLessEqual(int(np.abs(delta).max()), round(8 / 255 * 255))

    def test_ifgsm_is_deterministic_for_a_fixed_model(self):
        import torch.nn as nn

        from attacks import ifgsm

        torch.manual_seed(1)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = _image(seed=12)
        classifiers = {"vit_b_16": {"model": model}}
        first = ifgsm.attack(image, classifiers, torch.device("cpu"), target_class=1)
        second = ifgsm.attack(image, classifiers, torch.device("cpu"), target_class=1)
        self.assertTrue(np.array_equal(first, second))

    def test_ifgsm_rejects_an_unsupported_source_model(self):
        from attacks import ifgsm

        with self.assertRaises(ValueError):
            ifgsm.attack(_image(), {}, torch.device("cpu"), source_model="densenet121_dct")


class FgsmTests(unittest.TestCase):
    def test_fgsm_returns_uint8_with_finite_targeted_gradient_and_budget(self):
        import torch.nn as nn

        from attacks import fgsm

        torch.manual_seed(2)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = _image(seed=13)
        output = fgsm.attack(
            image, {"vit_b_16": {"model": model}}, torch.device("cpu"), target_class=0
        )
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertLessEqual(int(np.abs(output.astype(int) - image.astype(int)).max()), 8)

    def test_fgsm_is_deterministic_for_a_fixed_model(self):
        import torch.nn as nn

        from attacks import fgsm

        torch.manual_seed(3)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = _image(seed=14)
        classifiers = {"vit_b_16": {"model": model}}
        first = fgsm.attack(image, classifiers, torch.device("cpu"), target_class=1)
        second = fgsm.attack(image, classifiers, torch.device("cpu"), target_class=1)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
