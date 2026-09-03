import numpy as np

from direction4.isp_noise import _texture_mask, attack


def test_isp_noise_is_deterministic_and_budgeted():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    first = attack(image, {}, None, seed=7, texture_mask=True)
    second = attack(image, {}, None, seed=7, texture_mask=True)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first.astype(int) - image.astype(int))) <= 8


def test_texture_mask_allocates_more_noise_to_detail():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    mask = _texture_mask(image)
    assert np.isfinite(mask).all()
    assert mask.min() >= 0.25 and mask.max() <= 1.0
    assert mask[16, 15] > mask[2, 2]
