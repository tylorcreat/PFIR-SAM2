from __future__ import annotations

import unittest

import numpy as np

from pfir_sam2.reconstruction import (
    ReconstructionConfig,
    RescueConfig,
    reconstruct_instances,
    rescue_small_objects,
)


class ReconstructionTests(unittest.TestCase):
    def test_watershed_and_rescue_return_sequential_labels(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        foreground = (
            ((xx - 20) ** 2 + (yy - 32) ** 2 < 10**2)
            | ((xx - 44) ** 2 + (yy - 32) ** 2 < 10**2)
        ).astype(np.float32)
        boundary = np.zeros_like(foreground)
        center = np.maximum(
            np.exp(-((xx - 20) ** 2 + (yy - 32) ** 2) / 8.0),
            np.exp(-((xx - 44) ** 2 + (yy - 32) ** 2) / 8.0),
        ).astype(np.float32)
        reconstructed = reconstruct_instances(
            foreground,
            boundary,
            center,
            ReconstructionConfig(minimum_area=5, watershed_min_distance=5),
        )
        raw = reconstructed.copy()
        raw[2:5, 2:5] = int(raw.max()) + 1
        final, events = rescue_small_objects(
            raw,
            reconstructed,
            foreground_probability=np.maximum(foreground, (raw > 0).astype(float)),
            config=RescueConfig(maximum_small_area=20),
        )
        self.assertEqual(set(np.unique(final)), set(range(int(final.max()) + 1)))
        self.assertTrue(any(event.get("action") == "rescued" for event in events))


if __name__ == "__main__":
    unittest.main()
