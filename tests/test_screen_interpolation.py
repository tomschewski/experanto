import numpy as np
import pytest
from create_screen_data import create_screen_data

from experanto.interpolators import Interpolator, ScreenInterpolator


@pytest.mark.parametrize("fps", [5.0, 30.0])
def test_nearest_neighbor_screen_interpolation(fps):
    with create_screen_data(
        n_frames=20, frame_shape=(32, 32), t_end=4.0, fps=fps
    ) as timestamps:
        screen_interp = Interpolator.create("tests/screen_data")
        assert isinstance(
            screen_interp, ScreenInterpolator
        ), "Expected ScreenInterpolator"

        delta_t = 1.0 / fps
        idx = slice(1, 11)
        times = timestamps[idx] + 0.4 * delta_t  # some offset inside the frame period

        # Load all frames from the data folder for comparison
        data_dir = screen_interp.root_folder / "data"
        frames = [np.load(data_dir / f"{i:05d}.npy") for i in range(len(timestamps))]
        frames = np.stack(frames)

        # Nearest neighbor frame indices for given times (round to nearest index)
        expected_indices = np.round((times - timestamps[0]) * fps).astype(int)
        expected_frames = frames[expected_indices]

        interp, valid = screen_interp.interpolate(times=times)

        assert np.all(valid), "All interpolated frames should be valid"
        assert np.allclose(
            interp, expected_frames, atol=1e-5
        ), "Nearest neighbor interpolation mismatch"
