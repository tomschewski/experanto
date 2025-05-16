from contextlib import contextmanager
import shutil
import yaml
from pathlib import Path
import numpy as np

SCREEN_ROOT = Path("tests/screen_data")


@contextmanager
def create_screen_data(n_frames=10, frame_shape=(64, 64), t_end=10.0, fps=10.0):
    try:
        data_dir = SCREEN_ROOT / "data"
        meta_dir = SCREEN_ROOT / "meta"

        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.linspace(0.0, t_end, int(t_end * fps) + 1)
        np.save(SCREEN_ROOT / "timestamps.npy", timestamps)

        for i, t in enumerate(timestamps):
            frame = np.random.rand(*frame_shape).astype(np.float32)

            np.save(data_dir / f"{i:05d}.npy", frame)
            with open(meta_dir / f"{i:05d}.yml", "w") as f:
                yaml.dump({
                    "first_frame_idx": i,
                    "image_size": list(frame_shape),
                    "modality": "image",
                    "tier": "train"
                }, f)

        with open(SCREEN_ROOT / "meta.yml", "w") as f:
            yaml.dump({
                "modality": "screen",
                "frame_rate": fps,
            }, f)

        yield timestamps

    finally:
        shutil.rmtree(SCREEN_ROOT)
