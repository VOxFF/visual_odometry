import numpy as np
import h5py


class LandmarkH5Exporter:
    """
    Records new landmarks born each frame and writes an HDF5 file.

    Drop-in replacement for LandmarkUsdExporter — same record_new / write API,
    no pxr dependency so it runs cleanly inside obj_env.

    HDF5 layout
    -----------
    /                       root
      attrs:
        fps          float  frames-per-second (for pointsview playback speed)
        total        int    total landmark count across all frames
      /frames/
        /<frame:06d>  dataset  (N, 3) float32  new landmarks born at that frame
    """

    def __init__(self):
        self._frames: list[tuple[int, np.ndarray]] = []

    def record_new(self, frame_idx: int, xyzs: list) -> None:
        if xyzs:
            self._frames.append((frame_idx, np.array(xyzs, dtype=np.float32)))

    def write(self, output_path: str, fps: float = 30.0) -> None:
        if not self._frames:
            print("LandmarkH5Exporter: nothing recorded, skipping export.")
            return

        total = sum(len(x) for _, x in self._frames)

        with h5py.File(output_path, 'w') as f:
            f.attrs['fps']   = fps
            f.attrs['total'] = total

            grp = f.create_group('frames')
            for frame_idx, xyzs in self._frames:
                grp.create_dataset(
                    f'{frame_idx:06d}',
                    data=xyzs,
                    compression='gzip',
                    compression_opts=4,
                )

        print(f"Landmarks HDF5 → {output_path}  "
              f"({len(self._frames)} frames, {total:,} total landmarks)")
