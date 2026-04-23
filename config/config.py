import yaml
from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    dataset_path: str
    yaml_file: str
    output_path: str
    stereo_checkpoint: str
    flow_checkpoint: str

    # Run control
    compute_trajectory: bool = True
    render_images: bool = True
    compose_movie: bool = True
    limit: int = 0              # 0 = no limit

    # Depth filtering
    min_depth: float = 0.0
    max_depth: float = 15.0

    # Keypoints
    max_keypoints: int = 320
    dz_threshold: float = 1.0

    # RAFT iterations
    raft_iters: int = 16
    raft_disparity_warmstart: bool = False
    raft_optflow_warmstart: bool = False
    subpixel_keypoints: bool = False

    # Plot view
    elevation: float = 90.0
    azimuth: float = 0.0
    zoom_distance: float = 5.0

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        paths = data.get('paths', {})
        run = data.get('run', {})
        params = data.get('parameters', {})
        plot = data.get('plot', {})

        return cls(
            dataset_path=paths['dataset_path'],
            yaml_file=paths['yaml_file'],
            output_path=paths['output_path'],
            stereo_checkpoint=paths['stereo_checkpoint'],
            flow_checkpoint=paths['flow_checkpoint'],

            compute_trajectory=run.get('compute_trajectory', True),
            render_images=run.get('render_images', True),
            compose_movie=run.get('compose_movie', True),
            limit=run.get('limit', 0),

            min_depth=params.get('min_depth', 0.0),
            max_depth=params.get('max_depth', 15.0),
            max_keypoints=params.get('max_keypoints', 320),
            dz_threshold=params.get('dz_threshold', 1.0),
            raft_iters=params.get('raft_iters', 16),
            raft_disparity_warmstart=params.get('raft_disparity_warmstart', False),
            raft_optflow_warmstart=params.get('raft_optflow_warmstart', False),
            subpixel_keypoints=params.get('subpixel_keypoints', False),

            elevation=plot.get('elevation', 90.0),
            azimuth=plot.get('azimuth', 0.0),
            zoom_distance=plot.get('zoom_distance', 5.0),
        )
