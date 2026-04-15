import argparse
from config.config import Config
from pipeline.pipeline import CameraTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="Stereo Visual Odometry")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    pipeline = CameraTrackingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
