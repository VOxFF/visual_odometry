import argparse
from config.config import Config


def main():
    parser = argparse.ArgumentParser(description="Stereo Visual Odometry")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if config.pipeline == 'landmark':
        from pipeline.landmark_pipeline import LandmarkPipeline
        pipeline = LandmarkPipeline(config)
    else:
        from pipeline.flow_pipeline import FlowPipeline
        pipeline = FlowPipeline(config)

    pipeline.run()


if __name__ == "__main__":
    main()
