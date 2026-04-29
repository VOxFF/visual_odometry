import argparse
from config.config import Config
from pipeline.flow_pipeline import FlowPipeline


def main():
    parser = argparse.ArgumentParser(description="Stereo Visual Odometry")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    pipeline = FlowPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
