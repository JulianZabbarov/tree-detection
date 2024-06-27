import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from argparse import ArgumentParser

import tomllib
from dacite import from_dict

from src.configs.config_definition import PipelineConfig

def load_config():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config",
        action="store",
        help="relative path to config file for experiment configurations",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = from_dict(data_class=PipelineConfig, data=tomllib.load(f))
    return config