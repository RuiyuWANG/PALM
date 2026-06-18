import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Union

from easydict import EasyDict  # Assuming you are using EasyDict
from rich.console import Console
from rich.syntax import Syntax


def print_config_to_terminal(config: Union[Dict, EasyDict]):
    """
    Print the configuration to the terminal with rich formatting.

    Parameters:
    - config: Union[Dict, EasyDict], the configuration to print.
    """
    # If config is an EasyDict, convert it to a standard dict.
    if isinstance(config, EasyDict):
        config = namespace_to_dict(config)  # Ensure namespace_to_dict is defined/imported

    # Convert the dictionary to a formatted JSON string.
    json_str = json.dumps(config, indent=4, sort_keys=True)

    # Set up a Rich console and syntax highlighter for JSON.
    console = Console()
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)

    console.print(syntax)


def namespace_to_dict(namespace: EasyDict) -> Dict[str, Any]:
    """
    Convert a namespace to a dictionary.

    Parameters:
    - namespace: EasyDict, the namespace to convert.

    Returns:
    - dict, the converted dictionary.
    """
    return dict(namespace)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config_to_namespace(config_filename: str, full_path: bool = False) -> object:
    """
    Load the configuration file and convert it to a namespace.

    Parameters:
    - config_filename: str, the name of the configuration file.

    Returns:
    - object, the loaded configuration as a namespace.
    """

    config = load_config(config_filename, full_path)
    return EasyDict(config)


def load_config(config_filename: str, full_path=False) -> dict:
    """
    Load the configuration file.

    Parameters:
    - config_filename: str, the name of the configuration file.

    Returns:
    - dict, the loaded configuration.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    if "." not in config_filename:
        config_filename += ".json"

    if full_path:
        config_path = Path(config_filename)
    else:
        config_path = config_dir / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    return config


def get_config_file_path(config_filename: str) -> str:
    """
    Get the full path of the configuration file.

    Parameters:
    - config_filename: str, the name of the configuration file.

    Returns:
    - str, the full path of the configuration file.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    if "." not in config_filename:
        config_filename += ".json"

    config_path = config_dir / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    return config_path


def update_json_file(file_path, new_data):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)