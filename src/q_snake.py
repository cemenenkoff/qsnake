import json
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from agent import train_dqn
from environment import Snake
from gif_builder import GifBuilder
from plotting import plot_history


def parse_args() -> Namespace:
    """Define CLI options.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = ArgumentParser(
        prog="qSnake", description="Train a neural network to play Snake."
    )
    parser.add_argument(
        dest="config",
        nargs="?",
        default="configs/config.json",
        help="path to the configuration file",
    )
    return parser.parse_args()


def get_config(config_path: str) -> dict:
    """Convert a JSON config file into a dictionary.

    Args:
        config_path (str): Path of the JSON config file.

    Returns:
        dict: The JSON config file loaded as a dictionary.
    """
    with open(config_path) as json_file:
        return json.load(json_file)


def check_config(config: dict) -> dict:
    """Check a JSON config for required keys and value types.

    Args:
        config (dict): The raw config file.

    Raises:
        ValueError: Raised if a config value is the wrong type.
        KeyError: Raised if a required key is missing from the config file.

    Returns:
        dict: The validated config file.
    """
    req = {
        "human": bool,
        "name": str,
        "save_for_gif": bool,
        "make_gif": bool,
        "params": dict,
    }
    req_params = {
        "epsilon": float,
        "gamma": float,
        "batch_size": int,
        "epsilon_min": float,
        "epsilon_decay": float,
        "learning_rate": float,
        "layer_sizes": list,
        "num_episodes": int,
        "max_steps": int,
        "state_definition_type": str,
    }
    iterables = [(req, config), (req_params, config["params"])]
    for codex, subdict in iterables:
        for k, v in codex.items():
            try:
                if not isinstance(subdict[k], v):
                    raise ValueError
            except KeyError:
                print(f"ERROR: missing required key {k}")
                sys.exit(1)
            except ValueError:
                print(f"ERROR: {k} is currently {type(subdict[k])} and should be {v}")
                sys.exit(1)
    return config


def main():
    args = parse_args()
    config = check_config(get_config(args.config))
    params = config["params"]  # The main parameters for the agent.
    runs_dir = Path.cwd() / "runs"

    # Name a folder to store the output of this run.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    num_ep = params["num_episodes"]
    bat_sz = params["batch_size"]
    state_def = params["state_definition_type"]
    params_str = f"{state_def}-{num_ep}ep-{bat_sz}batch"
    if config["name"]:
        params_str = f"{config['name']}-{params_str}"
    instance_folder = f"{ts}-{params_str}"

    if config["save_for_gif"]:
        # Create folders to store the eps files for gif creation.
        eps_dir = runs_dir / instance_folder / "gif-build" / "eps"
        config["eps_dir"] = eps_dir
        eps_dir.mkdir(exist_ok=True, parents=True)

    env = Snake(config)
    # If we are just playing the game, no folders should be created.
    if config["human"]:
        while True:
            env.run_game()
    elif not config["human"]:
        history = train_dqn(env, params)
        # If an agent plays, create a folder to store our learning curve graph.
        instance_dir = runs_dir / instance_folder
        instance_dir.mkdir(exist_ok=True, parents=True)
        plot_name = f"learning-curve-{params_str}.png"
        plot_history(history, outpath=instance_dir / plot_name, params=params)
    if config["make_gif"]:
        png_dir = eps_dir.parent / "png"
        gif_name = f"training-montage-{params_str}.gif"
        bob_the_builder = GifBuilder(config["eps_dir"], png_dir)
        bob_the_builder.convert_eps_files()
        bob_the_builder.make_gif(outpath=runs_dir / instance_folder / gif_name)


if __name__ == "__main__":
    main()
