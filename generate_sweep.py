
from typing import List
import os
import yaml
import hiyapyco
import argparse
import wandb

def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate a wandb sweep configuration.')
    parser.add_argument('wandb_group', type=str, 
                        help='Weights & Biases group name.')
    parser.add_argument('yaml_files', type=str, nargs='+', 
                        help='List of YAML configuration files.')
    parser.add_argument('--analyze', action='store_true', 
                        help='If set, use analyze.py instead of train.py and set with_wandb to False.')
    parser.add_argument('--save', action='store_true', 
                        help='If set, save the sweep configuration to a YAML file instead of creating a sweep.')
    args = parser.parse_args()

    # Load and merge the YAML files
    # configs: List[dict] = [yaml.safe_load(open(f"resources/sweeps/{filename}.yaml")) for filename in args.yaml_files]
    # merged_config: dict = hiyapyco.load(configs, method=hiyapyco.METHOD_MERGE)
    # Load and merge the YAML files
    configs: List[str] = [f"resources/sweeps/{filename}.yaml" for filename in args.yaml_files]
    merged_config: dict = hiyapyco.load(*configs, method=hiyapyco.METHOD_MERGE)


    # Construct the job type variable
    wandb_job_type: str = "-".join([os.path.splitext(os.path.basename(filename))[0] for filename in args.yaml_files])

    # Define the experiment variable
    experiment: str = "_".join(
        f"{param_name}-{{{param_name}}}"
        for param_name, param_info in merged_config["parameters"].items()
        if "values" in param_info and len(param_info["values"]) > 1
    )
    # Update the merged config
    merged_config["parameters"]["wandb_job_type"] = {"value": wandb_job_type}
    merged_config["parameters"]["experiment"] = {"value": experiment}
    merged_config["parameters"]["train_dir"] = {"value": f"train_dir/{args.wandb_group}/{wandb_job_type}"}

    # If the analyze flag is set, use analyze.py and set with_wandb to False
    if args.analyze:
        merged_config["program"] = "analyze.py"
        merged_config["parameters"]["with_wandb"] = {"value": False}

    # Convert the merged config to a wandb sweep configuration
    sweep_config: dict = {
        "name": f"{args.wandb_group}-{wandb_job_type}",
        "description": merged_config["description"],
        "method": merged_config["method"],
        "parameters": {
            name: dict(info) for name, info in merged_config["parameters"].items() if "value" in info
        },
    }
    if "program" in merged_config:
        sweep_config["program"] = merged_config["program"]

    if args.save:
        # If the save flag is set, save the sweep configuration to a YAML file
        with open("sweep_config.yaml", "w") as f:
            yaml.dump(sweep_config, f)
    else:
        # Otherwise, create the wandb sweep
        wandb.sweep(sweep_config, project=merged_config["project"])

if __name__ == "__main__":
    main()

