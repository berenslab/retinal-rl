from typing import List
import os
import yaml
import hiyapyco
import argparse
import wandb
import random

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
    parser.add_argument('--trial', action='store_true', 
                        help='If set, select a random set of parameters and print a command.')
    args = parser.parse_args()

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

    # Convert the merged config to a wandb sweep configuration
    sweep_config: dict = {
        "name": f"{args.wandb_group}-{wandb_job_type}",
        "project": merged_config["project"],
        "description": merged_config["description"],
        "method": merged_config["method"],
        "program": "analyze.py" if args.analyze else "train.py",
        "parameters": {
            name: dict(info) for name, info in merged_config["parameters"].items()
        },
    }
    
    # Update the wandb group, and set the experiment and job type
    sweep_config["parameters"].update({
        "wandb_group": {"value": args.wandb_group},
        "wandb_job_type": {"value": wandb_job_type},
        "experiment": {"value": experiment},
        "train_dir": {"value": f"train_dir/{args.wandb_group}/{wandb_job_type}"},
    })

    # If the --analyze flag is set, set with_wandb to False
    if args.analyze:
        sweep_config["parameters"]["with_wandb"] = {"value": False}

    if args.trial:
        # Select a random set of parameters
        trial_params = {name: random.choice(info["values"]) if "values" in info else info["value"]
                        for name, info in sweep_config["parameters"].items()}
        command = " ".join([f"--{name}={value}" for name, value in trial_params.items()])
        print(f"python {sweep_config['program']} {command}")
    elif args.save:
        # If the --save flag is set, save the sweep configuration to a YAML file
        with open(f"{sweep_config['name']}.yaml", 'w') as file:
            yaml.dump(sweep_config, file)
    else:
        # Otherwise, create a sweep
        sweep_id = wandb.sweep(sweep_config, project=merged_config["project"])
        print(f"Sweep ID: {sweep_id}")

if __name__ == "__main__":
    main()

