import os
import subprocess
import yaml
import hiyapyco
import argparse
import time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run training with multiple configurations.')
    parser.add_argument('yaml_files', type=str, nargs='+', 
                        help='List of YAML configuration files.')
    parser.add_argument('--sleep', type=int, default=60, 
                        help='Number of seconds to sleep between each launch (default: 60).')
    parser.add_argument('--analyze', action='store_true', 
                        help='If set, run analyze.py instead of train.py and set with_wandb to False.')
    args = parser.parse_args()

    # Load and merge the YAML files
    configs = [yaml.safe_load(open(filename)) for filename in args.yaml_files]
    merged_config = hiyapyco.load(configs, method=hiyapyco.METHOD_OVERWRITE)

    # Construct the job type variable
    wandb_job_type = "_".join(os.path.splitext(os.path.basename(filename))[0] for filename in args.yaml_files)

    # Define the experiment variable
    experiment = "-".join(
        f"{param_name}_{{{param_name}}}"
        for param_name, param_info in merged_config["parameters"].items()
        if "values" in param_info and len(param_info["values"]) > 1
    )

    # Update the merged config
    merged_config["parameters"]["wandb_job_type"] = {"value": wandb_job_type}
    merged_config["parameters"]["experiment"] = {"value": experiment}

    # If the analyze flag is set, run analyze.py and set with_wandb to False
    if args.analyze:
        script_name = "analyze.py"
        merged_config["parameters"]["with_wandb"] = {"value": False}
    else:
        script_name = "train.py"

    # Loop over variables with multiple values
    for param_name, param_info in merged_config["parameters"].items():
        if "values" in param_info and len(param_info["values"]) > 1:
            for value in param_info["values"]:
                # Update the value for the current parameter
                merged_config["parameters"][param_name] = {"value": value}

                # Construct the command line arguments
                args = [
                    f"--{name} {info['value']}"
                    for name, info in merged_config["parameters"].items()
                    if "value" in info
                ]

                # Launch the script
                command = f"python {script_name} {' '.join(args)}"
                subprocess.run(command, shell=True)

                # Sleep for the desired number of seconds
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
