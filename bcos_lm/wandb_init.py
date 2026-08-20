# This python code is used to load the results of a training run from an already run checkpoint and has no wandb implementation.
#  It reads the training state from trainer_state.json file, initializes a WandB run with the specified project name, and logs the training and evaluation metrics step by step.

import json
import wandb
import argparse

def load_results(checkpoint_path, project_name):
    with open(f"{checkpoint_path}/trainer_state.json") as f:
        state = json.load(f)

    wandb.init(project=project_name, 
               name=f"results_{checkpoint_path.split('/')[-1]}",
               config={
        "model": "BCos-GPT2",
        "batch_size": state["train_batch_size"],
        "num_train_epochs": state["num_train_epochs"],
        "max_steps": state["max_steps"],
        "logging_steps": state["logging_steps"],
        "save_steps": state["save_steps"],
        "total_flos": state["total_flos"],
    },)

    for entry in state["log_history"]:
        step = entry.get("step")
        if step is None:
            continue

        metrics = {}

        #Training metrics
        if "loss" in entry:
            metrics["train/loss"] = entry["loss"]
        if "learning_rate" in entry:
            metrics["train/learning_rate"] = entry["learning_rate"]

        #Evaluation metrics
        if "eval_loss" in entry:
            metrics["eval/loss"] = entry["eval_loss"]

        if metrics:
            wandb.log(metrics, step=step)

    wandb.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--project",
                        type=str,
                        default="bcos_gpt2",
                        help="WandB project name.")

    args = parser.parse_args()

    load_results(args.checkpoint, args.project)