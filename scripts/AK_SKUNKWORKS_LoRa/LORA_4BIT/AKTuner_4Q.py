
import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, PEFT


class Prompter:
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "../../../datasets/LoRa/4Q",
    output_dir: str = "./scripts/AK_SKUNKWORKS_LoRa/output",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # Instantiate the Prompter object
    prompter = Prompter(prompt_template_name)

    # ... (the rest of the code from the provided example)

    # In the training loop, use the 'prompter' object to generate prompts and extract responses as needed
    # For example:
    instruction = "Write a Python function that takes two arguments and returns their sum."
    input_code = "def add(a, b):"
    prompt = prompter.generate_prompt(instruction, input_code)

    # To extract the response from the model's output:
    model_output = "Answer: def add(a, b): return a + b"
    response = prompter.get_response(model_output)

    # ... (continue with the rest of the training loop and other necessary code)

    pass


def main():
    parser = argparse.ArgumentParser(description="LoRa Tuner for Alpaca Model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model to use for training")
    parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./lora-alpaca", help="Directory to store the output files")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--cutoff_len", type=int, default=256, help="Cutoff length for training")
    parser.add_argument("--val_set_size", type=int, default=2000, help="Validation set size")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRa r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRa alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRa dropout parameter")
    parser.add_argument("--lora_target_modules", nargs="*", default=["q_proj", "v_proj"], help="LoRa target modules")
    parser.add_argument("--train_on_inputs", action="store_true", help="Train on inputs in loss")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token")
    parser.add_argument("--group_by_length", action="store_true", help="Group by length for faster training")
    parser.add_argument("--wandb_project", type=str, default="", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Wandb run name")
    parser.add_argument("--wandb_watch", type=str, default="", help="Wandb watch setting")
    parser.add_argument("--wandb_log_model", type=str, default="", help="Wandb log model setting")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca", help="Prompt template name")

    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
  learning_rate=args.learning_rate,
    cutoff_len=args.cutoff_len,
    val_set_size=args.val_set_size,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    lora_target_modules=args.lora_target_modules,
    train_on_inputs=args.train_on_inputs,
    add_eos_token=args.add_eos_token,
    group_by_length=args.group_by_length,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name,
    wandb_watch=args.wandb_watch,
    wandb_log_model=args.wandb_log_model,
    resume_from_checkpoint=args.resume_from_checkpoint,
    prompt_template_name=args.prompt_template_name,
)


if __name__ == "__main__":
    main()






