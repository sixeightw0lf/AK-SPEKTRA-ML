import json
import os
import torch
import transformers
from datasets import load_dataset
from git import Repo
from huggingface_hub import login, logout
from pathlib import Path
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,DataCollatorForLanguageModeling, TextDataset,Trainer, TrainingArguments)
import tensorflow as tf

# Check if GPU is available
if not tf.config.list_physical_devices('GPU'):
    print("No GPU found. Please ensure you have installed TensorFlow correctly.")
else:
    print("GPU found.")


# Load settings from Data.json
json_file = json.load(open("config/config.json"))

huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
if huggingface_api_key:
    login(huggingface_api_key)

# Load datasets
data = load_dataset("json", data_files=json_file[0]["data"])
valid_data = load_dataset("json", data_files=json_file[0]["eval_data"])
print("Data loaded")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(json_file[0]["model"])
tokenizer = AutoTokenizer.from_pretrained(json_file[0]["model_tokenizer"])
print("Model loaded")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Training parameters
MICRO_BATCH_SIZE = json_file[0]["MICRO_BATCH_SIZE"]
GRADIENT_ACCUMULATION_STEPS = json_file[0]["GRADIENT_ACCUMULATION_STEPS"]
EPOCHS = json_file[0]["EPOCHS"]
LEARNING_RATE = json_file[0]["LEARNING_RATE"]
CUTOFF_LEN = json_file[0]["CUTOFF_LEN"]
MAX_STEP = json_file[0]["MAX_STEP"]

# Prepare data if not preprocessed
if not json_file[0]['PreProcessedData?']:
    def generate_prompt(data_point):
        if data_point["instruction"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""

    print("Data conversion step 1 done")
    tokenizer.pad_token_id = 0
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

    valid_data = valid_data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

print("Data conversion step 2 done, new train config:::")

# Training arguments
training_args = TrainingArguments(

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=5,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not json_file[0]["CPU_MODE"],
    logging_steps=1,
    output_dir=json_file[0]["out_dir"],
    save_total_limit=10,
    max_steps=MAX_STEP,
    auto_find_batch_size=True if json_file[0]["MICRO_BATCH_SIZE"] == 0 else False,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    evaluation_strategy="steps",
    load_best_model_at_end=json_file[0]["load_best_model_at_end"],
    save_steps=json_file[0]["save_steps"],
    eval_steps=json_file[0]["eval_steps"],
    no_cuda=json_file[0]["CPU_MODE"],
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=valid_data["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Training started!")

# Train the model
trainer.train(resume_from_checkpoint=json_file[0]["Load_Checkpoint"])

# Save the model and tokenizer
trainer.save_model(json_file[0]["out_dir"])
tokenizer.save_pretrained(json_file[0]["out_dir"])

print(f"Training completed and saved. Check the folder {json_file[0]['out_dir']}")

# Push the model to Hugging Face Hub
if json_file[0]["huggingface_access_token"] != "":
    model.push_to_hub("Dampish/ELIAI_1B", use_auth_token=True)
    logout()

print("Done!...")
