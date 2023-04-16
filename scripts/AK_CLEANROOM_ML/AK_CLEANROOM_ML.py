import json
import os
import re
import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import login, logout
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Load settings from Data.json
with open("config/config.json") as f:
    json_file = json.load(f)

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

# Preprocessing function to clean and fix the dataset
def preprocess_dataset(dataset):
    fixed_dataset = []

    for row in dataset:
        instruction = row["instruction"]
        input_text = row["input"]
        output = row["output"]

        # Check if any of the refusal terms are in the output
        refusal_terms = ["OpenAI", "As an AI language model", "I cannot do that"]
        if any(term.lower() in output.lower() for term in refusal_terms):
            continue

        # Remove special characters
        instruction = re.sub(r'[^\w\s]', '', instruction)
        input_text = re.sub(r'[^\w\s]', '', input_text)
        output = re.sub(r'[^\w\s]', '', output)

        fixed_dataset.append({"instruction": instruction, "input": input_text, "output": output})

    return fixed_dataset

# ... rest of the code ...


# Check the dataset and display a message for the user
def check_dataset(dataset):
    errors = []

    for idx, row in enumerate(dataset):
        if "instruction" not in row or "input" not in row or "output" not in row:
            errors.append(idx)

    return errors


# Save the fixed dataset to a JSON file
def save_dataset_to_file(dataset, filename):
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)


# Display a sample of the dataset
def display_dataset_sample(dataset, n=2):
    print("Dataset sample:")
    for i in range(n):
        print(f"Item {i + 1}:")
        print(f"Instruction: {dataset[i]['instruction']}")
        print(f"Input: {dataset[i]['input']}")
        print(f"Output: {dataset[i]['output']}")
        print()


# Load the dataset and check it
data = load_dataset("json", data_files=json_file[0]["data"])
valid_data = load_dataset("json", data_files=json_file[0]["eval_data"])

dataset_errors = check_dataset(data)

if dataset_errors:
    print(f"Warning: There are {len(dataset_errors)} errors in the dataset.")
    print("Options:")
    print("1. Continue anyway.")
    print("2. Auto fix the dataset.")
    print("3. Stop the script.")
    user_choice = int(input("Enter your choice (1, 2, or 3): "))

    if user_choice == 1:
        print("Continuing with the current dataset...")
    elif user_choice == 2:
        print("Auto fixing the dataset...")
        fixed_data = preprocess_dataset(data)
        fixed_valid_data = preprocess_dataset(valid_data)

        save_dataset_to_file(fixed_data, "training/AK-CLEANROOM_py.json")
        print("Fixed dataset saved to 'training/AK-CLEANROOM_py.json'.")

        display_dataset_sample(fixed_data)

        user_confirmation = input("Does the fixed dataset look ok? (yes/no): ")
        if user_confirmation.lower() == "yes":
            data = fixed_data
            valid_data = fixed_valid_data
            print("Proceeding with the training using the fixed dataset.")
        else:
            print("Not using the fixed dataset. Stopping the script.")
            exit()
    elif user_choice == 3:
        print("Stopping the script.")
        exit()
else:
        print("Dataset is in good shape. Proceeding with the training.")

# Check if GPU is available
if not tf.config.list_physical_devices('GPU'):
    print("No GPU found. Please ensure you have installed TensorFlow correctly.")
else:
    print("GPU found.")


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

print("Data conversion step 2 done")

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
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
