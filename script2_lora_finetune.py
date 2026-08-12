# import

import os
import jsonlines
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "6" # physical cores, not hyperthreads
os.environ["MKL_NUM_THREADS"] = "6"
torch.set_num_threads(6)

# Config
BASE_MODEL = "microsoft/phi-1_5"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "models/adapters/lora_adp"
BATCH_SIZE = 1
GRAD_ACCUM = 1
EPOCHS = 5
LR = 2e-4
MAX_LENGTH = 32


def load_training_dataset():
    return load_dataset("json", data_files=DATA_PATH)


def tokenize(example, tokenizer):
    prompt = f"Instruction: {example['instruction']}\nResponse: {example['output']}"

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def main():
    print("\n===== Loading Base Model =====")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("\n=== Loading Dataset ===")
    dataset = load_training_dataset()

    print("\n=== Tokenizing samples or questions or data ===\n")
    tokenized = dataset.map(lambda ex: tokenize(ex, tokenizer), batched=False)

    # LoRA Config
    print("\n=== Applying LoRA ===\n")
    lora_config = LoraConfig(
        r = 8,
        lora_alpha = 128,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"], 
        lora_dropout=0.05,
        bias="none",
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training ---
    print("\n=== Starting Training ===\n")
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs = EPOCHS,
        learning_rate = LR,
        fp16 = False,
        bf16 = False,
        logging_steps = 1,
        save_strategy = "no",
        report_to = "none"
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized["train"],
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    )

    trainer.train()

    print("\n=== Saving LoRA Adapter ===\n")
    model.save_pretrained(OUTPUT_DIR)

    print("\n=== Saving Finetuning Complete ===\n\n")

main()