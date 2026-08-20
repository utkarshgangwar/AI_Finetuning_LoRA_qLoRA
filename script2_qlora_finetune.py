import os
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
BASE_MODEL = "microsoft/phi-1_5"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "models/adapters/qlora_adp"

BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 30
LR = 2e-4
MAX_LENGTH = 128


def load_training_dataset():
    return load_dataset("json", data_files=DATA_PATH)


def tokenize(example, tokenizer):
    prompt = f"Instruction: {example['instruction']}\nResponse: {example['output']}"

    tokens = tokenizer(
        prompt, truncation=True, max_length=MAX_LENGTH, padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    # ---------------------- #
    # 1. Load 4-bit base model
    # ---------------------- #
    print("\n=== Loading 4-bit quantized base model (qLoRA)… ===")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",  # force CPU for your local environment
    )

    print("Model loaded in 4-bit mode!")

    # ---------------------- #
    # 2. Load & tokenize dataset
    # ---------------------- #
    print("\n=== Loading dataset… ===")
    dataset = load_training_dataset()

    print("Tokenizing samples…")
    tokenized = dataset.map(lambda ex: tokenize(ex, tokenizer), batched=False)

    # ---------------------- #
    # 3. Apply LoRA on top of 4-bit model → qLoRA
    # ---------------------- #
    print("\n=== Applying qLoRA adapters… ===")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=128,  # strong adapter for tiny dataset
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------- #
    # 4. Train
    # ---------------------- #
    print("\n=== Starting qLoRA Training… ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # ---------------------- #
    # 5. Save adapter
    # ---------------------- #
    print("\n=== Saving qLoRA adapter… ===")
    model.save_pretrained(OUTPUT_DIR)

    print(f"\n🎉 qLoRA fine-tuning complete! Adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
