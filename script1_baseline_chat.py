# imports the libraries
# path of the data directory
# base model
# load eval questions function
# chat with the model locally

import jsonlines  # to work with json data
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # paramter efficient fine tuning

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "data"
EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.jsonl")
BASE_MODEL = "microsoft/phi-1_5"  # CPU Friendly


def load_eval_question():
    questions = []
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for l in reader:
            questions.append(l["question"].strip())
    return questions


def chat(model, tokenizer, question):
    """Generate response for a given question"""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():  # tells pytorch don't track gradients (the values do I need to go for or back)
        # Normally, PyTorch tracks all operations on tensors to compute gradients for training neural networks. (backpropagation).
        # When you care only running inference ( making predictions, not training), you don't need gradients.
        output_ids = model.generate(
            **inputs,  # model receives the tokenized prompts
            max_new_tokens=80,  # The model can generate up to 80 tokens
            do_sample=False,  # Disable randomness, The model alway picks the most probable next token
            # ex: happiness, sadness then with happiness joy will be used not ecstasy/blissfull.
            # Will use frequents ones
            temperature=0.1,  # Controls how "sharp" or "flat" token probabilities are, more factual, less creative
            pad_token_id=tokenizer.eos_token_id,
            # Huggin Face requires a pad token for generation, Prevents warnings or runtime error
            # Some models need all sequence in a batch to be the same length, so they add "padding" token to
            # shorter sequences.
            # The pad_token_id argument specifies which token ID should be used for this padding.
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_token=True)
    return decoded


def run_baseline_evaluation():
    print("\nLoading base model ... \n\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # tokenizer - Load the text-to-numbes converter that was trained alongside this model.

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    # model - Load the actual neural network weights of the language model.

    questions = load_eval_question()

    print("\n==== Baseline Response (Before Finetunning) ====\n\n")

    for q in questions:
        print(f"Question: {q}\n")
        answer = chat(model, tokenizer, q)
        print(f"Model's Final Response: {answer}\n\n")
        print("-"*60)

run_baseline_evaluation()
