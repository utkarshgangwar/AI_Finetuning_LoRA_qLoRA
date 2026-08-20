import os
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import warnings

warnings.filterwarnings("ignore")

BASE_MODEL = "microsoft/phi-1_5"
# ADAPTER_PATH = "models/adapters/lora_adp" #LoRA
ADAPTER_PATH = "models/adapters/qlora_adp" #qLoRA
EVAL_FILE = "data/eval_questions.jsonl"


# Creating list of questions from the json file, appending list by iteration.
def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions


# model response get formatted response.
# ex: "Response: Hello, I'm good. How are you?     "
# result of the fnc: "Response: Hello, I'm good. How are you?"
# clean the response
def clean_answer(text):
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()

    for tok in ["\n", "(1)", "(2)", "(3)", "1.", "2.", "3."]:
        if tok in text:
            text = text.split(tok)[0].strip()
    return text


# chat function with the model
def chat(model, tokenizer, question):
    prompt = f"Instruction: {question}\nResponse:"
    input = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_id = model.generate(
            **input,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_id[0], skip_special_token=True)

    return clean_answer(decoded)


def main():
    print("===== Loading the base model (Tokenizer) =====")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("===== Loading the base model (LLM) =====")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("\n\n======= Loading LoRA adapter =======")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("*** Running Evaluation ***\n\n")

    questions = load_eval_questions()

    for question in questions:
        print(f"Question: {question}")
        answer = chat(model, tokenizer, question)
        print(f"Model's Response: {answer} \n\n")
        print("-" * 50, "\n")

main()