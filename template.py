import os
import pathlib


def create_lora_qlora_structure():
    # Define library with key value pair, value as a list
    # which will create the folders and files
    structure = {
        "": [  # Root directory files
            "script1_baseline_chat.py",
            "script2_lora_finetune.py",
            "script2_qlora_finetune.py",
            "script3_chat_with_adapter.py",
            "requirements.txt",
            "README.md",
        ],
        "data": ["train.jsonl", "eval_questions.jsonl"],
        "models/base": [".gitkeep"],  # Placeholder file to keep the directory
        "models/adapters": [".gitkeep"],  # Placeholder file to keep the directory
    }

    # for loop on the structure
    # create folders from keys
    # loop to check for files from values
    # create empty files with pathlib library

    for folder, items in structure.items():
        if folder:
            # create folder if it doesn't exist
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created folder: {folder}/")

        for item in items:
            # create files within the foler
            file_path = os.path.join(folder, item) if folder else item

            if item.endswith("/"):
                os.makedirs(file_path, exist_ok=True)
                print(f"📁 Created folder: {file_path}")
            else:
                pathlib.Path(file_path).touch()
                print(f"🗒️ Created File: {file_path}")

    print("\n ✔️ LoRa/qLoRA Fine-tuning Project structure created successfully")
    print("📂 All files and folders are now ready.")


if __name__ == "__main__":
    create_lora_qlora_structure()
