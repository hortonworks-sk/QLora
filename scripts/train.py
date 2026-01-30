import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 1. Argument Parsing (Handles SageMaker Hyperparameters) ---
def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker passes hyperparameters as command-line arguments
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e -4)

    # QLoRA/PEFT parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # SageMaker paths (Environment Variables)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    
    args, _ = parser.parse_known_args()
    return args

# --- 2. Main Training Function ---
def main():
    args = parse_args()

    # Define the device and data type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load the pre-processed dataset from the SageMaker input channel
    print(f"Loading dataset from: {args.training_dir}")
    # NOTE: The dataset must be saved in the 'datasets' library format (e.g., as part of the data preparation step)
    dataset = load_from_disk(args.training_dir)
    # Ensure the dataset has a column named 'text' containing the instruction/response pairs

    # --- 3. QLoRA and Quantization Configuration (BitsAndBytes) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # Recommended NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16 # Recommended for better stability
    )

    # --- 4. Load Model and Tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True # Needed for some models like Mistral/Llama
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token # Essential for Causal LMs

    # --- 5. LoRA Configuration (PEFT) ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Target the linear layers commonly used in Llama/Mistral/Gemma
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    )

    # --- 6. Training Arguments (SageMaker Integration) ---
    # SageMaker requires output_dir to be SM_MODEL_DIR for artifact upload
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # Optimization flags
        bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
        fp16=True if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else False,
        gradient_checkpointing=True,
        # Logging and saving
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=False,
    )

    # --- 7. Initialize and Run SFTTrainer (for Instruction Tuning) ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text", # The column in your dataset containing the full, pre-formatted text
        tokenizer=tokenizer,
        max_seq_length=1024, # Maximum length of the sequences
        packing=False, # Set to True for higher throughput if your data is short
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # --- 8. Save the final model (LoRA weights) to the required path ---
    # This automatically saves the LoRA adapter weights only.
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")
    
if __name__ == "__main__":
    main()