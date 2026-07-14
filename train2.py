from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load dataset
dataset = load_dataset(
    "json",
    data_files="taylor_dataset.json",
)["train"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

# LoRA config for Qwen
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)

# Convert conversations to chat template
def formatting(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(formatting)

# Training settings
training_args = TrainingArguments(
    output_dir="./taylor_adapter",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    optim="adamw_torch",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save adapter
trainer.save_model("./taylor_adapter")
tokenizer.save_pretrained("./taylor_adapter")