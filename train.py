from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,Y
)
from trl import SFTTrainer
import torch

MODEL = "microsoft/Phi-3-mini-4k-instruct"

dataset = load_dataset(
    "json",
    data_files="taylor_dataset.json"
)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)

training_args = TrainingArguments(
    output_dir="taylor_adapter",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
    dataset_text_field="text",
)

trainer.train()

trainer.save_model("taylor_adapter")