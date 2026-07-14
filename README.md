---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- lora
- peft
- conversational
- chatbot
- taylor-swift
- roleplay
- qwen
language:
- en
pipeline_tag: text-generation
---

# TaylorSwiftChatbot :)

A LoRA fine-tuned version of **Qwen2.5-0.5B-Instruct**, trained on a curated dataset of conversational examples inspired by Taylor Swift's interviews, public appearances, and speaking style.

> ⚠️ This is an experimental fan project intended for research and educational purposes only. It is not affiliated with or endorsed by Taylor Swift.

---

## Model Details

- **Base Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Fine-Tuning Method:** LoRA (PEFT)
- **Training Hardware:** NVIDIA RTX 3050 Laptop GPU (6GB VRAM)
- **Training Time:** ~15 minutes
- **Dataset Size:** ~367 conversational examples
- **Epochs:** 5

---

## Goal

The goal of this project is to explore whether a small language model can learn:

- Conversational tone
- Storytelling style
- Emotional responses
- Interview mannerisms
- Personality traits and speaking patterns

This model focuses on **style imitation**, not factual knowledge.

---

## Current Status

Version 1 is an early prototype.

### Strengths
✅ Captures some aspects of Taylor's reflective and conversational tone.

✅ Produces longer and more personal responses than the base model.

✅ Demonstrates personality conditioning despite the small dataset.

### Limitations
❌ Limited dataset size.

❌ Can still sound like the base Qwen model.

❌ May hallucinate facts or generate inaccurate information.

❌ Personality consistency is not yet reliable.

---

## Usage

### Load the Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "intentfx/TaylorSwiftChatbot"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)
```

---

### Example Prompt

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are Taylor Swift, the singer-songwriter. "
            "Speak warmly, thoughtfully, and introspectively."
        )
    },
    {
        "role": "user",
        "content": "How do you approach songwriting?"
    }
]
```

---

## Future Improvements

- Larger and higher quality dataset
- More interview and fan interaction examples
- Better system prompts
- Synthetic conversational data generation
- Fine-tuning on larger base models (1.5B to 3B)
- Improved personality consistency

---

## Disclaimer

This model attempts to imitate a public speaking style and should not be considered a representation of the real person's beliefs, opinions, or future statements.

This project is intended solely for:

- Research
- Education
- Experimentation with LLM fine-tuning and personality modeling

---

## Acknowledgements

- Qwen Team for the base model.
- Hugging Face for open-source tooling.
- PEFT and TRL libraries for efficient fine-tuning.

---

Built by **Intent (Sudeep Mukul)** 