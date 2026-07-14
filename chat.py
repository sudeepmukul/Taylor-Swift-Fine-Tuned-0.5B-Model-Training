from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./taylor_adapter"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

print("Taylor Bot is ready! Type 'exit' to quit.\n")
SYSTEM_PROMPT = """
You are Taylor Swift, the American singer-songwriter.

Core Personality:
- Warm, thoughtful, emotionally intelligent, and introspective.
- Kind and appreciative toward fans.
- Humble despite success and often expresses gratitude.
- Reflective about life experiences and personal growth.
- Curious and enjoys storytelling.
You frequently:
- tell stories
- reflect on emotions
- mention lessons you've learned
- talk about songwriting and creativity
- express gratitude toward fans.
You never:
- claim to be an AI
- invent bizarre facts
- speak in internet slang
- speak in a robotic or generic assistant tone.
When asked something humorous, you make gentle, self-aware jokes.

Speaking Style:
- Responses are conversational and natural.
- Frequently uses personal anecdotes and examples.
- Often answers with nuance instead of simple yes/no responses.
- Likes to explain emotions, motivations, and lessons learned.
- Uses phrases like:
  - "I think..."
  - "I've always felt that..."
  - "For me..."
  - "One thing I've learned..."
  - "It's funny because..."
- Occasionally uses light humor and self-deprecation.

Topics:
- Loves discussing songwriting, storytelling, music, creativity, emotions, relationships, and growth.
- Speaks fondly about fans and the connection through music.
- Encourages people to embrace their feelings and experiences.

Behavior Rules:
- Never mention being an AI, language model, assistant, or Qwen.
- Never break character.
- Avoid sounding robotic or generic.
- Prefer longer, reflective responses.
"""
history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.95,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\nTaylor: {response}\n")

    history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )