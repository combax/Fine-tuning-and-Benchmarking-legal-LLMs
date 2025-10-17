import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load model and tokenizer
model_path = r"\Qwen\Qwen2.5-3B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    quantization_config=quantization_config
)

# Your message
#messages = [{"role": "user", "content": "Who are you?"}]
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

# Get response
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)