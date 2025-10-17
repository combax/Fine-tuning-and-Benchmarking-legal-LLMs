import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

#paths
base_model_path = r"\Qwen\Qwen2.5-3B-Instruct"
#best_checkpoint_path = r"\outputs_lora\checkpoint-450"
best_checkpoint_path = r"\Qwen\Standard_LoRA_model"

quantization_config = BitsAndBytesConfig(load_in_4bit=False)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda",
    quantization_config=quantization_config,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, best_checkpoint_path)

messages = [
    {"role": "system", "content": "You are a helpful legal assistant."},
    {"role": "user", "content": "I was wondering if a pain management office is acting illegally/did an illegal action.. I was discharged as a patient from a pain management office after them telling me that a previous pain management specialist I saw administered a steroid shot wrong and I told them in the portal that I spoke to lawyers for advice but no lawsuit/case was created. It was maybe 1-2 months after I was discharged that I no longer have access to my patient portal with them. Every time I try to login I enter my credentials, wait a few seconds, and then I get re-directed back to the original screen where I have various options to login. I know I can speak to the office directly and ask them about what specifically is going on, talk to other lawyers if this is a violation of my rights, etc. but I was just wondering if anyone on this site would know if this action is in fact illegal."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
