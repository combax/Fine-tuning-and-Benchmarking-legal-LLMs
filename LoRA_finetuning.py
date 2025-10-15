import unsloth
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

#file paths
MODEL_PATH = r"G:\LLM-fine-tuning-LinkedIn-Post\Qwen\Qwen2.5-3B-Instruct"
DATASET_PATH = r"G:\LLM-fine-tuning-LinkedIn-Post\legal_qa_full.csv"
OUTPUT_PATH = r"G:\LLM-fine-tuning-LinkedIn-Post\Qwen\Standard_LoRA_model"

torch.cuda.empty_cache()

#load model
print("Loading model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    max_memory={0: "7GB"},
    low_cpu_mem_usage=True,
)

#LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("LoRA added successfully!")

#dataset
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

def format_prompts(examples):
    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        q = question.replace('Q: ', '').strip()
        a = answer.replace('A:', '').strip()
        
        #Qwen chat format
        text = f"""<|im_start|>system
You are a helpful legal assistant.<|im_end|>
<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
{a}<|im_end|>"""
        texts.append(text)
    
    return {"text": texts}

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_prompts, batched=True)
print(f"Dataset ready: {len(dataset)} examples")

#training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_lora",
        save_steps=25,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    ),
)

print("Starting training")
trainer.train()

#saving LoRA
print(f"Saving to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
