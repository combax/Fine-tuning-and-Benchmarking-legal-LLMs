import os
import json
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
import warnings
import transformers

#silence warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers.logging.set_verbosity_error()

#paths
repo_base_dir = r"\legalbench"
data_base_dir = r"\legalbench_dataset\data"

#fine-tuned paths
base_model_path = r"\Qwen\Qwen2.5-3B-Instruct" #base model
qlora_path = r"\Qwen\Standard_LoRA_model" #replace with QLoRA/LoRA as required

#output
output_path = os.path.join(repo_base_dir, "qlora_high_performer_results_3B_4bit.json")

#selected benchmarks
manual_selected_tasks = [
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_permissible_copy",
    "contract_nli_return_of_confidential_information",
    "contract_qa",
    "cuad_third_party_beneficiary",
    "hearsay",
    "jcrew_blocker",
    "legal_reasoning_causality",
    "maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier",
    "maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures",
    "personal_jurisdiction",
    "proa",
    "telemarketing_sales_rule",
    "ucc_v_common_law",
    "consumer_contracts_qa"
]

print(f"\nSelected {len(manual_selected_tasks)} high-performing tasks:")
for t in manual_selected_tasks:
    print(" -", t)


#generation and evaluation scripts from legalbench repo
target_dir = os.path.abspath(repo_base_dir)
sys.path.append(target_dir)
from evaluation import evaluate
from utils import generate_prompts


#load model
print("\nLoading model")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #False for LoRA model
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

#below code is for testing base models without LoRA/QLoRA
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     device_map="cuda",
#     quantization_config=bnb_config,
#     trust_remote_code=True
# )
# model.eval()
# print("Base model loaded")

#QLoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, qlora_path)
model.eval()
print("QLoRA Loaded\n")


#generate function according to legalbench
def generate_response(prompt, max_new_tokens=100):
    messages = [
        {"role": "system", "content": "You are a helpful legal assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2000).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()


#output json
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    completed = set(results.get("task_scores", {}).keys())
    print(f"Loaded existing results file ({len(completed)} completed).") #incase the benchmark crashes
else:
    results = {"task_scores": {}, "summary": {}}
    completed = set()

tasks_to_run = [t for t in manual_selected_tasks if t not in completed]
print(f"Remaining to benchmark: {len(tasks_to_run)}")


#benchmarking loop
for task in tqdm(tasks_to_run, desc="QLoRA (3B) high-performing tasks"):
    try:
        test_path = os.path.join(data_base_dir, task, "test.tsv")
        prompt_path = os.path.join(repo_base_dir, "tasks", task, "base_prompt.txt")

        if not os.path.exists(test_path) or not os.path.exists(prompt_path):
            print(f"\nMissing files for {task}, to skip")
            continue

        test_df = pd.read_csv(test_path, sep="\t")
        if test_df.empty or "answer" not in test_df.columns:
            print(f"\nEmpty/invalid dataset for {task}, to skip")
            continue

        #prompt loading
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompts = generate_prompts(prompt_template, test_df)

        #dynamically selecting max_new_tokens
        first_answer = str(test_df["answer"].iloc[0])
        first_answer_token_len = len(tokenizer.tokenize(first_answer))
        if first_answer_token_len == 1:
            max_new_tokens = 1
        elif first_answer_token_len <= 5:
            max_new_tokens = 5
        else:
            max_new_tokens = first_answer_token_len + 3

        print(f"\n{task}: using max_new_tokens={max_new_tokens} (first answer len={first_answer_token_len})")

        #predictions
        predictions = []
        for prompt in tqdm(prompts, desc=f"{task}", leave=False):
            pred = generate_response(prompt, max_new_tokens=max_new_tokens)
            predictions.append(pred)

        #evalutaion
        true_answers = test_df["answer"].tolist()
        score = evaluate(task, predictions, true_answers)
        results["task_scores"][task] = float(score)

        #save directly after execution in case of crashing
        results["summary"] = {
            "total_tasks_completed": len(results["task_scores"]),
            "overall_average": round(sum(results["task_scores"].values()) /
                                     len(results["task_scores"]), 4)
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"{task} complete — score {score:.4f} (saved)")

    except Exception as e:
        print(f"\nError while processing {task}: {str(e)}")
        continue


#summary of testing
print("Benchmark ended")
print(f"Results saved to: {output_path}")
print(f"Tasks completed: {len(results['task_scores'])}")
print(f"Overall average: {results['summary'].get('overall_average', 0):.4f}")
