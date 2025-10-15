# Fine-tuning and benchmarking legal LLMs:

This repository contains code for:
1. Fine-tuning Qwen2.5-3B with 4-bit QLoRA and full LoRA using Unsloth.
2. Benchmarking final models against base model(Qwen2.5-3B) and Qwen2.5-7B using LegalBench.

---

## Dataset:

Taken from hugging face **[dzunggg/legal-qa-v1](https://huggingface.co/datasets/dzunggg/legal-qa-v1)** for legal question-answering style model fine-tuning.

---

## Model selection:

Based on **[vals.ai's legal leaderboard](https://www.vals.ai/benchmarks/legal_bench)** Qwen2.5 Turbo Instruct(7B) is the top most model (with open weights) that fits locally on my 4060 8GB laptop GPU with 4-bit quantization enabled.

For reference this model has higher accuracy than bigger models like **Llama 2(70B), Gemma 2(27B)**, and many more.

Fine-tuning Qwen2.5 Turbo Instruct(7B) QLoRA with 4-bit quantization possible but LoRA with no quantization isn't, so I downgraded to **Qwen2.5 Turbo Instruct(3B)**.

---

## Benchmarking:

[LegalBench](https://github.com/HazyResearch/legalbench) is industry standard for testing LLMs on legal tasks and is also used by leaderboards(Vals.ai).

The benchmark uses **Few-Shot prompting** with lesser than 10 examples in train.tsv and testing the responses on test.tsv for each task.

It has total of 163 tasks(162 automated, 1 manual), but according do the dataset selected, following tasks were performed:

1.  contract_nli_confidentiality_of_agreement
2.  contract_nli_permissible_copy
3.  contract_nli_return_of_confidential_information
4.  contract_qa
5.  cuad_third_party_beneficiary
6.  hearsay
7.  jcrew_blocker
8.  legal_reasoning_causality
9.  maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier
10. 
    maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures
11. personal_jurisdiction
12. proa
13. telemarketing_sales_rule
14. ucc_v_common_law
15. consumer_contracts_qa

---

## Results:

![results](/results.png)

- QLoRA: **Outperformed base 3B in 10 tasks**
  - (contract_nli_confidentiality_of_agreement: 0.4268 vs. 0.0, contract_nli_permissible_copy: 0.50 vs. 0.0, contract_nli_return_of_confidential_information: 0.4007 vs. 0.0, contract_qa: 0.6329 vs. 0.0, cuad_third_party_beneficiary: 0.3824 vs. 0.0, hearsay: 0.6424 vs. 0.0782, jcrew_blocker: 0.4556 vs. 0.0, maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier: 0.3404 vs. 0.0, telemarketing_sales_rule: 0.6109 vs. 0.2566, ucc_v_common_law: 0.7037 vs. 0.0). 
  - Its 4-bit quantization and r=4 reduced overfitting, **beating LoRA in 11 tasks**.

- LoRA: **Improved over base 3B in 9 tasks**
  -  (contract_nli_confidentiality_of_agreement: 0.1707 vs. 0.0, contract_nli_permissible_copy: 0.50 vs. 0.0, contract_nli_return_of_confidential_information: 0.0781 vs. 0.0, contract_qa: 0.5854 vs. 0.0, hearsay: 0.50 vs. 0.0782, jcrew_blocker: 0.3444 vs. 0.0, maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier: 0.2215 vs. 0.0, maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures: 0.4038 vs. 0.0556, telemarketing_sales_rule: 0.7086 vs. 0.2566), **but lower scores suggest overfitting** (r=16, training loss 2.0616 vs. QLoRA’s 2.1034).

- Base Models: **7B led in 6 tasks**
  -  (consumer_contracts_qa: 0.8637, contract_qa: 0.9744, hearsay: 0.6866, legal_reasoning_causality: 0.7695, personal_jurisdiction: 0.6305, proa: 0.9574)
  -  **QLoRA/LoRA excelled in specialized tasks** like contract_nli_* and ucc_v_common_law compared to 7B.


