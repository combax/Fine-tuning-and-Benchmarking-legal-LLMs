import os
import json
import pandas as pd
import plotly.graph_objects as go

# Define paths to JSON files
base_dir = r"\legalbench"
json_files = {
    "Base 3B": os.path.join(base_dir, "base_model_3B.json"),
    "Base 7B": os.path.join(base_dir, "base_model_7B.json"),
    "Standard LoRA": os.path.join(base_dir, "standard_lora_legalbench.json"),
    "QLoRA": os.path.join(base_dir, "qlora_legalbench.json")
}

# Load data from JSON files
data_dict = {}
for name, path in json_files.items():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data_dict[name] = data.get("task_scores", {})

# Define reference tasks
reference_tasks = set([
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
])

# Combine data into DataFrame
rows = []
for model_name, scores in data_dict.items():
    for task in reference_tasks:
        rows.append({
            "task": task,
            "model": model_name,
            "score": scores.get(task, None)
        })

combined_df = pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)

# Round scores to 4 decimal places
combined_df["score"] = combined_df["score"].round(4)

# Create pivot table for plotting
pivot_df = combined_df.pivot(index="task", columns="model", values="score")
pivot_df = pivot_df.sort_values(by="Base 7B", ascending=True)

# Define label mapping for display
label_map = {
    "maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier": "maud_modifier",
    "maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures": "maud_measures",
    "contract_nli_return_of_confidential_information": "contract_nli_information",
    "contract_nli_confidentiality_of_agreement": "contract_nli_agreement"
}
pivot_df["display_label"] = pivot_df.index.map(lambda t: label_map.get(t, t))

# Define model colors
model_colors = {
    "Base 3B": "#264653",
    "Base 7B": "#2a9d8f",
    "Standard LoRA": "#f4a261",
    "QLoRA": "#e76f51"
}

# Create Plotly figure
fig = go.Figure()
for model in pivot_df.columns.drop("display_label"):
    fig.add_trace(go.Bar(
        x=pivot_df["display_label"],
        y=pivot_df[model],
        name=model,
        marker_color=model_colors.get(model, "#333333")
    ))

fig.update_layout(
    barmode='group',
    xaxis=dict(
        title=dict(
            text='Tasks',
            font=dict(family="cursive", size=14, color='black')
        ),
        showline=True,
        linewidth=2,
        linecolor='black',
        tickfont=dict(family="cursive")
    ),
    yaxis=dict(
        title=dict(
            text='Accuracy',
            font=dict(family="cursive", size=14, color='black')
        ),
        showline=True,
        linewidth=2,
        linecolor='black',
        range=[0, 1],
        tickfont=dict(family="cursive")
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        title='       Models',
        x=0.02,
        y=0.98,
        xanchor='left',
        yanchor='top',
        borderwidth=0,
        font=dict(family="cursive")
    ),
    font=dict(family="cursive", size=12, color='black'),
    title=dict(
        text="Results",
        x=0.5,
        xanchor='center',
        font=dict(family="cursive", size=18, color='black')
    ),
    width=1000,
    height=550
)

fig.show()

# Print ranked results
ranked_df = combined_df.copy()
ranked_df["rank"] = ranked_df.groupby("task")["score"].rank(ascending=False, method="min")
ranked_df = ranked_df.sort_values(["task", "rank"]).reset_index(drop=True)

for task in ranked_df["task"].unique():
    print(f"\n🔹 Task: {task}")
    task_rows = ranked_df[ranked_df["task"] == task][["model", "score", "rank"]]
    for _, row in task_rows.iterrows():
        print(f"  {int(row['rank'])}. {row['model']} — {row['score']:.4f}")
        