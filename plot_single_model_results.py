import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load JSON data from file
with open("output/single_model_results.json", "r") as f:
    data = json.load(f)

# Define models and tasks
models = [
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
]
tasks = ["sentiment", "recommendation"]
tasks_long = ["sentiment analysis", "movie recommendation"]

# Compute average latencies
avg_latency = {
    task: [np.mean(data[task][model]["latency"]) for model in models]
    for task in tasks
}

# Prepare bar chart
x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, [avg_latency[t][0] for t in tasks], width, label="Llama 3.2 1B")
bars2 = ax.bar(x + width/2, [avg_latency[t][1] for t in tasks], width, label="Llama 3.2 3B")

# Labels & styling
ax.set_ylabel("Average Latency (s)")
ax.set_title("Average Latency per Model and Task")
ax.set_xticks(x)
ax.set_xticklabels(tasks_long)
ax.legend()

# Annotate bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.show()
