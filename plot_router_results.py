import numpy as np
import json

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open('output/router_results.json') as f:
    router_results = json.load(f)

# extract number of routings
tasks = ["sentiment", "recommendation"]
tasks_long = ["sentiment analysis", "movie recommendation"]

models = [
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
]

# Count how many times each model was routed per task
routing_counts = {
    task: [len(router_results[task][model]["routing_latency"]) for model in models]
    for task in tasks
}

# plotting
x = np.arange(len(tasks))  # task positions
width = 0.5

fig, ax = plt.subplots(figsize=(8,6))

# stack bars
bar1 = ax.bar(x, [routing_counts[t][0] for t in tasks], width, label='Llama 3.2 1B')
bar2 = ax.bar(x, [routing_counts[t][1] for t in tasks], width,
              bottom=[routing_counts[t][0] for t in tasks],
              label='Llama 3.2 3B')

# labels and style
ax.set_ylabel('Number of Routings')
ax.set_title('Model Routing Counts per Task')
ax.set_xticks(x)
ax.set_xticklabels(tasks_long)
ax.legend()

# Add value labels
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2,
                f'{int(height)}', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
