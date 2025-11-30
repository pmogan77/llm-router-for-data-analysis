import json
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load JSON data
with open("output/router_results.json", "r") as f:
    data = json.load(f)

tasks = ["sentiment", "recommendation"]
latency_types = ["routing_latency", "total_latency"]

# Compute average latencies aggregated across models
avg_latencies = {task: {} for task in tasks}

for task in tasks:
    for latency_type in latency_types:
        values = []
        for model, model_data in data[task].items():
            raw = model_data.get(latency_type, [])
            # Filter out None/null values
            clean = [x for x in raw if x is not None]
            if clean:
                values.extend(clean)
        avg_latencies[task][latency_type] = np.mean(values) if values else 0.0

# Prepare data for plotting
x = np.arange(len(tasks))
width = 0.35

routing_vals = [avg_latencies[t]["routing_latency"] for t in tasks]
total_vals = [avg_latencies[t]["total_latency"] for t in tasks]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, routing_vals, width, label="Routing Latency")
bars2 = ax.bar(x + width/2, total_vals, width, label="Total Latency")

# Labels & styling
ax.set_ylabel("Average Latency (s)")
ax.set_title("Average Routing vs Total Latency (Aggregated Across Models)")
ax.set_xticks(x)
ax.set_xticklabels(["Sentiment Analysis", "Movie Recommendation"])
ax.legend()

# Annotate each bar with its value
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.show()
