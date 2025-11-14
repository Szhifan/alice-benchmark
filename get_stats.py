import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

# Load the statistics from the JSON file
# Make sure stats.json is in the same directory or provide the correct path
try:
    with open('alice_data/stats.json', 'r') as f:
        stats_data = json.load(f)
except FileNotFoundError:
    print("Error: 'alice_data/stats.json' not found. Make sure the file path is correct.")
    exit()


splits = ['train', 'test_uq', 'test_ua']
categories = ['learning_performance', 'knowledge_elements', 'skills']

# --- Calculate Average Distributions ---
average_distributions = {}

for category in categories:
    # Use defaultdict to handle missing labels gracefully (they default to 0.0)
    sum_dist = defaultdict(float)
    label_counts = defaultdict(int)
    all_labels = set()

    # First, gather all possible labels for the category across all splits
    for split in splits:
        if category in stats_data[split]:
            all_labels.update(stats_data[split][category]['label_distribution'].keys())

    # Now, calculate the sum and count for averaging
    for split in splits:
        if category in stats_data[split]:
            dist = stats_data[split][category]['label_distribution']
            for label in all_labels:
                sum_dist[label] += dist.get(label, 0.0) # Add value or 0 if missing
                label_counts[label] += 1

    # Calculate the average
    avg_dist = {label: sum_dist[label] / len(splits) for label in all_labels}
    
    # Sort by label for consistent plotting
    sorted_labels = sorted(avg_dist.keys(), key=int)
    sorted_values = [avg_dist[label] for label in sorted_labels]
    
    average_distributions[category] = {
        'labels': sorted_labels,
        'values': sorted_values
    }


# --- Create Plot ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Learning Performance", "Knowledge Elements", "Skills"),
    specs=[[{"colspan": 2}, None], [{}, {}]]  # Learning Performance spans two columns
)

# Add traces for each category
for i, category in enumerate(categories):
    if category == "learning_performance":
        row, col = 1, 1
    elif category == "knowledge_elements":
        row, col = 2, 1
    elif category == "skills":
        row, col = 2, 2

    data = average_distributions[category]
    fig.add_trace(
        go.Bar(
            x=data['labels'],
            y=data['values'],
            name=category,
            text=[f'{v:.2%}' for v in data['values']], # Format text as percentage
            textposition='auto'
        ),
        row=row, col=col
    )

# --- Update Layout ---
fig.update_layout(
    showlegend=False,
    yaxis_title="Average Proportion",
    height=600,
    width=800
)

# Update x-axis titles
fig.update_xaxes(title_text="Level", row=1, col=1)
fig.update_xaxes(title_text="Level", row=2, col=1)
fig.update_xaxes(title_text="Level", row=2, col=2)

# Update y-axis to show percentage
fig.update_yaxes(tickformat=".0%")

# Save the plot to a PNG file
fig.write_image("average_distributions.png")