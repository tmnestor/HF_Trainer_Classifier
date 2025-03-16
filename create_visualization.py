import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Load the data
data_path = Path("results/classifier_results_summary.csv")
df = pd.read_csv(data_path)

# Sort the data by F1 macro score (descending)
df = df.sort_values(by='eval_f1_macro', ascending=False)

# Extract the main performance metrics
metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1_macro', 'eval_matthews_correlation']
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Macro', 'MCC']

# Create a dataframe for plotting
plot_df = df[['classifier'] + metrics].copy()
plot_df.columns = ['Classifier'] + metrics_labels

# Melt the dataframe for seaborn
melted_df = pd.melt(plot_df, id_vars=['Classifier'], var_name='Metric', value_name='Score')

# Create the figure
plt.figure(figsize=(14, 10))

# Create the bar plot
ax = sns.barplot(x='Classifier', y='Score', hue='Metric', data=melted_df, palette='viridis')

# Configure the plot
plt.title('Performance Comparison of Classifier Architectures', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.70, 1.0)  # Set y-axis limits
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('evaluation/classifier_comparison_barplot.png', dpi=300, bbox_inches='tight')

# Create second visualization - heatmap of metrics
plt.figure(figsize=(12, 10))
heatmap_df = df.set_index('classifier')[metrics]
heatmap_df.index = heatmap_df.index.str.capitalize()
heatmap_df.columns = metrics_labels

# Create the heatmap
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
plt.title('Performance Metrics Heatmap', fontsize=20)
plt.tight_layout()
plt.savefig('evaluation/classifier_metrics_heatmap.png', dpi=300, bbox_inches='tight')

# Create a third visualization focused just on F1 and MCC (paper-friendly)
plt.figure(figsize=(10, 6))
paper_metrics = ['eval_f1_macro', 'eval_matthews_correlation']
paper_labels = ['F1 Macro', 'Matthews Correlation']

paper_df = df[['classifier'] + paper_metrics].copy()
paper_df.columns = ['Classifier'] + paper_labels
paper_melted = pd.melt(paper_df, id_vars=['Classifier'], var_name='Metric', value_name='Score')

# Create horizontal bar chart
sns.barplot(y='Classifier', x='Score', hue='Metric', data=paper_melted, palette=['#3498db', '#e74c3c'])
plt.title('F1 Score and Matthews Correlation Coefficient by Classifier Type', fontsize=16)
plt.xlim(0.70, 0.95)  # Set x-axis limits
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Metric', loc='lower right')

# Add value labels on bars
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.3f', padding=3, fontsize=10)

plt.tight_layout()
plt.savefig('evaluation/classifier_f1_mcc_comparison.png', dpi=300, bbox_inches='tight')

print("Visualizations created at:")
print("1. evaluation/classifier_comparison_barplot.png")
print("2. evaluation/classifier_metrics_heatmap.png")
print("3. evaluation/classifier_f1_mcc_comparison.png")