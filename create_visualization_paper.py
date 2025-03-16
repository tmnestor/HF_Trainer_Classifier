import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Load the data
data_path = Path("results/classifier_results_summary.csv")
df = pd.read_csv(data_path)

# Rename classifiers to be more descriptive and paper-friendly
classifier_rename = {
    'standard': 'Standard Linear',
    'custom': 'Custom MLP',
    'bilstm': 'BiLSTM',
    'attention': 'Self-Attention',
    'cnn': 'CNN',
    'fourier_kan': 'FourierKAN',
    'wavelet_kan': 'WaveletKAN',
    'mean_pooling': 'Mean Pooling',
    'combined_pooling': 'Combined Pooling'
}
df['classifier'] = df['classifier'].map(classifier_rename)

# Sort the data by F1 macro score (descending)
df = df.sort_values(by='eval_f1_macro', ascending=False)

# Create a figure focused on the key metrics
plt.figure(figsize=(9, 6))
metrics = ['eval_f1_macro', 'eval_matthews_correlation']
labels = ['F1 Score (Macro)', 'Matthews Correlation Coefficient']

# Set up a color palette
colors = ['#3498db', '#e74c3c']

# Extract and prep the data
df_plot = df[['classifier'] + metrics]
df_plot.columns = ['Classifier', 'F1 Score', 'MCC']

# Create horizontal bar chart with error bars
plt.figure(figsize=(10, 7))
ax = sns.barplot(
    y='Classifier',
    x='F1 Score',
    data=df_plot,
    color=colors[0],
    alpha=0.85,
    label='F1 Score'
)

# Add MCC scores as text annotations
for i, row in df_plot.iterrows():
    ax.text(
        row['F1 Score'] + 0.01,  # Slightly offset from the bar end
        i,                        # Y-position matches the bar
        f"MCC: {row['MCC']:.3f}",   # Format the MCC value
        va='center',              # Vertically center the text
        fontsize=9,
        color=colors[1],
        fontweight='bold'
    )

# Add value labels on the bars
for i, v in enumerate(df_plot['F1 Score']):
    ax.text(
        v - 0.05,                  # Position inside the bar
        i,                         # Y-position matches the bar
        f"{v:.3f}",                # Format the F1 value
        va='center',               # Vertically center the text
        color='white',
        fontweight='bold',
        fontsize=9
    )

# Configure the plot
plt.title('Classifier Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('F1 Score (Macro)', fontsize=12)
plt.xlim(0.70, 1.0)  # Set x-axis limits for better visualization
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Add annotation to explain MCC
plt.figtext(
    0.5, 0.01, 
    "Note: Matthews Correlation Coefficient (MCC) values shown in red next to each bar.",
    ha='center', fontsize=9, style='italic'
)

# Save in high resolution with tight layout for paper publication
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust margins to make room for the note
plt.savefig('evaluation/classifier_performance_paper.png', dpi=300, bbox_inches='tight')
plt.savefig('evaluation/classifier_performance_paper.pdf', format='pdf', bbox_inches='tight')

print("Paper-optimized visualization created at:")
print("1. evaluation/classifier_performance_paper.png")
print("2. evaluation/classifier_performance_paper.pdf")