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

# Prepare the data for the plot
df_plot = pd.DataFrame({
    'Classifier': df['classifier'],
    'F1 Score': df['eval_f1_macro'],
    'MCC': df['eval_matthews_correlation']
})

# Create the figure
plt.figure(figsize=(10, 7))

# Set up a color palette
bar_color = '#3498db'  # Blue for bars
mcc_color = '#e74c3c'  # Red for MCC text

# Create horizontal bar chart
ax = sns.barplot(
    y='Classifier',
    x='F1 Score',
    data=df_plot,
    color=bar_color,
    alpha=0.85
)

# Add MCC scores as text annotations
for i, row in df_plot.iterrows():
    ax.text(
        row['F1 Score'] + 0.01,  # Position after the bar
        i,                       # Y-position matches the bar
        f"MCC: {row['MCC']:.3f}",  # Format the MCC value
        va='center',             # Vertically center the text
        fontsize=9,
        color=mcc_color,
        fontweight='bold'
    )

# Add value labels on the bars
for i, v in enumerate(df_plot['F1 Score']):
    ax.text(
        v - 0.05,                # Position inside the bar
        i,                       # Y-position matches the bar
        f"{v:.3f}",              # Format the F1 value
        va='center',             # Vertically center the text
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

# Add a subtitle with information about standardized parameters
plt.figtext(
    0.5, 0.96, 
    "All models trained with standardized hyperparameters: batch size=16, learning rate schedule",
    ha='center', fontsize=10, style='italic'
)

# Save in high resolution with tight layout for paper publication
plt.tight_layout(rect=[0, 0.03, 1, 0.94])  # Adjust margins to make room for the note
plt.savefig('evaluation/classifier_performance_final.png', dpi=300, bbox_inches='tight')
plt.savefig('evaluation/classifier_performance_final.pdf', format='pdf', bbox_inches='tight')

# Create a second visualization: table of exact values
plt.figure(figsize=(10, 5))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Create the table data
table_data = df_plot[['Classifier', 'F1 Score', 'MCC']].round(4)
table_data = table_data.sort_values('F1 Score', ascending=False)  # Ensure sorted order

# Create the table
table = plt.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc='center',
    cellLoc='center',
    colColours=['#f2f2f2', '#f2f2f2', '#f2f2f2']
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title('Classifier Performance Metrics (Sorted by F1 Score)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('evaluation/classifier_metrics_table.png', dpi=300, bbox_inches='tight')
plt.savefig('evaluation/classifier_metrics_table.pdf', format='pdf', bbox_inches='tight')

print("Final visualizations created at:")
print("1. evaluation/classifier_performance_final.png")
print("2. evaluation/classifier_performance_final.pdf")
print("3. evaluation/classifier_metrics_table.png")
print("4. evaluation/classifier_metrics_table.pdf")