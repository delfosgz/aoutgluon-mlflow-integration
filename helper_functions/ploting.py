import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp

def mixed_correlation_heatmap(
    df,
    target: str = None,
    figsize=(10, 8),
    annot=True,
    encode=False,
    cmap="coolwarm_r"  # Reversed colormap for blue at 1, red at -1
):
    """
    ------------------------------------------------------------------------------
    Mixed Correlation Heatmap
    ------------------------------------------------------------------------------
    
    Interpretation:
    - Pearson: [-1, 1], linear relationship
    - Cramér's V: [0, 1], association strength between categories
    - Correlation Ratio: [0, 1], variance explained by category

    Parameters:
    - df      : pandas DataFrame
    - target  : optional column name to highlight its row
    - figsize : heatmap figure size
    - annot   : show numeric values in the heatmap
    - encode  : label-encode categorical columns if needed
    - cmap    : color palette (default blue-red diverging)
    ------------------------------------------------------------------------------
    """

    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
        rcorr = r - ((r - 1)**2) / (n - 1)
        kcorr = k - ((k - 1)**2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

    def correlation_ratio(categories, measurements):
        try:
            fcat, _ = pd.factorize(categories)
            cat_num = np.max(fcat) + 1
            y_avg = np.mean(measurements)
            numerator = sum([
                len(measurements[fcat == i]) * (np.mean(measurements[fcat == i]) - y_avg)**2 
                for i in range(cat_num)
            ])
            denominator = sum((measurements - y_avg)**2)
            return np.sqrt(numerator / denominator) if denominator != 0 else 0
        except:
            return np.nan

    df = df.copy()
    col_types = {
        col: 'num' if pd.api.types.is_numeric_dtype(dtype) else 'cat'
        for col, dtype in df.dtypes.items()
    }

    # Label columns with type
    renamed_cols = [f"{col} ({col_types[col]})" for col in df.columns]
    df.columns = renamed_cols

    if encode:
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    corr_matrix = pd.DataFrame(index=renamed_cols, columns=renamed_cols, dtype=float)

    for i in renamed_cols:
        for j in renamed_cols:
            try:
                if i == j:
                    corr = 1.0
                elif col_types[i.split(" (")[0]] == 'num' and col_types[j.split(" (")[0]] == 'num':
                    corr = df[i].corr(df[j])
                elif col_types[i.split(" (")[0]] == 'num':
                    corr = correlation_ratio(df[j], df[i])
                elif col_types[j.split(" (")[0]] == 'num':
                    corr = correlation_ratio(df[i], df[j])
                else:
                    corr = cramers_v(df[i], df[j])
            except:
                corr = np.nan
            corr_matrix.loc[i, j] = corr

    # Mask upper triangle, keep diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,  # Using reversed colormap for blue at 1 and red at -1
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.75,
        linecolor='white',
        cbar_kws={"shrink": 0.8, 'label': 'Correlation Value'},
        annot_kws={"size": 7, "weight": "bold", "color": "black"},
        vmin=-1, vmax=1  # Explicitly set color scale to range from -1 to 1
    )

    plt.title("Mixed-Type Correlation Heatmap", fontsize=16, pad=16)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # Optional: highlight the target row
    if target:
        matches = [c for c in corr_matrix.index if c.startswith(target + " (")]
        if matches:
            idx = list(corr_matrix.index).index(matches[0])
            ax.add_patch(plt.Rectangle((0, idx), len(renamed_cols), 1,
                                       fill=False, edgecolor='gold', lw=2))

    plt.tight_layout(pad=0.5)
    plt.show()

def get_target_distribution(df, target_var):
    """
    Get the distribution of the target variable.
    """
    plt.figure(figsize=(10, 3))
    plt.hist(df[target_var], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {target_var}')
    plt.xlabel(target_var)
    plt.ylabel('Frequency')
    plt.show()
    
def target_comparison_plot(train_df, test_df, target_column):
    # Extract target values
    y_train = train_df[target_column]
    y_test = test_df[target_column]

    # KS test
    ks_stat, p_value = ks_2samp(y_train, y_test)
    interpretation = "Similar" if p_value > 0.05 else "Different"

    # Define common bins
    bins = np.linspace(min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max()), 50)

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot histograms (left y-axis) with white borders on the bars
    ax1.hist(y_train, bins=bins, alpha=0.5, label='Train Histogram', color='royalblue', density=True, edgecolor='white', linewidth=1.5)
    ax1.hist(y_test, bins=bins, alpha=0.5, label='Test Histogram', color='orange', density=True, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel(target_column, fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Twin axis for CDFs (right y-axis)
    ax2 = ax1.twinx()

    sorted_train = np.sort(y_train)
    sorted_test = np.sort(y_test)

    cum_train = np.linspace(0, 1, len(sorted_train))
    cum_test = np.linspace(0, 1, len(sorted_test))

    ax2.plot(sorted_train, cum_train, color='blue', linestyle=':', linewidth=2, label='Train CDF', alpha=0.7)
    ax2.plot(sorted_test, cum_test, color='darkorange', linestyle=':', linewidth=2, label='Test CDF', alpha=0.7)
    ax2.set_ylabel('Cumulative Probability', fontsize=14)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ks_text = f"KS Statistic={ks_stat:.4f}\nP-value={p_value:.4f}\nDistributions: {interpretation}"

    # Floating legend inside plot area
    ax1.legend(
        lines_1 + lines_2 + [plt.Line2D([0], [0], color='white')], 
        labels_1 + labels_2 + [ks_text], 
        loc='upper right', 
        bbox_to_anchor=(0.95, 0.95),  # adjust position inside plot
        fontsize=12,
        frameon=True,
        framealpha=1  # solid background
    )

    # Title
    plt.title('Train vs Test Target Distribution with CDF and KS Test', fontsize=16)
    plt.tight_layout()
    plt.show()