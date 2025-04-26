import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

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

def get_outlier_drop(df,target_var):
    df = df.copy()
    df['z_scores'] = (df[target_var] - df[target_var].mean()) / df[target_var].std()
    df['outlier_z'] = np.where(df['z_scores'].abs() > 3, 1, 0)
    df = df[df['outlier_z'] == 0]
    df = df.drop(columns=['z_scores', 'outlier_z'])
    return df