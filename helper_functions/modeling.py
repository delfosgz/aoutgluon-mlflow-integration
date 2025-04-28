import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split_csv(input_df, target_column, output_train_csv, output_test_csv, test_size=0.2, n_bins=10, random_state=42):
    df = input_df.copy()
    df['target_bin'] = pd.qcut(df[target_column], q=n_bins, duplicates='drop')

    # Stratified split based on bins
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target_bin'],
        random_state=random_state
    )

    # Drop the auxiliary bin column
    train_df = train_df.drop(columns=['target_bin'])
    test_df = test_df.drop(columns=['target_bin'])

    # Save to CSV
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)


def get_outlier_drop(df,target_var):
    df = df.copy()
    df['z_scores'] = (df[target_var] - df[target_var].mean()) / df[target_var].std()
    df['outlier_z'] = np.where(df['z_scores'].abs() > 3, 1, 0)
    df = df[df['outlier_z'] == 0]
    df = df.drop(columns=['z_scores', 'outlier_z'])
    return df


