import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

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

def evaluate_model(y_test,y_test_pred):
    result = pd.DataFrame(index=y_test.index)
    result['REAL'] = y_test
    result['PREDICTION'] = y_test_pred
    
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2  = explained_variance_score(y_test, y_test_pred)
    
    return result, mse, rmse, mae, r2

def pop_all_ocurrences(d,key):
    if isinstance(d,dict):
        d.pop(key, None)
        for sub_dict in d.values():
            pop_all_ocurrences(sub_dict, key)
           

def get_experiment(experiment_name, experiment_path):
    if mlflow.get_experiment_by_name(experiment_path +experiment_name) is not None:
        experiment = mlflow.get_experiment_by_name(f'{experiment_path}{experiment_name}')
        print (f'The existing experiment {experiment_name} has been set up successfully')
    else:
        mlflow.create_experiment(experiment_path + experiment_name)
        experiment = mlflow.get_experiment_by_name( f'{experiment_path}{experiment_name}')
        print(f'The new experiment {experiment_name} has been set up successfully')
    
    return experiment.experiment_id




        