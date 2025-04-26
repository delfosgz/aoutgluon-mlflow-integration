import pandas as pd
import numpy as np


def evalueate_model(y_test, y_test_pred):
    result = pd.DataFrame(index =y_test.index)
    result['REAL'] = y_test
    result['PREDICTION'] = y_test_pred
    
    mae = mean_absolute_error(result['REAL'], result['PREDICTION'])
    mse = mean_squared_error(result['REAL'], result['PREDICTION'])
    rmse = np.sqrt(mse)
    