from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
import numpy as np
import pandas as pd
from .metrics import NRMSE


num_pipeline = Pipeline(steps=
                        [
                            ('Imputer', KNNImputer())
                            , ('standardizer', StandardScaler())
                        ]
                       )

# # Preprocessing Pipeline for categorical variables
cat_pipeline = Pipeline(
    steps=[
        ('Imputer', SimpleImputer(strategy='most_frequent'))
        , ('Encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number))
    , (cat_pipeline, make_column_selector(dtype_include=object))
    , n_jobs=-1
)


simple_preprocessing = make_column_transformer(
    (KNNImputer(), make_column_selector(dtype_include=np.number))
    , (SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include=object))
    , n_jobs=-1
)


pipeline = Pipeline(steps = 
                    [
                    ('preproces', preprocessing)
                    ,('estimator', RandomForestRegressor(n_estimators=1000, n_jobs=-1))
                    ]
                   )

pipeline_adb = Pipeline(steps = 
                    [
                    ('preproces', preprocessing)
                    ,('estimator', AdaBoostRegressor(n_estimators=1000))
                    ]
                   )

pipeline_gb = Pipeline(steps = 
                    [
                    ('preproces', preprocessing)
                    ,('estimator', GradientBoostingRegressor(n_estimators=1000))
                    ]
                   )



def GHI_factor(energy, independent, frac = 0.5):
    """
    Calculates an optimal scaling factor between energy production and an independent variable (like GHI)
    using a Monte Carlo approach to minimize the Normalized Root Mean Square Error (NRMSE).

    Parameters:
    ----------
    energy : pandas.Series or DataFrame
    Actual energy production values with timestamp index
    
    independent : pandas.Series or DataFrame
    Independent variable values (e.g., GHI) with timestamp index
    
    frac : float, default=0.5
    Fraction used to determine the standard deviation for the normal distribution
    in the Monte Carlo simulation

    Returns:
    -------
    float or None
    The best scaling factor found that minimizes NRMSE between actual and predicted values.
    Returns None if no better factor is found than the median.

    Notes:
    -----
    - Merges energy and independent data on timestamp index
    - Filters out non-positive values from both series
    - Starts with median ratio as initial factor
    - Uses Monte Carlo simulation with normal distribution to find optimal factor
    - Prints progress information including initial factor and improvements in NRMSE
    """

    test = pd.merge(left=energy, right=independent, left_index=True, right_index=True, how='inner')
    test.columns =[ 'energy', 'independent' ]
    

    test = test[ (test.energy>0) & (test.independent>0) ]


    factors = test.energy / test.independent

    f_median = factors.median()

    fs = np.random.normal(loc=f_median, scale = f_median*frac, size = 100000)

    y_true = test.energy
    y_pred = test.independent * f_median

    nrmse = NRMSE(y_true=y_true, y_pred=y_pred)

    print( f'Starting factor = {f_median}\nBest NRMSE = {nrmse}' )

    best_factor = None

    for f in fs:
        y_true = test.energy
        y_pred = test.independent * f

        if nrmse > NRMSE(y_true=y_true, y_pred=y_pred):
            nrmse = NRMSE(y_true=y_true, y_pred=y_pred)
            print(f'A better factor is {f} with a NRMSE of {nrmse}')
            best_factor = f
            
    return best_factor
