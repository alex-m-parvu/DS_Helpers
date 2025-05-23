import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def mm_outlier_remover(df:pd.DataFrame, ref_col:str, multiplyer:float,roll:int):
    """
    mm_outlier_remover is a filtersing function that removes the outliers from a dataframe dependent on a specified variable.
    The function calculates the moving median and the moving standard deviasion of the reference column and then based on the 
    multyplyer and roll calculates a moving span outside of which all values are considered outliers.
    The values outside this span are removed and only the data considered not to be outliers is returned
    
    Inputs :
    df           : a pandas dataframe
    ref_col      : the reference column on which the function shall determin the outliers
    mumultiplyer : the value by which the moving standard deviasion shall be multyplied with. A higher value leads to a wider span 
    a lower value leads to a narrower span
    roll         : roll represents the span of observations for which the moving median and standard deviasion is calculated
    
    Outputs:
    
    A dataframe with the mentioned outliers removed.
    """

    bounds = df[[ref_col]].rolling(roll).aggregate(['median','std'])
    bounds.columns = bounds.columns.map('_'.join).str.strip('_')
    bounds['valey'] = bounds[ref_col+'_median']- (bounds[ref_col+'_std']*multiplyer)
    bounds['peak'] = bounds[ref_col+'_median']+ (bounds[ref_col+'_std']*multiplyer)

    df = pd.merge(left=bounds, right=df, left_index=True, right_index=True)

    df['outlier'] = (df[ref_col]<df['valey']) | (df[ref_col] > df['peak'])

    df = df[~df.outlier]
    
    df = df.drop(labels=bounds.columns, axis=1)
    
    df = df.drop(labels='outlier', axis=1)
    
    return df


def density_outlier_remover(df
                            , n_components=0.95
                            , n_jobs=-1
                            , eps=0.5
                            , min_samples=5):
    
    
    """
    This is a function used to filter out outliers by removing the -1 clusters (outlier clusters) obtained from a DBSCAN clusterization
    The function creates an sklearn pipelien in the form of 
    
    [
    KNNImputer in case there are any null values present
    , A standard Scaler that is necessary for the PCA step
    , A PCA step in order to reduce dimensionality to make the process faster
    , A min max scaler in order for the DBSCAN step to function properly
    , The DBSCAN step that performs the clusterization
    ]
    
    Inputes :
    df           : a pandas dataframe
    n_components : the number of components that are to be kept in the PCA step
    n_jobs       : the number of cores to be used
    eps          : the eps value to be used in the dbscan step
    min_samples  : the min samples value to be used in the dbscan step
    """
    
    pipe = Pipeline(steps = [
        ('knnimputer', KNNImputer())
        ,('scaler', StandardScaler())
        ,('PCA', PCA(n_components=n_components))
        ,('MinMaxScaler', MinMaxScaler())
        ,('dbscan', DBSCAN(n_jobs=n_jobs, eps=eps, min_samples=min_samples))
    ])
    
    outlier_filter = pipe.fit_predict(df)
    
    return df[outlier_filter!=-1]


def quantile_outlier_remover(df:pd.DataFrame
                             , ref_col:str
                             , quantile:float = 0.05):
    
    """
    quantile_outlier_remover is a filtersing function that removes the outliers from a dataframe dependent on a specified variable.
    The function calculates the upper and lower quantile based on the quantile inpute.
    The values outside the lower and upper bound are removed and only the data considered not to be outliers is returned
    
    Inputs :
    df           : a pandas dataframe
    ref_col      : the reference column on which the function shall determin the outliers
    quantile     : the value of the lower quantile and the 1-quantile is the value of the upper quantile
    
    Outputs:
    A dataframe with the mentioned outliers removed. 
    """

    lower = df[ref_col].quantile(0.05)
    upper = df[ref_col].quantile(1- quantile)

    return df[ (df[ref_col] > lower) &  (df[ref_col] < upper)  ]



def column_filter(df, tolerance=0.5):
    """
    This is a function that fitlers out columns based on the % of missing values
    
    Inputs: 
    df        : A pandas data frame
    tolerance : the % of missing values that can pe tolerated. If the % is higher or equal to this percent then the column shall be removed
    
    Outputs:
    A pandas data frame minus the columns whose % of missing values was higher or equal to the tolerance
    """
    
    index = (df.isna().sum()/df.shape[0])>=tolerance
    drop_columns = list(index[index==True].index)
    df = df.drop(labels=drop_columns, axis=1)
    
    return df
