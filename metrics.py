import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def RMSE(y_true, y_pred):
    """
    A wrapper for root_mean_squared_error
    """
     
    
    return root_mean_squared_error(y_true, y_pred)


def NRMSE(y_true, y_pred):
    """
    NRMSE is a normalized verions of RMSE meant to create a consistent scale of the output regardless of the scale of the inputs.
    RMSE's "good" values and "bad" values are dependent on the input data, thus a subject matter knowladge of the good values is required
    NRMSE avoids those requirments by scallnig the output of RMSE to be beteween 0 and 1 thus assuring a consistent scale regardless of the input scale
    
    A 0 value for NRMSE means a perfect prediction.
    
    Inputs:
    y_true : a vector of values, the scales of which are irelevent
    y_pred : a vector of values, meant to be the predicted value of each of the values in y_true
    
    Output:
    A float number between 0 and 1 representing the average penalyzed difference between y_true and y_pred where 0 represents a perfect prediction
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    
    return root_mean_squared_error(y_true, y_pred)/(y_true.max() - y_true.min())

def NMAE(y_true, y_pred):
    """
    NMAE is a normalized verions of MAE meant to create a consistent scale of the output regardless of the scale of the inputs.
    RMSE's "good" values and "bad" values are dependent on the input data, thus a subject matter knowladge of the good values is required
    NRMSE avoids those requirments by scallnig the output of RMSE to be beteween 0 and 1 thus assuring a consistent scale regardless of the input scale
    
    A 0 value for NMAE means a perfect prediction.
    
    Inputs:
    y_true : a vector of values, the scales of which are irelevent
    y_pred : a vector of values, meant to be the predicted value of each of the values in y_true
    
    Output:
    A float number between 0 and 1 representing the average penalyzed difference between y_true and y_pred where 0 represents a perfect prediction
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    
    return mean_absolute_error(y_true, y_pred)/(y_true.max() - y_true.min())


def regression_result(y_true, y_pred, alpha:float=0.3, figsize:tuple=(8,8), return_metrics = False):
    """
    regression_result displayes 3 indicators in regard to the performance of a regressor model
    1. The RMSE as detailed in the helpers.RMSE doctstring
    2. The NRMSE as detailed in the helpers.NRMSE docstring
    3. A residual plot, showing the residual values between y_true and y_pred
    
    Inputs:
    y_true  : a vector of values, the scales of which are irelevent
    y_pred  : a vector of values, meant to be the predicted value of each of the values in y_true
    alpha   : a float value detailing the alpha values to be used in the residual plot
    figsize : a tuple of two values representing the size of the figure that is to plot the residual values
    
    Outputs:
    1. NRMSE
    2. RMSE
    3. Residual Plot
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    nrmse = NRMSE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    nmae = NMAE(y_true, y_pred)
    
    
    print(f'NRMSE : {nrmse}')
    print(f'RMSE : {rmse}')
    print(f'NMAE : {nmae}')
    
    plt.figure(figsize=figsize)
    sns.lineplot(x=y_true, y=y_true, color='green')
    sns.scatterplot(x=y_true, y=y_pred, alpha=alpha)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values');
    
    if return_metrics:
        return nrmse, rmse, nmae


def plot_difs(y_true, y_pred, return_result = False):
    """
    A function for ploting the rezult of a prediction as a series of normalized values
    """
    
    result = (y_true-y_pred)/(y_true.max()-y_true.min())
    plt.figure()
    result.plot()
    
    if return_result:
        return result   


def classification_result(y_true, y_pred):

    print(classification_report(y_true=y_true
                                , y_pred=y_pred))
    
    plt.figure(figsize=(7,7))
    sns.heatmap(confusion_matrix(y_true=y_true
                                 , y_pred=y_pred
                                 , normalize='true')
                , annot=True)
