import numpy as np   
import tensorflow as tf 

import model


def evaluate_msey_pehe(
        trained_model,
        test_input,
        agg_features_test,
        test_yf,
        test_ITE
    ):
    """ 
    Evaluate PEHE and MSE.
    
    Args:
        trained_model: Trained model instance.
        test_input (ndarray): Features of units in the test set.
        agg_features_test (ndarray): Aggregated results of units in the test set.
        test_yf (ndarray): Factual outcomes of units in the test set.
        test_ITE (ndarray): Ground-truth individual treatment effects of test sets.

    Returns:
        tuple: (PEHE, MSE)
    """
    pred_y1, pred_y0 = trained_model(test_input, agg_features_test, False)

    pred_ITE = pred_y1 - pred_y0
    pred_yf = Model.pre_yf(test_input, agg_features_test, False)

    pehe = np.mean((pred_ITE - test_ITE) ** 2)
    msey = np.mean((pred_yf - test_yf) ** 2)

    return pehe, msey
