import pandas as pd
import numpy as np

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    A custom metric for the German credit dataset
    '''
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
    costs = compute_costs(solution['LoanAmount'])
    y_true = solution['Risk']
    y_pred = submission['Risk']
    loss = (y_true=='Risk') * custom_weight['Risk'] *\
               ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) +\
            (y_true=='No Risk') * custom_weight['No Risk'] *\
               ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
    return loss.mean()

## Other version with the correct loan amounts
def calculate_custom_loss(y_true, y_pred, loan_amounts):
    """
    Calculate custom loss based on the competition metric

    Parameters:
    -----------
    y_true : array-like
        True labels (0 for 'No Risk', 1 for 'Risk')
    y_pred : array-like
        Predicted labels (0 for 'No Risk', 1 for 'Risk')
    loan_amounts : array-like
        Loan amounts for each prediction

    Returns:
    --------
    float
        Calculated loss value
    """
    # Convert numeric labels to string format if needed
    y_true_str = np.where(y_true == 1, 'Risk', 'No Risk')
    y_pred_str = np.where(y_pred == 1, 'Risk', 'No Risk')

    # Real world vs training set proportions
    real_prop = {'Risk': 0.02, 'No Risk': 0.98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}

    # Calculate weights to adjust for class imbalance
    custom_weight = {
        'Risk': real_prop['Risk'] / train_prop['Risk'],
        'No Risk': real_prop['No Risk'] / train_prop['No Risk']
    }

    # Calculate costs for each prediction
    costs = {
        'Risk_No Risk': 5.0 + 0.6 * loan_amounts,
        'No Risk_No Risk': 1.0 - 0.05 * loan_amounts,
        'Risk_Risk': np.ones_like(loan_amounts),
        'No Risk_Risk': np.ones_like(loan_amounts)
    }

    # Calculate losses for each sample
    loss = np.zeros_like(loan_amounts, dtype=float)

    # Risk samples
    risk_mask = (y_true_str == 'Risk')
    loss[risk_mask] = custom_weight['Risk'] * (
        (y_pred_str[risk_mask] == 'Risk') * costs['Risk_Risk'][risk_mask] +
        (y_pred_str[risk_mask] == 'No Risk') * costs['Risk_No Risk'][risk_mask]
    )

    # No Risk samples
    no_risk_mask = (y_true_str == 'No Risk')
    loss[no_risk_mask] = custom_weight['No Risk'] * (
        (y_pred_str[no_risk_mask] == 'Risk') * costs['No Risk_Risk'][no_risk_mask] +
        (y_pred_str[no_risk_mask] == 'No Risk') * costs['No Risk_No Risk'][no_risk_mask]
    )

    return loss.mean()

def compute_costs(LoanAmount):
    return({'Risk_No Risk': 5.0 + .6 * LoanAmount, 'No Risk_No Risk': 1.0 - .05 * LoanAmount,
            'Risk_Risk': 1.0, 'No Risk_Risk': 1.0})