###### Your ID ######
# ID1: 206395139
# ID2: 206998635
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Mean - normalization on the features
    means_x = X.mean(axis=0)
    nominator_x = X - means_x
    maxs_x = X.max(axis=0)
    mins_x =X.min(axis=0)
    denominator_x = maxs_x-mins_x
    X = nominator_x/denominator_x

    # Mean - normalization on the labels
    means_y = y.mean(axis=0)
    nominator_y = y - means_y
    maxs_y = y.max(axis=0)
    mins_y = y.min(axis=0)
    denominator_y = maxs_y - mins_y
    y = nominator_y / denominator_y

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # Add column of ones at the beginning for bias
    bias = np.ones(len(X))
    X = np.c_[bias, X]
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = len(X)  # Get the amount of instances
    product_results = X.dot(theta)
    subs_results = (product_results - y) ** 2
    J = subs_results.sum() / (2 * m)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()    # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = len(X)

    for i in range(num_iters):  # Perform num_iters steps of gradient descent
        theta_cost = compute_cost(X, y, theta)
        J_history.append(theta_cost)
        product_results = X.dot(theta)
        subs_results = (product_results - y)
        gradient = (X.T).dot(subs_results)
        theta -= theta - alpha * gradient / m

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    X_tX = X.T.dot(X)   # Compute the product matrix of X and X transpose
    X_tX_inv = np.linalg.inv(X_tX)  # Find it's inverse matrix if invertible
    pinv_theta = X_tX_inv.dot(X.T).dot(y)   # Compute the theta that minimizes MSE
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()    # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = len(X)

    for i in range(num_iters):  # Perform num_iters steps of gradient descent or reach minor difference in cost:
        product_results = X.dot(theta)
        subs_results = (product_results - y)
        gradient = (X.T).dot(subs_results)
        theta = theta - (alpha / m) * gradient
        theta_cost = compute_cost(X, y, theta)
        J_history.append(theta_cost)
        if i > 0 and (J_history[i-1] - J_history[i] < 1e-8):
            break

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}     # {alpha_value: validation_loss}
    for i in range(len(alphas)):
        init_theta = np.ones(X_train.shape[1])
        cur_theta = efficient_gradient_descent(X_train, y_train, init_theta, alphas[i], iterations)[0]
        alpha_dict[alphas[i]] = compute_cost(X_val, y_val, cur_theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = [0]
    best_feature = -1
    apply_bias_trick(X_train)
    apply_bias_trick(X_val)
    n = X_train.shape[1]
    for i in range(5):  # Till we attain the set of 5 best features
        min_cost = float('inf')
        for j in range(0, n):
            init_theta = np.ones(i+2)
            if j not in selected_features:
                selected_features.append(j)
                theta_by_feature_j = efficient_gradient_descent(X_train[:, selected_features], y_train, init_theta,
                                                                best_alpha, iterations)[0]
                cur_cost = compute_cost(X_val[:, selected_features], y_val, theta_by_feature_j)
                if cur_cost < min_cost:
                    min_cost = cur_cost
                    best_feature = j
                selected_features.remove(j)
        selected_features.append(best_feature)

    selected_features.remove(0)
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    n = df_poly.shape[1]    # The original number of features
    columns_names = df_poly.columns
    empty_df = pd.DataFrame()
    for i in range(n):
        cur_feature = columns_names[i]
        for j in range(i,n):
            multiplied_feature = columns_names[j]
            if i == j:
                new_feature_name = f"{cur_feature}^2"
            else:
                new_feature_name = f"{cur_feature}*{multiplied_feature}"
            product_values = df_poly[cur_feature]*df_poly[multiplied_feature]
            empty_df[new_feature_name] = product_values
        df_poly = pd.concat([df_poly, empty_df], axis=1)
        empty_df = pd.DataFrame()

    return df_poly
