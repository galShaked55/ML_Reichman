import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    # Get the cov matrix for feature x and label y
    # Default degree of freedom for np.cov is 1, modify it to match the course's definitions
    cov_x_y = np.cov(x, y, ddof=0)

    # Get standard deviation of feature x and label y
    std_x = np.std(x)
    std_y = np.std(y)

    # Calc the pearson correlation by dividing
    r = (cov_x_y[0, 1])/(std_x * std_y)

    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    # X = pd.DataFrame(X)
    # X = X.drop(columns=[col for col in ['id', 'date'] if col in X], errors='ignore')
    best_features = []

    # dropping the non numeric columns in order to avoid computing pearson correlation on it
    X_numeric = X.select_dtypes(include=['number'])

    cor_per_feature = {}

    # Iterate over the columns names and calc it's pearson correlation with the labels column
    for column in X_numeric.columns:
        correlation = pearson_correlation(X_numeric[column], y)
        cor_per_feature[column] = abs(correlation)

    # Get the sorted by correlation's magnitude list of features
    sorted_features = sorted(cor_per_feature.items(), key=lambda item: item[1], reverse=True)

    # Attain the n_features max correlated
    for column, correlation in sorted_features[:n_features]:
        best_features.append(column)

    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        #   Set random seed
        np.random.seed(self.random_state)

        #  Apply bias trick on X
        X = self.apply_bias_trick(X)

        # Initialize theta vector with random weights
        self.theta = np.random.random(X.shape[1])

        self.thetas.append(self.theta)
        self.Js.append(self.calc_cost(X, y))

        # Learn self.theta
        self.gradient_descent(X, y)

    def gradient_descent(self, X, y):
        """
        Learns the best theta using gradient descent.
        Input:
        - X: Input data (m instances over n features).
        - y: True labels (m instances).
        """

        # Perform n_iter steps of gradient descent or reach redundant change in cost
        for i in range(self.n_iter):
            inner_prod = np.dot(X, self.theta)
            cur_hypo = self.calc_sigmoid(inner_prod)
            gradient = np.dot(X.T, (cur_hypo - y))
            self.theta = self.theta - self.eta * gradient

            # Calculate the cost
            # and add it and the corresponding theta to
            # self.Js and self.thetas respectively
            cur_cost = self.calc_cost(X, y)
            self.Js.append(cur_cost)
            self.thetas.append(self.theta.copy())

            if i > 1 and abs(self.Js[i] - cur_cost) < self.eps:
                break



    def apply_bias_trick(self, X):
        """
            Applies the bias trick to the input data.

            Input:
            - X: Input data (m instances over n features).

            Returns:
            - X: Input data with an additional column of ones in the
                zeroth position (m instances over n+1 features).
            """
        # Add column of ones at the beginning for bias
        ones_column = np.ones((X.shape[0], 1))
        return np.hstack((ones_column, X))

    def calc_sigmoid(self, dot_prod):
        """
        Returns: the value of the sigmoid function for a given dot product result
        """

        return 1.0 / (1.0 + np.exp(-1.0 * dot_prod))

    def calc_cost(self, X, y):
        """
        Compute the current cost.
        Where current cost is the cost according to the current self.theta
        Returns: current cost
        """
        inner_prod = np.dot(X, self.theta)
        cur_hypo = self.calc_sigmoid(inner_prod)
        # Adds small value to avoid calculating log of 0
        cost = (-1.0 / len(y)) * (np.dot(y.T, np.log(cur_hypo + 1e-6)) + np.dot((1-y).T, np.log(1 - cur_hypo + 1e-6)))
        return cost

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        X = self.apply_bias_trick(X)
        # Determine the Logistic regression model's prediction for X
        inner_prod = np.dot(X, self.theta)
        h = self.calc_sigmoid(inner_prod)
        # Round to zero \ 1 depends on whether it's closer to 1 or to 0
        preds = np.round(h).astype(int)
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0
    accuracies = []

    # set random seed
    np.random.seed(random_state)

    # Shuffle the data
    shuffled_data = np.random.permutation(X.shape[0])
    fold_data = np.array_split(shuffled_data, folds)

    for i in range(folds):
        # Set the indices for the k-1 folds 'training folds' and 1 for the 'validation fold'
        train_indices = np.concatenate(fold_data[:i] + fold_data[i+1:])
        validation_indices = fold_data[i]

        X_train = X[train_indices]
        y_train = y[train_indices]

        # Train a model
        algo.fit(X_train, y_train)

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        # Check the trained model performance on the validation set
        predicted_y = algo.predict(X_validation)
        accuracy = np.mean(predicted_y == y_validation)
        accuracies.append(accuracy)

    # Calc the mean accuracy of the k CV iterations
    cv_accuracy = np.mean(accuracies)

    return cv_accuracy


def norm_pdf(x, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """

    return (np.exp(np.square((x - mu)) / (-2 * np.square(sigma)))) / (np.sqrt(2 * np.pi * np.square(sigma)))


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # Initial guesses for parameters.
    def init_params(self, data):
        """
        Initialize distribution params
        """
        # Choose k instances in random to be the initial k mus for the k gaussians
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes].reshape(self.k)
        # Choose k sigmas to be the initials sigmas for the k guassians
        self.sigmas = np.random.random_integers(self.k)
        # Distribute the weights uniformly
        self.weights = np.ones(self.k) / self.k

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # Multiply the w_j by the probability to observe x under gaussian_j's model
        weights_mul_probs = self.weights * norm_pdf(data, self.mus, self.sigmas)

        # Sum the weighted probabilities to observe x under each gaussian
        complete_probs = np.sum(weights_mul_probs, axis=1, keepdims=True)

        # Divide to attain r(x,k) for every instance x and gaussian k
        self.responsibilities = weights_mul_probs / complete_probs

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # Set the new w_j to be the mean over all instances of r(x,G_j)
        self.weights = np.mean(self.responsibilities, axis=0)
        # Set the new mu_j to be the weighted (weight by responsibility) mean over all instances x
        # and divide by the new w_j
        self.mus = np.sum(self.responsibilities * data.reshape(-1, 1), axis=0) / np.sum(self.responsibilities, axis=0)
        # Compute the new variances according to the new mu_s and weighted by the responsibilities
        vs = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(vs / self.weights)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        # Init tha parameters to be learned
        self.init_params(data)
        self.costs.append(self.cost(data))

        # Expectation and maximization to discover
        # the hidden RV's distribution parameters: weights.
        # And each guassian's [1,k] mu and sigma
        # Perform n_iters of EM or reach a redundant improvement in cost(eps)
        for i in range(self.n_iter):
            cost = self.cost(data)
            self.costs.append(cost)
            self.expectation(data)
            self.maximization(data)
            if self.costs[-1] - cost < self.eps:
                if self.costs[-1] > cost:
                    self.costs.append(cost)
                    break
            self.costs.append(cost)

    def cost(self, data):
        """
        Calculates the cost of the current model according to the -log-likelihood
        cost function provided in the notebook
        Args:
            data: The data we want to compute the cost over.

        Returns: the total cost over the data for the current's EM obj sigmas,mus, weights

        """
        sum_cost = 0
        cost = self.weights * norm_pdf(data, self.mus, self.sigmas)
        for i in range(len(data)):
            sum_cost = sum_cost + cost[i]
        return -1.0 * np.sum(np.log(sum_cost))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """

    pdf = None
    pdf = np.sum(weights * norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)

    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior_probs_dict = None # A dictionary of class: prior_P(class)
        self.likelihood_models_dict = None # A dictionary of dictionaries of the form - class_i : {feature_j : P(x_j|class_i)}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # Calculate the prior probability for each class
        self.init_prior_dict(y)
        # Create the likelihoods dictionary and fills it with the likelihood models for each feature given each class
        self.init_likelihoods_dict(X,y)

    def init_prior_dict(self, y):
        """
        Initializes a dictionary of the form: class: prior_P(class)
        Args:
            y: array-like of labels

        """
        self.prior_probs_dict = {class_Label: len(y[y == class_Label]) / len(y) for class_Label in np.unique(y)}

    def init_likelihoods_dict(self, X, y):
        """
        Initializes a dictionary of the form class_i : {feature_j : P(x_j|class_i)}
        Args:
            X: array-like dataset
            y: array-like of labels
        """
        self.likelihood_models_dict = {class_Label: {feature: EM(self.k) for feature in range(X.shape[1])} for class_Label in np.unique(y)}
        for label in self.likelihood_models_dict.keys():
            for feature in self.likelihood_models_dict[label].keys():
                self.likelihood_models_dict[label][feature].fit(X[y == label][:, feature].reshape(-1, 1))

    def get_prior(self, class_label):
        """
        Returns the prior probability of the input class label
        """
        return self.prior_probs_dict[class_label]

    def get_likelihood(self, X, class_label):
        """
        Calculates the likelihood to observe instance X given the class_label class
        under the naive bayes assumption.
        """
        likelihood = 1
        for feature in range(len(X)):
            weights, mus, sigmas = self.likelihood_models_dict[class_label][feature].get_dist_params()
            gmm = gmm_pdf(X[feature], weights, mus, sigmas)
            likelihood = likelihood * gmm
        return likelihood


    def get_posterior(self, X, class_label):
        """
        # Calculate the posterior according to the formula.
        """
        return self.get_prior(class_label) * self.get_likelihood(X, class_label)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]

        Returns
        -------
        preds : array-like, shape = [n_examples]
          Predicted class labels.
        """
        # Determine the prediction for each instance as the max posterior probability over the classes
        preds = [max([(self.get_posterior(instance, class_Label), class_Label) for class_Label in self.prior_probs_dict.keys()],
                     key=lambda t: t[0])[1] for instance in X]
        return np.array(preds)


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ['blue', 'red']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Learn LOr model on the given dataset calculate its accuracy
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)
    lor_train_preds = lor_model.predict(x_train)
    lor_test_preds = lor_model.predict(x_test)
    lor_train_acc = np.mean(y_train == lor_train_preds)
    lor_test_acc = np.mean(y_test == lor_test_preds)

    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=lor_model, title="Logistic Regression Decision Boundaries")

    # Learn Naive Bayes model on the given dataset calculate its accuracy
    NB_Gaussian = NaiveBayesGaussian(k=k)
    NB_Gaussian.fit(x_train, y_train)
    NB_train_preds = NB_Gaussian.predict(x_train)
    NB_test_preds = NB_Gaussian.predict(x_test)
    bayes_train_acc = np.mean(y_train == NB_train_preds)
    bayes_test_acc = np.mean(y_test == NB_test_preds)

    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=NB_Gaussian, title="Naive Bayes Gaussian Decision Boundaries")

    # Plotting the cost as a function of the gradient descent iterations
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lor_model.Js)), lor_model.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations for Logistic Regression Model')
    plt.grid(True)
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''

    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    dataset_a_means = [(1, 1, 1), (3, 3, 3), (6, 6, 6), (9, 9, 9)]
    dataset_a_std = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 4
    dataset_a_features, dataset_a_labels = generate_3d_data(200, 4, dataset_a_means, dataset_a_std, [0, 1, 1, 0])
    dataset_b_means = [(0, 1, 0), (0, 2, 0)]
    dataset_b_std = [[[5, 5, 5], [5, 5, 5], [5, 5, 5]]]*2
    dataset_b_features, dataset_b_labels = generate_3d_data(200, 2, dataset_b_means, dataset_b_std, [0, 1])

    plot_dataset(dataset_a_features, dataset_a_labels, question=1)
    plot_dataset(dataset_b_features,  dataset_b_labels, question=2)

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }


def generate_3d_data(n_instances_per_group, n_groups, means, covariances, classes):

    dataset_features = []
    dataset_labels = []

    for i in range(n_groups):
        # Define the mean and covariance matrix for the multivariate normal distribution
        mean = means[i]
        covariance = covariances[i]

        # Generate the samples for this group
        samples = multivariate_normal.rvs(mean=mean, cov=covariance, size=n_instances_per_group)

        # Append samples to the dataset_features and the corresponding labels to dataset_labels
        dataset_features.append(samples)
        dataset_labels.extend([classes[i]] * n_instances_per_group)

    dataset_features = np.vstack(dataset_features)
    dataset_labels = np.array(dataset_labels)

    return dataset_features, dataset_labels


def plot_dataset(features, labels, question):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    colors = ['red', 'blue']
    markers = ['o', '^']

    for class_value in np.unique(labels):
        class_points = features[labels == class_value]
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2],
                   c=colors[int(class_value)], marker=markers[int(class_value)], label=f'Class {int(class_value)}')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    if question == 1:
        plt.title('Dataset A - Naive Bayes > LOr')
    else:
        plt.title('Dataset B - Naive bayes < LOr')
    plt.show()


# Functions used for testing generate_datasets()


def merge_features_labels(features, labels):
    labels = labels.reshape(-1, 1)
    merged_dataset = np.hstack((features, labels))
    return np.array(merged_dataset)


def shuffle_split_dataset(dataset, train_fraction=0.75):
    """
    Shuffles a dataset randomly and splits it into a training set and a test set.

    """
    np.random.shuffle(dataset)

    split_idx = int(len(dataset) * train_fraction)

    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]

    return train_set, test_set


def separate_features_labels(dataset):
    """
    Separates the features and labels from a dataset where the last column is known to be the labels.
    """
    features = dataset[:, :-1]
    labels = dataset[:, -1]

    return features, labels


def model_evaluation_without_plotting(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    '''
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Learn LOr model on the given dataset calculate its accuracy
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)
    lor_train_preds = lor_model.predict(x_train)
    lor_test_preds = lor_model.predict(x_test)
    lor_train_acc = np.mean(y_train == lor_train_preds)
    lor_test_acc = np.mean(y_test == lor_test_preds)

    # Learn Naive Bayes model on the given dataset calculate its accuracy
    NB_Gaussian = NaiveBayesGaussian(k=k)
    NB_Gaussian.fit(x_train, y_train)
    NB_train_preds = NB_Gaussian.predict(x_train)
    NB_test_preds = NB_Gaussian.predict(x_test)
    bayes_train_acc = np.mean(y_train == NB_train_preds)
    bayes_test_acc = np.mean(y_test == NB_test_preds)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}





