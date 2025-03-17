import numpy as np
import matplotlib.pyplot as plt
import queue

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    # Creating variables for calculate of the gini.
    gini = 0.0
    info = data[:, -1]
    param, count = np.unique(info, return_counts=True)

    # Calculating the gini.
    prob = count / len(data)
    gini = 1 - np.sum(prob ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    # Creating variables for calculate of the gini.
    entropy = 0.0
    info = data[:, -1]
    param, count = np.unique(info, return_counts=True)

    # Calculating the entropy.
    prob = count / len(data)
    entropy = (-1) * np.sum(prob * np.log2(prob))
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        param, count = np.unique(self.data.T[-1], return_counts=True)
        dict_param_count = dict(zip(param, count))

        # Finding the prediction of the node.
        pred = max(dict_param_count, key=dict_param_count.get)
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """

        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """

    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset

        # If the gain_ratio is True then we need to compute information gain and split_in_information by entropy.
        if self.gain_ratio == True:
            self.impurity_func = calc_entropy

        # Creating the variables that we need for the calculating
        split_in_information = 0
        impurity_data = self.impurity_func(self.data)
        impurity_attribute = 0
        param = np.unique(self.data.T[feature])

        # Going over the types of parameters that are in the feature and handle them according to the calculation.
        for val in param:
            groups[val] = self.data[self.data[:, feature] == val]
            size_attribute = len(groups[val]) / len(self.data)
            split_in_information += size_attribute * np.log2(size_attribute)
            impurity_attribute += size_attribute * self.impurity_func(groups[val])

        # If gain_ration is False we compute the goodness of split according to the formula.
        if self.gain_ratio == False:
            goodness = impurity_data - impurity_attribute

        # If gain_ration is True we compute the goodnes by division between information gain and split in information.
        else:
            information_gain = impurity_data - impurity_attribute

            # To avoid division by 0.
            if split_in_information == 0:
                return 0, groups

            # Adjust the variable according to the formula.
            split_in_information = split_in_information * (-1)
            goodness = information_gain / split_in_information
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Checking if we have reached the maximum depth of the tree.
        if self.depth == self.max_depth:
            self.terminal = True
            return

        # Finding the best feature to split by and take his data by goodness_of)split function
        self.feature = self.index_best_feature(self.data, self.impurity_func, self.gain_ratio)
        _, subdata = self.goodness_of_split(self.feature)

        # Checking if the condition of chi pruning is exists and if we will have more than 1 child.
        if len(subdata) > 1 and check_chi(self,subdata, self.chi):
            for key, group in subdata.items():
                child = DecisionNode(data=group,impurity_func=self.impurity_func, depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth,
                                     gain_ratio=self.gain_ratio)
                self.add_child(child, key)
            return

        else:
            self.terminal = True
            return

    def index_best_feature(self, data, impurity_func, gain_ratio):
        """
        finds the index of the best feature to split by.
        Input:
        - data: any dataset where the last column holds the labels.
        - impurity_func: The impurity function that should be used as the splitting criteria.
        - gain_ratio: goodness of split or gain ratio flag.

        Returns:
        - best_feature: the index of the criteria that is the best to split by.
        """
        # Creating the variables that help us find the index of the best feature to split by.
        best_feature = None
        best_feature_goodness = 0

        # Going over all the features and calculate the goodness of split by those features
        for index in range(len(data[0]) - 1):
            current_feature_goodness, _ = self.goodness_of_split(index)

            # Finding the index of the feature that gives the highest goodness of split.
            if current_feature_goodness > best_feature_goodness:
                best_feature_goodness = current_feature_goodness
                best_feature = index
        return best_feature

def check_chi(node, subdata, chi):
    """
    This function checks if all the conditions of doing chi pruning are exists.

    Input:
    - node: the tree itself
    - subdata: the data we have after we calculate the goodness of split by the best feature to split by.
    - chi: the chi value that we got in creating the tree.

    Returns:
        True or False if the condition that chi value that we calculate is equal or bigger than the value from the chi table
            """
    # Check if the chi value is 1 then no need to preform chi pruning.
    if chi == 1:
        return True

    # calculate the chi value by the formula and check this value with the value from chi table.
    chi_val = chi_square_compute(node.data, subdata)
    deg_of_freedom = len(subdata) - 1
    chi_val_from_table = chi_table[deg_of_freedom][chi]
    return chi_val >= chi_val_from_table

def dict_label_number(labels):
    """
    The function creates a dictionary of labels of a column and there amount in the column.

    Input:
    -labels : An array of the labels.

    Returns:
    - dictionary: A dictionary that the key is the label and the value is the amount of the label
    """

    # For each label count how much this label appears in the array of the label.
    return {label: np.sum(labels == label) for label in np.unique(labels)}

def chi_square_compute(data,subdata):
    """
    calculates the chi square according to the formula.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: The index of the feature that we will calculate the chi square according to it.

    Returns:
    - chi_square: the chi square of the dataset according to the feature.
    """
    # Creating variables that we will use in the formula.
    chi_square = 0
    size = len(data[:, -1])
    final_label_count = dict_label_number(data[:, -1])
    for feature_val, sub in subdata.items():
        sub_size = len(sub)
        sub_label_count = dict_label_number(sub[:, -1])
        for label, count in final_label_count.items():

            # Calculate the parameters in the formula.
            expected = sub_size * (count / size)
            observed = sub_label_count.get(label, 0)  # Default to 0 if label is not found

            # Check if expected is zero to avoid division by zero error
            if expected == 0:
                chi_square_contribution = 0  # Set contribution to 0 if expected is zero
            else:
                # Calculate the chi square according to the formula.
                chi_square_contribution = ((observed - expected) ** 2) / expected

            # Update the chi square sum
            chi_square += chi_square_contribution
    return chi_square

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        # Creating the root of the tree.

        self.root = DecisionNode(data=self.data,impurity_func=self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        # Using a queue to insert the nodes into the tree in order.
        q = queue.Queue()
        q.put(self.root)

        # A loop that runs as long as there is a node that can be inserted to the tree.
        while not q.empty():
            node = q.get()

            # Check if the data is already arranged
            if len(np.unique(node.data)) == 1:
                node.terminal = True
                continue

            # Give the node the index of the feature that is the best feature to split by.
            node.feature = DecisionNode.index_best_feature(node, data=node.data, impurity_func=self.impurity_func, gain_ratio=self.gain_ratio)

            # Check if there is no feature to split by.
            if node.feature == None:
                node.terminal = True
                continue

            # Split the node by the impurity function.
            node.split()

            # Insert the children of the node to the queue that we will insert them to the tree.
            for child in node.children:
                q.put(child)


    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        node = self.root
        found_child = True

        # A loop that runs as long as we in the tree and we don't reach to a leaf.
        while (not node.terminal and found_child == True):
            found_child = False

            # A loop that runs on the children of the node.
            for child in node.children:

                # Checking if the information of the instance is the same as the data of a node in the tree.
                if (child.data[:, node.feature] == instance[node.feature]).all():
                    node = child
                    found_child = True
                    break
        # Taking the prediction of the node.
        pred = node.pred
        return pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        correct = 0

        # Going over all the instances in the dataset.
        for instance in dataset:

            # Taking the prediction of the instance according to predict function and the actual value.
            pred = self.predict(instance)
            pred_check = instance[-1]

            # Checking if what we predict to the instance is the actual value.
            if pred == pred_check:
                correct += 1

        # Calculate the accuracy of our tree.
        accuracy = (correct / (len(dataset)))
        return accuracy
        
    def depth(self):
        return self.root.depth()

def calc_depth(node):
    """
    Calculate the depth of the tree by adding each depth of a node to a list and returns the maximum value
    of the depth that we got on the list. By doing this we basically get a node and then go down to the leaf to get the
    depth of the whole tree.

    Input:
    - node: a node from a tree.

    Output: the depth of the tree.
    """
    # Check if we got to a leaf.
    if node.terminal:
        return node.depth

    # Creating a list that will contain the depth of the node's children.
    depths = []

    # Going over all the children of the node.
    for child in node.children:
        child_depth = calc_depth(child)
        depths.append(child_depth)
    return max(depths)

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Creating a tree according to the max_depth we got and compute the accuracy of the tree according to our data.
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depths = []
    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for val in p_values:

        # Creating a tree that his chi value is according to the p_values and then we calculate the accuracy and the depth of the tree.
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=val, gain_ratio=True)
        tree.build_tree()
        accurcy_train = tree.calc_accuracy(X_train)
        accurcy_test = tree.calc_accuracy(X_test)
        chi_training_acc.append(accurcy_train)
        chi_testing_acc.append(accurcy_test)
        tree_depth = calc_depth(tree.root)
        depths.append(tree_depth)

    return chi_training_acc, chi_testing_acc, depths


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






