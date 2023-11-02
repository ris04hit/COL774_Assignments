from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DTNode:
    def __init__(self, depth: int, is_leaf: bool = False, value = -1, column: int = None, type: int = 0, threshold = 0):
        #to split on column
        self.depth = depth

        #add children afterwards
        self.children = {}

        #if leaf then also need value
        self.is_leaf = is_leaf
        self.value = value
        if not self.is_leaf:
            self.column = column
            self.type = type
            if self.type == 1:
                self.threshold = threshold

    def get_children(self, X: np.ndarray):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
            None: if children does not exist or cant be assigned
        '''
        if self.is_leaf:
            return None
        if self.type == 1:
            if X[self.column] < self.threshold:
                return self.children[0]
            else:
                return self.children[1]
        else:
            if X[self.column] in self.children:
                return self.children[X[self.column]]
            else:
                return None

    def print_node(self, tree: bool = False):
        space = '\t'*self.depth
        if self.is_leaf:
            print(f'{space}depth = {self.depth}\n{space}value = {self.value}\n')
        elif self.type == 0:
            print(f'{space}depth = {self.depth}\n{space}value = {self.value}\n{space}column = {self.column}')
        else:
            print(f'{space}depth = {self.depth}\n{space}threshold = {self.threshold}\n{space}column = {self.column}')
        if tree:
            for val in self.children:
                self.children[val].print_node(True)

def entropy(col: np.ndarray):       # returns entropy of distribution
    total_ct = col.shape[0]
    if not total_ct:
        return 0
    count = {}
    for val in col.tolist():
        if val not in count:
            count[val] = 0
        count[val]+=1
    ent = 0
    for val in count:
        ent -= (count[val]/total_ct)*np.log2(count[val]/total_ct)
    return ent

def avg_child_entropy(X: np.ndarray, Y: np.ndarray, type_val: int):        # returns entropy of child split
    total_ct = X.shape[0]
    ent = 0
    if type_val == 0:
        count = {}
        for val in X.tolist():
            if val not in count:
                count[val] = 0
            count[val]+=1
        for val in count:
            ent += (count[val]/total_ct)*entropy(Y[np.where(X == val)])
    else:
        threshold = np.median(X)
        Y_split1 = Y[np.where(X < threshold)]
        Y_split2 = Y[np.where(X >= threshold)]
        ent += (Y_split1.shape[0]/total_ct)*entropy(Y_split1)
        ent += (Y_split2.shape[0]/total_ct)*entropy(Y_split2)
    return ent

def best_column(X: np.ndarray, Y: np.ndarray, types: list):       # returns the best column for split, -1 if no split
    col_num = len(types)
    best_col = -1
    max_gain = 0
    parent_entropy = entropy(Y.reshape(-1))
    for ind in range(col_num):
        type_val = types[ind]
        ent = avg_child_entropy(X[:, ind].reshape(-1), Y.reshape(-1), type_val)
        gain = parent_entropy - ent
        if gain > max_gain:
            max_gain = gain
            best_col = ind
    return best_col

class DTTree:
    def __init__(self):
        #Tree root should be DTNode
        self.root = None       

    def fit(self, X: np.ndarray, Y: np.ndarray, types: list, max_depth: int):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continuous then
                    types = [0,0,1,1]
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        def rec_fit(X: np.ndarray, Y: np.ndarray, types: list, depth: int):      # recursive function to fit create DTTree
            # Finding best val for the node
            best_val = 0
            count = {}
            for val in Y.tolist():
                if val not in count:
                    count[val] = 0
                count[val]+=1
            max_ct = max(count.values())
            for val in count:
                if count[val] == max_ct:
                    best_val = val
                    break
            # Creating Node
            if max_depth == depth:      # If reached max depth
                return DTNode(depth, 1, best_val)
            else:                       # If not at max depth
                column = best_column(X, Y, types)
                if column == -1:        # If no column split leads to positive gain in entropy
                    return DTNode(depth, 1, best_val)
                else:
                    type_val = types[column]
                    if type_val == 0:   # If categorical split
                        node = DTNode(depth, value = best_val, column = column)
                        X_values = set(X[:, column].reshape(-1).tolist())
                        for val in X_values:
                            indices = np.where(X[:, column] == val)
                            node.children[val] = rec_fit(X[indices], Y[indices], types, depth + 1)
                    else:               # If continuous split
                        threshold = np.median(X[:, column].reshape(-1))
                        node = DTNode(depth, column = column, type = 1, threshold = threshold)
                        split1 = np.where(X[:, column] < threshold)
                        split2 = np.where(X[:, column] >= threshold)
                        node.children[0] = rec_fit(X[split1], Y[split1], types, depth + 1)
                        node.children[1] = rec_fit(X[split2], Y[split2], types, depth + 1)
                    return node
        
        self.root = rec_fit(X, Y.reshape(-1), types, 0)

    def predict(self, x: np.ndarray):
        node = self.root
        if not node:
            return None
        while True:
            new_node = node.get_children(x)
            if new_node:
                node = new_node
            else:
                break
        return node.value
    
    def predict_arr(self, X: np.ndarray):
        Y = []
        for ind in range(X.shape[0]):
            Y.append(self.predict(X[ind, :]))
        return np.array(Y)
    
    def count_node(self):       # Counts Number of Node in tree
        def rec_count(node: DTNode):        # Recursively counts number of nodes
            ct = 1
            if node.is_leaf:
                return ct
            for child in node.children.values():
                ct += rec_count(child)
            return ct
        return rec_count(self.root)                    
    
    def post_prune(self, X_val: np.ndarray, Y_val: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray):     # Post Prune Tree Nodes
        def rec_prune(node: DTNode, X_val: np.ndarray, Y_val: np.ndarray):      # Recursively prunes and selects the best node to be pruned
            node.is_leaf = True
            acc = accuracy(Y_val, self.predict_arr(X_val))
            return_pair = (acc, node)
            node.is_leaf = False
            for child in node.children.values():
                if not child.is_leaf:
                    new_pair = rec_prune(child, X_val, Y_val)
                    if new_pair[0] > return_pair[0]:
                        return_pair = new_pair
            return return_pair
        num_nodes = []
        acc_train_wprune, acc_val_wprune, acc_test_wprune = [], [], []
        curr_acc = accuracy(Y_val.reshape(-1), self.predict_arr(X_val))
        prev_acc = curr_acc
        while True:
            prev_acc = curr_acc
            num_nodes.append(self.count_node())
            acc_train_wprune.append(accuracy(Y_train.reshape(-1), self.predict_arr(X_train)))
            acc_test_wprune.append(accuracy(Y_test.reshape(-1), self.predict_arr(X_test)))
            acc_val_wprune.append(curr_acc)
            curr_acc, curr_node = rec_prune(self.root, X_val, Y_val.reshape(-1))
            if curr_node.is_leaf:
                break
            if curr_acc <= prev_acc:
                break
            curr_node.is_leaf = True
        return num_nodes, acc_train_wprune, acc_val_wprune, acc_test_wprune
    
    def print_tree(self):
        if self.root:
            self.root.print_node(True)
        else:
            print(None)

def get_np_array(file_name, encoder):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    label_encoder = encoder
    label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

def accuracy(Y_true: np.ndarray, Y_predict: np.ndarray):    # Calculates accuracy of prediction
    correct_ct = 0
    total_ct = Y_true.shape[0]
    for ind in range(total_ct):
        if Y_true[ind] == Y_predict[ind]:
            correct_ct+=1
    return correct_ct/total_ct

def plot_accuracy(depth: list, acc_train: list, acc_val: list, acc_test: list, msg: str):      # Plots Accuracy vs Max Depth of tree
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(depth, acc_train, label = 'Training Accuracy', color = 'r')
    ax.plot(depth, acc_val, label = 'Validation Accuracy', color = 'g')
    ax.plot(depth, acc_test, label = 'Testing Accuracy', color = 'b')
    ax.set_title(f'Accuracy vs Max Depth ({msg})')
    ax.set_xlabel('Maximum Depth')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(f'Accuracy vs Max Depth ({msg})')

def plot_prune(depth: list, acc_train_hot: list, acc_train_hot_prune: list, acc_test_hot: list, acc_test_hot_prune: list):     # Plots Accuracy vs Max Depth for pruning
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(depth, acc_train_hot, label = 'Train Data Without Pruning', color = 'r')
    ax.plot(depth, acc_train_hot_prune, label = 'Train Data With Pruning', color = 'orange')
    ax.plot(depth, acc_test_hot, label = 'Test Data Without Pruning', color = 'g')
    ax.plot(depth, acc_test_hot_prune, label = 'Test Data With Pruning', color = 'b')
    ax.set_title(f'Accuracy vs Max Depth (Compare Pruning)')
    ax.set_xlabel('Maximum Depth')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(f'Accuracy vs Max Depth (Compare Pruning)')

def plot_wprune(depth: int, num_nodes: list, acc_train_wprune: list, acc_val_wprune: list, acc_test_wprune: list):     # Plots accuracy against number of nodes
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(num_nodes, acc_train_wprune, label = 'Training Accuracy', color = 'r')
    ax.plot(num_nodes, acc_val_wprune, label = 'Validation Accuracy', color = 'g')
    ax.plot(num_nodes, acc_test_wprune, label = 'Testing Accuracy', color = 'b')
    ax.set_title(f'Accuracy vs Num Nodes (While Pruning) depth = {depth}')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(f'Accuracy vs Num Nodes (While Pruning) depth = {depth}')

def plot_ccp(ccp_alpha_sci: list, acc_train_sci_ccp: list, acc_val_sci_ccp: list, acc_test_sci_ccp: list):     # Plots accuracy against ccp_alpha
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(ccp_alpha_sci, acc_train_sci_ccp, label = 'Training Accuracy', color = 'r')
    ax.plot(ccp_alpha_sci, acc_val_sci_ccp, label = 'Validation Accuracy', color = 'g')
    ax.plot(ccp_alpha_sci, acc_test_sci_ccp, label = 'Testing Accuracy', color = 'b')
    ax.set_title(f'Accuracy vs ccp_alpha')
    ax.set_xlabel('ccp_alpha')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(f'Accuracy vs ccp_alpha')

def create_report():        # Creates Report
    file = open('report.txt', 'w')
    file.write(f'Ord Depth:\t\t\t\t\t{depth_ord}\n')
    file.write(f'Training Accuracy:\t\t\t{acc_train_ord}\n')
    file.write(f'Validation Accuracy:\t\t{acc_val_ord}\n')
    file.write(f'Testing Accuracy:\t\t\t{acc_test_ord}\n')
    file.write(f'Test Accuracy for only win:\t{acc_only_win}\n')
    file.write(f'Test Accuracy for only lose:\t{acc_only_lose}\n\n')
    file.write(f'Hot Depth:\t\t\t\t\t{depth_hot}\n')
    file.write(f'Training Accuracy:\t\t\t{acc_train_hot}\n')
    file.write(f'Validation Accuracy:\t\t{acc_val_hot}\n')
    file.write(f'Testing Accuracy:\t\t\t{acc_test_hot}\n\n')
    file.write(f'Pruned:\t\t\t\t\t\t{depth_hot}\n')
    file.write(f'Training Accuracy:\t\t\t{acc_train_hot_prune}\n')
    file.write(f'Validation Accuracy:\t\t{acc_val_hot_prune}\n')
    file.write(f'Testing Accuracy:\t\t\t{acc_test_hot_prune}\n\n')
    # file.write(f'Sci-Kit (depth):\t\t\t{depth_sci}\n')
    # file.write(f'Training Accuracy:\t\t\t{acc_train_sci_depth}\n')
    # file.write(f'Validation Accuracy:\t\t{acc_val_sci_depth}\n')
    # file.write(f'Testing Accuracy:\t\t\t{acc_test_sci_depth}\n\n')
    # file.write(f'Sci-Kit (ccp_alpha):\t\t{ccp_alpha_sci}\n')
    # file.write(f'Training Accuracy:\t\t\t{acc_train_sci_ccp}\n')
    # file.write(f'Validation Accuracy:\t\t{acc_val_sci_ccp}\n')
    # file.write(f'Testing Accuracy:\t\t\t{acc_test_sci_ccp}\n\n')
    # file.write(f'Forest:\t\t\t\t\t\t\n')
    # file.write(f'Best Parameters:\t\t\t{best_param}\n')
    # file.write(f'oob Accuracy:\t\t\t\t{acc_forest_oob}\n')
    # file.write(f'Training Accuracy:\t\t\t{acc_forest_train}\n')
    # file.write(f'Validation Accuracy:\t\t{acc_forest_val}\n')
    # file.write(f'Testing Accuracy:\t\t\t{acc_forest_test}\n')
    file.close()

types = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]        # 0 = categorical   1 = continuous

# Reading Data
X_train,Y_train = get_np_array('../data/q1/train.csv', OrdinalEncoder())
X_test, Y_test = get_np_array('../data/q1/test.csv', OrdinalEncoder())
X_val, Y_val = get_np_array('../data/q1/val.csv', OrdinalEncoder())
print("Read Data Ordinary Encoder")

# Part (a)
depth_ord = [5, 10, 15, 20, 25]
acc_train_ord = []
acc_val_ord = []
acc_test_ord = []
for max_depth in depth_ord:
    # Creating Tree
    tree = DTTree()
    tree.fit(X_train, Y_train, types, max_depth)

    # Calculating Accuracies
    acc_train_ord.append(accuracy(Y_train.reshape(-1), tree.predict_arr(X_train)))
    acc_val_ord.append(accuracy(Y_val.reshape(-1), tree.predict_arr(X_val)))
    acc_test_ord.append(accuracy(Y_test.reshape(-1), tree.predict_arr(X_test)))

    print(f"Accuracy Calculated for max depth = {max_depth}")
# Plotting Accuracy vs Max Depth
plot_accuracy(depth_ord, acc_train_ord, acc_val_ord, acc_test_ord, 'Ordinary Encoder')

# Only Win and Only Lose Accuracy
acc_only_win = accuracy(Y_test.reshape(-1), np.ones((Y_test.shape[0],)))
acc_only_lose = accuracy(Y_test.reshape(-1), np.zeros((Y_test.shape[0],)))

# Reading Data
X_train,Y_train = get_np_array('../data/q1/train.csv', OneHotEncoder(sparse_output=False))
X_test, Y_test = get_np_array('../data/q1/test.csv', OneHotEncoder(sparse_output=False))
X_val, Y_val = get_np_array('../data/q1/val.csv', OneHotEncoder(sparse_output=False))
print("Read Data Hot Encoder")

types = [0]*(X_train.shape[1] - len(types)) + types

# Part (b)
depth_hot = [15, 25, 35, 45]
# depth_hot = [15]
acc_train_hot = []
acc_val_hot = []
acc_test_hot = []
acc_train_hot_prune = []
acc_val_hot_prune = []
acc_test_hot_prune = []
for max_depth in depth_hot:
    # Creating Tree
    tree = DTTree()
    tree.fit(X_train, Y_train, types, max_depth)

    # Calculating Accuracies
    acc_train_hot.append(accuracy(Y_train.reshape(-1), tree.predict_arr(X_train)))
    acc_val_hot.append(accuracy(Y_val.reshape(-1), tree.predict_arr(X_val)))
    acc_test_hot.append(accuracy(Y_test.reshape(-1), tree.predict_arr(X_test)))
    
    print(f"Accuracy Calculated for max depth = {max_depth}")
    
    # Part (c)
    # Pruning Tree
    num_nodes, acc_train_wprune, acc_val_wprune, acc_test_wprune = tree.post_prune(X_val, Y_val, X_test, Y_test, X_train, Y_train)
    
    plot_wprune(max_depth, num_nodes, acc_train_wprune, acc_val_wprune, acc_test_wprune)
    
    # Calculating Accuracies
    acc_train_hot_prune.append(acc_train_wprune[-1])
    acc_test_hot_prune.append(acc_test_wprune[-1])
    acc_val_hot_prune.append(acc_val_wprune[-1])
    
    print(f"Pruned for max depth = {max_depth}")

# Plotting Accuracy vs Max Depth
plot_accuracy(depth_hot, acc_train_hot, acc_val_hot, acc_test_hot, 'OneHot Encoder')
plot_accuracy(depth_hot, acc_train_hot_prune, acc_val_hot_prune, acc_test_hot_prune, 'Pruned')
plot_prune(depth_hot, acc_train_hot, acc_train_hot_prune, acc_test_hot, acc_test_hot_prune)


# # Part (d(i))
# depth_sci = [15, 25, 35, 45]
# acc_train_sci_depth = []
# acc_val_sci_depth = []
# acc_test_sci_depth = []
# for max_depth in depth_sci:
#     tree = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
#     tree.fit(X_train, Y_train)

#     # Calculating Accuracies
#     acc_train_sci_depth.append(accuracy(Y_train.reshape(-1), tree.predict(X_train).reshape(-1)))
#     acc_val_sci_depth.append(accuracy(Y_val.reshape(-1), tree.predict(X_val).reshape(-1)))
#     acc_test_sci_depth.append(accuracy(Y_test.reshape(-1), tree.predict(X_test).reshape(-1)))
    
#     print(f"Accuracy Calculated for max depth = {max_depth}")
# # Plotting Accuracy vs Max Depth
# plot_accuracy(depth_sci, acc_train_sci_depth, acc_val_sci_depth, acc_test_sci_depth, 'Sci-Kit')

# # Part (d(ii))
# ccp_alpha_sci = [0.001, 0.01, 0.1, 0.2]
# acc_train_sci_ccp = []
# acc_val_sci_ccp = []
# acc_test_sci_ccp = []
# for ccp_alpha in ccp_alpha_sci:
#     tree = DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha)
#     tree.fit(X_train, Y_train)

#     # Calculating Accuracies
#     acc_train_sci_ccp.append(accuracy(Y_train.reshape(-1), tree.predict(X_train).reshape(-1)))
#     acc_val_sci_ccp.append(accuracy(Y_val.reshape(-1), tree.predict(X_val).reshape(-1)))
#     acc_test_sci_ccp.append(accuracy(Y_test.reshape(-1), tree.predict(X_test).reshape(-1)))
    
#     print(f"Accuracy Calculated for ccp_alpha = {ccp_alpha}")
# # Plotting Accuracy vs ccp_alpha
# plot_ccp(ccp_alpha_sci, acc_train_sci_ccp, acc_val_sci_ccp, acc_test_sci_ccp)


# # Part (e)
# param_grid = {'n_estimators': [50, 150, 250, 350], 'max_features': [0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': [2, 4, 6, 8, 10]}
# forest = RandomForestClassifier(oob_score=True, criterion='entropy')
# grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, scoring='accuracy')
# grid_search.fit(X_train, Y_train.reshape(-1))
# print("Grid Search Completed")
# best_param = grid_search.best_params_
# best_forest = grid_search.best_estimator_
# acc_forest_oob = best_forest.oob_score_
# acc_forest_train = accuracy(Y_train.reshape(-1), best_forest.predict(X_train))
# acc_forest_test = accuracy(Y_test.reshape(-1), best_forest.predict(X_test))
# acc_forest_val = accuracy(Y_val.reshape(-1), best_forest.predict(X_val))

# Create Report
create_report()