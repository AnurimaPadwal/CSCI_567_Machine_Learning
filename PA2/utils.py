import numpy as np
import math


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    
    branches = np.array(branches)
    num_samples = np.sum(branches)
    res = 0
   
    for i in range(len(branches)):
        sum_sample = sum(branches[i])
        entropy = 0
        for j in range(len(branches[i])):
            if branches[i][j]!=0:
                log_p = (np.log2(branches[i][j]) - np.log2(sum_sample))
                p = (branches[i][j]/float(sum_sample))
                entropy += -(branches[i][j]/float(sum_sample)) * (np.log2(branches[i][j]) - np.log2(sum_sample))
        res = res + (sum_sample/float(num_samples))*entropy
        
    
    return (S - res)
    #raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    #raise NotImplementedError
    
    
    
    queue = []
    
    queue.append(decisionTree.root_node)
    y_true = decisionTree.predict(X_test)
    
    def calculate_accuracy(y_true, y_predicted):
        count = 0
        for i in range(len(y_true)):
            if y_true[i] == y_predicted[i]:
                count += 1
        
        accuracy = float(count)/len(y_true)

        return accuracy
    
    original_accuracy = calculate_accuracy(y_test, y_true)
    max_accuracy = original_accuracy
    
   
    def can_prune(node): 
        node.splittable = False
        predicted_vals = decisionTree.predict(X_test)
        node.splittable = True
        
        accuracy = calculate_accuracy(y_test, predicted_vals)
        
        node.accuracy = accuracy
        if accuracy >= max_accuracy:
            return True
        else:
            return False

    while queue:
        root = queue.pop()
        
        if root is not decisionTree.root_node and root.splittable and can_prune(root):
            root.splittable = False
            root.children = None
            max_accuracy = root.accuracy
            
        else:
            queue.extend(root.children)
    
    if decisionTree.root_node.splittable:
        decisionTree.root_node.splittable = False
        predicted_vals = decisionTree.predict(X_test)
        
        
        accuracy = calculate_accuracy(y_test, predicted_vals)
        if accuracy >= max_accuracy:
            decisionTree.root_node.children = False
        else:
            decisionTree.root_node.splittable = True
        

    return decisionTree.predict(X_test)  
   
    """
    if X_test is None or len(X_test) == 0 or len(X_test[0]) == 0:
        decisionTree.root_node.children = None
        decisionTree.root_node.splittable = False
        return decisionTree.predict(X_test)
    
    def calculate_accuracy(y_true, y_predicted):
        count = 0
        for i in range(len(y_true)):
            if y_true[i] == y_predicted[i]:
                count += 1
        
        accuracy = float(count)/len(y_true)

        return accuracy
    
    def prune(node, max_accuracy):
        if not node.splittable:
            return max_accuracy
        
        for child in node.children:
            max_accuracy = prune(child, max_accuracy)
            node.splittable = False
            y_predicted = decisionTree.predict(X_test)
            accuracy = calculate_accuracy(y_test, y_predicted)
            if accuracy >= max_accuracy:
                max_accuracy = accuracy
                node.splittable = False
                node.children = None
                return max_accuracy
            else:
                node.splittable = True
        
            
        return max_accuracy

    y_labels = decisionTree.predict(X_test)
    max_accuracy = calculate_accuracy(y_test, y_labels)
    root = decisionTree.root_node
    
    
    max_accuracy = prune(root, max_accuracy)
    if root.splittable:
        root.splittable = False
        y_predicted = decisionTree.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_predicted)
        if accuracy >= max_accuracy:
            root.children = None
        else:
            root.splittable = True
        
    return decisionTree.predict(X_test)
    """
    

    
        

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
