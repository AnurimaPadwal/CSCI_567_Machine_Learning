import numpy as np
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.value = None
        self.split_index = None
        self.accuracy = None
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self): 
        node_entropy = 0
        
        for label in set(self.labels):
            node_entropy += -(self.labels.count(label)/float(len(self.labels))) * (np.log2(self.labels.count(label)) - np.log2(len(self.labels)))

        feature_map = {}
        feature_count_map = {}
        for item in enumerate(zip(*self.features)):
            attr_no = item[0]
            attr_vals = item[1]
            feature_map[attr_no] = set(attr_vals)

            for i in zip(attr_vals, self.labels):
                if (attr_no, i[0], i[1]) not in feature_count_map:
                    count = 0
                else:
                    count = feature_count_map[(attr_no, i[0], i[1])]
                feature_count_map[(attr_no, i[0], i[1])] = count + 1
        
        branch_list = []
        for num in feature_map:
            branches = []
            for vals in feature_map[num]:
                branch = []
                for label in set(self.labels):
                    if (num, vals, label) not in feature_count_map:
                        count = 0
                    else:
                        count = feature_count_map[(num, vals, label)]
                    branch.append(count)
                branches.append(branch)
            branch_list.append(branches)
        
        max_gain = -float("inf")
        max_gain_branch_id = -1
        for i, branch in enumerate(branch_list):
            gain = Util.Information_Gain(node_entropy, branch)
            if gain > max_gain:
                max_gain = gain
                max_gain_branch_id = i
            elif gain == max_gain:
                if len(feature_map[max_gain_branch_id]) < len(feature_map[i]):
                    max_gain_branch_id = i
        
        self.dim_split = max_gain_branch_id
       
        
        x  = np.array(self.features)
        values = np.unique(x[:,max_gain_branch_id])
        values = sorted(values)
        self.feature_uniq_split = values
        
        
        
        #self.features = np.delete(self.features, len(self.features)-1, axis = 1)
        x_i = np.array(self.features)[:, self.dim_split]
        
        for val in self.feature_uniq_split:
            indexes = np.where(x_i == val)
            child_features = x[indexes].tolist()
            child_features = np.delete(child_features, self.dim_split,axis=1)
            child_labels = np.array(self.labels)[indexes].tolist()   
            num_cls = np.unique(child_labels).size
            node = TreeNode(child_features, child_labels, num_cls)
            if np.array(child_features).size == 0 or all(x is None for x in child_features[0]):
                node.splittable = False
 
            self.children.append(node)
        
        #self.children = sorted(self.children, key = lambda x : x.value)
        
        for child in self.children:
            if child.splittable:
                child.split()
        
        return
        
        #self.children = sorted(self.children, key = lambda x: len(x.labels))

        #raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        #raise NotImplementedError
        
        if self.splittable:
            idx = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx].predict(np.delete(feature, self.dim_split))
        else:
            return self.cls_max