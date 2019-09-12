import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    f1_score_val = (2 * sum(np.asarray(real_labels) * np.asarray(predicted_labels)))/(sum(real_labels) + sum(predicted_labels))
    return f1_score_val
    
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p = 3
        diff = np.asarray(point1) - np.asarray(point2)
        result = np.linalg.norm(diff, ord = p)
        return result

        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        result = np.asarray(point1) - np.asarray(point2)
        result = np.linalg.norm(result)
        return result

        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        result = np.dot(point1, point2)
        return result

        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):

        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
<<<<<<< HEAD
        
        result = (np.dot(point1, point2))/(np.linalg.norm(point1) * np.linalg.norm(point2))
        return (1 - result)
        
        #raise NotImplementedError
=======
        result = (np.dot(point1, point2))/(np.linalg.norm(point1) * np.linalg.norm(point2))
        return (1 - result)
        
        raise NotImplementedError
>>>>>>> origin/master

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x = np.asarray(point1) - np.asarray(point2)
        x = -0.5 * np.dot(x, x)
        return -np.exp(x)
<<<<<<< HEAD
        
        #raise NotImplementedError
=======
        raise NotImplementedError
>>>>>>> origin/master


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
<<<<<<< HEAD
      

        f1_score_max = -float("inf")

        for distance in distance_funcs:
            for k in range(1, 31, 2):
                if k>len(x_train):
                    break
                knn_model = KNN(k, distance_funcs[distance])
                knn_model.train(x_train, y_train)
                predicted_vals = knn_model.predict(x_val)
=======
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        f1_score_max = -float("inf")

        for distance in distance_funcs.items():
            for k in range(1, 31, 2):
                knn = knn.KNN(k, distance_funcs[distance])
                knn.train(x_train, y_train)
                predicted_vals = knn.predict(x_val)
>>>>>>> origin/master
                f1_score_val = f1_score(y_val, predicted_vals)
                if f1_score_val > f1_score_max:
                    f1_score_max = f1_score_val
                    self.best_k = k
                    self.best_distance_function = distance
<<<<<<< HEAD
                    self.best_model = knn_model
=======
                    self.best_model = knn
>>>>>>> origin/master
                elif f1_score == f1_score_max:
                    if distance == self.best_distance_function:
                        if k < self.k:
                            self.k = k
                            self.best_distance_function = distance
<<<<<<< HEAD
                            self.best_model = knn_model
=======
                            self.best_model = knn
>>>>>>> origin/master
                    else:
                        if distance == 'euclidean':
                            self.best_distance_function = distance
                            self.best_k = k
<<<<<<< HEAD
                            self.best_model = knn_model
=======
                            self.best_model = knn
>>>>>>> origin/master
                        elif distance == 'minkowski':
                            if self.best_distance_function == 'gaussian' or self.best_distance_function == 'inner_prod' or self.best_distance_function == 'cosine_dist':
                                self.best_distance_function = distance
                                self.best_k = k
<<<<<<< HEAD
                                self.best_model = knn_model
=======
                                self.best_model = knn
>>>>>>> origin/master
                        elif distance == 'gaussian':
                            if self.best_distance_function == 'inner_prod' or self.best_distance_function == 'cosine_dist':
                                self.best_distance_function = distance
                                self.best_k = k
<<<<<<< HEAD
                                self.best_model = knn_model
=======
                                self.best_model = knn
>>>>>>> origin/master
                        elif distance == 'inner_prod':
                            if self.best_distance_function == 'cosine_dist':
                                self.best_distance_function = distance
                                self.best_k = k
<<<<<<< HEAD
                                self.best_model = knn_model
                
        #raise NotImplementedError
=======
                                self.best_model = knn
                
        raise NotImplementedError
>>>>>>> origin/master
    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
<<<<<<< HEAD
     
    

        f1_score_max = -float("inf")
        
        scalers = ["min_max_scale", "normalize"]

        for scaler in scalers:
            for distance in distance_funcs:
                for k in range(1, 31, 2):
                    if k > len(x_train):
                        break
                    knn_model = KNN(k, distance_funcs[distance])
                    if scaler == "min_max_scale":
                        scale = MinMaxScaler()
                     
                    else:
                        scale = NormalizationScaler()
                        
                        
                    scaled_x_train = scale(x_train)
                    scaled_x_val = scale(x_val)

                    knn_model.train(scaled_x_train, y_train)
                    predicted_vals = knn_model.predict(scaled_x_val)

                    f1_score_val = f1_score(y_val, predicted_vals)

                    if f1_score_val > f1_score_max:
                        f1_score_max = f1_score_val
                        self.best_k = k
                        self.best_distance_function = distance
                        self.best_model = knn_model
                        self.best_scaler = scaler
                    elif f1_score_val == f1_score_max:
                        if scaler == "min_max_scale" and self.best_scaler == "normalize":
                            self.best_scaler = scaler
                            self.best_k = k
                            self.best_model = knn_model
                            self.best_distance_function = distance
                        elif scaler == "normalize" and self.best_scaler == "min_max_scale":
                            continue
                        else:
                            if distance == self.best_distance_function:
                                if k < self.best_k:
                                    self.best_k = k
                                    self.best_distance_function = distance
                                    self.best_model = knn_model
                                else:
                                    if distance == 'euclidean':
                                        self.best_distance_function = distance
                                        self.best_k = k
                                        self.best_model = knn_model
                                    elif distance == 'minkowski':
                                        if self.best_distance_function == 'gaussian' or self.best_distance_function == 'inner_prod' or self.best_distance_function == 'cosine_dist':
                                            self.best_distance_function = distance
                                            self.best_k = k
                                            self.best_model = knn_model
                                    elif distance == 'gaussian':
                                        if self.best_distance_function == 'inner_prod' or self.best_distance_function == 'cosine_dist':
                                            self.best_distance_function = distance
                                            self.best_k = k
                                            self.best_model = knn_model
                                    elif distance == 'inner_prod':
                                        if self.best_distance_function == 'cosine_dist':
                                            self.best_distance_function = distance
                                            self.best_k = k
                                            self.best_model = knn_model



        #raise NotImplementedError
=======
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        raise NotImplementedError
>>>>>>> origin/master


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
<<<<<<< HEAD
        normalized_features = []
        for feature in features:
            if not np.any(feature):
                normalized_features.append(feature)
                continue
            feature = feature/np.linalg.norm(feature)
            normalized_features.append(feature.tolist())

        return normalized_features   
        #raise NotImplementedError
=======
        raise NotImplementedError
>>>>>>> origin/master


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
<<<<<<< HEAD
        self.is_called = False
        self.features_max = None
        self.features_min = None
        self.max_min_difference = None
=======
>>>>>>> origin/master
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
<<<<<<< HEAD
        if not self.is_called:
            self.is_called = True
            self.features_max = np.amax(features, axis=0)
            self.features_min = np.amin(features, axis=0)
            self.max_min_difference = self.features_max - self.features_min
            
           
            
         
        
        result = (features - self.features_min)/self.max_min_difference
        return result.tolist()
        
        #raise NotImplementedError
=======
        raise NotImplementedError
>>>>>>> origin/master

