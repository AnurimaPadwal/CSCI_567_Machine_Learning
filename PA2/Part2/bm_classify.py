import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        ############################################
        for t in range(max_iterations):
            
            z = np.dot(X,w) + b
            y_new = y
            y_new[y_new == 0] = -1
            
            s = y_new * z
        
            
            sign = np.zeros(N)
            sign = (s <=0).astype(float)
            
            prod = sign * y_new
            w_grad = np.dot(prod, X)
            b_grad = np.sum(prod)
            
            w = w + step_size * w_grad/N
            b = b + step_size * b_grad/N
           
        
            """
            ypred = np.matmul(X,w) + b
            ypred[ypred > 0] = 1
            ypred[ypred <= 0] = 0
            sign = y - ypred
            w_grad = np.matmul(sign,X)/N
            b_grad = np.sum(sign)/N
      
            w = w + step_size * w_grad
            b = b + step_size * b_grad
            """
            
  
    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        for t in range(max_iterations):
            z = sigmoid(np.dot(X,w) + b)
            diff = z - y
            w_grad = diff.dot(X)/N
            b_grad = np.sum(diff)/N
            
            w = w - step_size * w_grad
            b = b - step_size * b_grad
            
            
            

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    ############################################
    
    value = 1.0/(1 + np.exp(-z))

    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        prediction = np.dot(X,w) + b
        preds = (prediction > 0).astype(float)
   
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        prediction = sigmoid(np.dot(X,w) + b)
        preds = (prediction > 0.5).astype(float)

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    
   
    
    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        ############################################
        y = np.eye(C)[y]
        
        def softmax(z):
            z = np.exp(z - np.amax(z))
            denominator = np.sum(z)
            return (z.T/denominator).T
        for t in range(max_iterations):
            #idx = np.random.randint(X.shape[0], size = 1)
            idx = np.random.choice(N)
            
            x = X[idx,:]
            y_new = y[idx,:]
            
            error = softmax((np.dot(w,x.T) + b)) - y_new
            
            error = error.reshape((C,1))
            x = x.reshape((1,D))
            
            w_gradient = np.dot(error, x)/N
            b_gradient = np.sum(error,axis = 0)/N
            
            w = w - step_size * w_gradient
            b = b - step_size * b_gradient
            
            
            
    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        ############################################
        
        y = np.eye(C)[y]
        
        def softmax(z):
            z = np.exp(z - np.amax(z))
            denominator = np.sum(z,axis=1)
            return (z.T/denominator).T
       
        
        for t in range(max_iterations):
            error = softmax(np.dot(X,w.T) + b) - y
            w_gradient = np.dot(error.T, X)/N
            b_gradient = np.sum(error,axis=0)/N
            
            w = w - step_size * w_gradient
            b = b - step_size * b_gradient
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    
    def softmax(z):
        numerator = np.exp(z - np.max(z))
        denominator = np.sum(numerator, axis=1)
       
        return (numerator.T/denominator).T
    
    
    preds = softmax(np.dot(X, w.T) + b)
    preds = np.argmax(preds, axis = 1)
    
    assert preds.shape == (N,)
    return preds