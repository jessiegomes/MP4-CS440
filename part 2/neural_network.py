import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    #IMPLEMENT HERE
    batch_size = 200

    for e in range(1, epoch):
        if shuffle:
            og_state = np.random.get_state()
            np.random.shuffle(x_train)
            np.random.set_state(og_state)
            np.random.shuffle(y_train)
        for i in range(1, len(x_train)/batch_size):
            x = x_train[((i-1)*batch_size):(i*batch_size)]
            y = y_train[((i-1)*batch_size):(i*batch_size)]
            w1, w2, w3, w4, loss = four_nn(x, w1, w2, w3, w4, b1, b2, b3, b4, y, not shuffle)

    return w1, w2, w3, w4, b1, b2, b3, b4, loss

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    classifications = four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, True)
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    class_totals = [0.0] * num_classes
    for i in range(len(x_test)):
        if y_test[i] == classifications[i]:
            class_rate_per_class += 1
            avg_class_rate += 1
        class_totals[y_test[i]] += 1
    for i in range(num_classes):
        class_rate_per_class[i] /= class_totals[i]
    avg_class_rate /= len(y_test)
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(x, w1, w2, w3, w4, b1, b2, b3, b4, y, test):
    eta = .1
    z1, ac1 = affine_forward(x, w1, b1)
    a1, rc1 = relu_forward(z1)
    z2, ac2 = affine_forward(a1, w2, b2)
    a2, rc2 = relu_forward(z2, w2, b2)
    z3, ac3 = affine_forward(a2, w3, b3)
    a3, rc3 = relu_forward(z3, w3, b3)
    f, ac4 = affine_forward(a3, w4, b4)
    if test:
        classifications = [np.argmax(x) for x  in f]
        return classifications
    loss, df = cross_entropy(f, y)
    da2, dw3, db3 = affine_backward(df, ac3)
    dz2 = relu_backward(da2, rc2)
    da1, dw2, db2 = affine_backward(dz2, ac2)
    dz1 = relu_backward(da1, rc1)
    dx, dw1, db1 = affine_backward(dz1, ac1)
    w1 = w1 - eta*dw1
    return w1, w2, w3, w4, loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    n = A.shape[0]
    d = np.prod(A.shape[1:])
    A2 = np.reshape(A, (n, d))
    Z = np.dot(A2, W) + b
    cache = (A, W, b)
    # print(Z)
    return Z, cache

def affine_backward(dZ, cache):
    dA, dW, dB = None, None, None
    A, W, b = cache
    n = A.shape[0]
    d = np.prod(A.shape[1:])
    A2 = np.reshape(A, (n, d))

    old_dA = np.dot(dZ, W.T)
    dW = np.dot(A2.T, dZ)
    dB = np.dot(dZ.T, np.ones(n))
    dA = np.reshape(old_dA, A.shape)
    # print(dA, dW, dB)
    return dA, dW, dB

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache = Z
    # print(A)
    return A, cache

def relu_backward(dZ, cache):
    Z = cache
    dA = np.array(dZ, copy=True)
    dA[Z <= 0] = 0
    # print(dA)
    return dA

def cross_entropy(F, y):
    loss = 0
    f_0 = len(F)
    f_1 = len(F[0])
    y_len = len(y)
    for a in range(f_0):
        loss += F[a][int(y[a])]
        exp_sum = 0
        for b in range(f_1):
            exp_sum += np.exp(F[a][b])
        loss -= np.log(exp_sum)
    loss = (-1 * loss) / y_len

    df = F
    flog=0
        
    for a in range(f_0):
        # flog = np.sum(np.exp(F))
        flog = 0
        for b in range(f_1):
            flog += np.exp(F[a][b])
        for b in range(f_1):
            temp = np.exp(F[a][b]) / flog
            df[a][b] = (-1) * (1*(b == y[a]) - temp) / y_len
    print(loss)
    print(df)
    return loss, df
