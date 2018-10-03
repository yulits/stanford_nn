from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        X = X.reshape(N, -1)

        q = X.dot(W1) + b1
        h = np.maximum(0, q)
        scores = h.dot(W2) + b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (scores.shape[0], -1))
        loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))

        # Add regularization to the loss.
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        sigma[range(y.shape[0]), y] -= 1

        dW2 = np.dot(h.T, sigma)
        dW2 /= N
        # Add regularization to gradient
        dW2 += self.reg * W2

        dh = np.dot(sigma, W2.T)
        ones = np.ones(N)
        db2 = ones.dot(sigma)
        db2 /= N

        # intermediate gradient
        dq = np.zeros(q.shape)
        dq[np.where(q > 0)] = 1
        dq = dq * dh

        dW1 = np.dot(X.T, dq)
        dW1 /= N
        # Add regularization to gradient
        dW1 += self.reg * W1
        ones = np.ones(N)
        db1 = ones.dot(dq)
        db1 /= N

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.cache = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        np.random.seed(231)

        Ws = ['W' + str(i) for i in range(1, self.num_layers + 1)]
        bs = ['b' + str(i) for i in range(1, self.num_layers + 1)]

        self.params[Ws[0]] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params[bs[0]] = np.zeros(hidden_dims[0])

        for i in range(1, self.num_layers - 1):
            self.params[Ws[i]] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
            self.params[bs[i]] = np.zeros(hidden_dims[i])

        self.params[Ws[-1]] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params[bs[-1]] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            gammas = ['gamma' + str(i) for i in range(1, self.num_layers + 1)]
            betas = ['beta' + str(i) for i in range(1, self.num_layers + 1)]
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for i in range(self.num_layers - 1):
                self.params[gammas[i]] = np.ones(hidden_dims[i])
                self.params[betas[i]] = np.zeros(hidden_dims[i])
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        Ws = ['W' + str(i) for i in range(1, self.num_layers + 1)]
        bs = ['b' + str(i) for i in range(1, self.num_layers + 1)]

        N = X.shape[0]
        X = X.reshape(N, -1)

        out = X.copy()
        self.cache = []

        for i in range(self.num_layers-1):
            # batch normalization
            if self.normalization=='batchnorm':
                gammas = ['gamma' + str(i) for i in range(1, self.num_layers + 1)]
                betas = ['beta' + str(i) for i in range(1, self.num_layers + 1)]
                # batchnorm_forward(X, self.params[gammas[i]], self.params[betas[i]], self.bn_param)
                out, cache, self.bn_params[i] = affine_bn_relu_forward(out, self.params[Ws[i]], self.params[bs[i]],
                                                    self.params[gammas[i]], self.params[betas[i]], self.bn_params[i])
            else:
                out, cache = affine_relu_forward(out, self.params[Ws[i]], self.params[bs[i]])
            if self.use_dropout:
                out, cache['dropout'] = dropout_forward(out, self.dropout_param)
            self.cache.append(cache)
        scores, cache = affine_forward(out, self.params[Ws[-1]], self.params[bs[-1]])
        self.cache.append(cache)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (scores.shape[0], -1))
        loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))

        # Add regularization to the loss.
        W_sum = 0
        for i in range(self.num_layers):
            W_sum += np.sum(self.params[Ws[i]] * self.params[Ws[i]])
        loss += 0.5 * self.reg * W_sum
        #start backprop
        sigma[range(N), y] -= 1

        dout = sigma

        dout, dw, db = affine_backward(dout, self.cache[-1])
        dw /= N
        # Add regularization to the weights
        dw += self.reg * self.params[Ws[-1]]
        grads[Ws[-1]] = dw

        db /= N
        grads[bs[-1]] = db

        for i in reversed(range(0, self.num_layers - 1)):

            if self.use_dropout:
                dout = dropout_backward(dout, self.cache[i]['dropout'])

            # batch normalization
            if self.normalization == 'batchnorm':


                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, self.cache[i])

                dgamma /= N
                grads[gammas[i]] = dgamma

                dbeta /= N
                grads[betas[i]] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, self.cache[i])

            # Add regularization to the weights
            dw /= N
            dw += self.reg * self.params[Ws[i]]
            grads[Ws[i]] = dw

            db /= N
            grads[bs[i]] = db


        return loss, grads

    def loss_old(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        Ws = ['W' + str(i) for i in range(1, self.num_layers + 1)]
        bs = ['b' + str(i) for i in range(1, self.num_layers + 1)]

        N = X.shape[0]
        X = X.reshape(N, -1)

        # out = X
        # for i in range(self.num_layers-1):
        #     out, cache = affine_relu_forward(out, self.params[Ws[i]], self.params[bs[i]])
        #     self.cache.append(cache)
        #     # batch normalization
        #     # if self.normalization=='batchnorm':
        #     #     gammas = ['gamma' + str(i) for i in range(1, self.num_layers + 1)]
        #     #     betas = ['beta' + str(i) for i in range(1, self.num_layers + 1)]
        #     #     batchnorm_forward(X, self.params[gammas[i]], self.params[betas[i]], self.bn_param)
        #
        # scores, cache = affine_forward(out, self.params[Ws[-1]], self.params[bs[-1]])
        # self.cache.append(cache)

        cur_layer = X
        affine = []
        relu = []
        for i in range(self.num_layers - 1):
            a = cur_layer.dot(self.params[Ws[i]]) + self.params[bs[i]]
            affine.append(a)
            # batch normalization
            # if self.normalization=='batchnorm':
            #     gammas = ['gamma' + str(i) for i in range(1, self.num_layers + 1)]
            #     betas = ['beta' + str(i) for i in range(1, self.num_layers + 1)]
            #     batchnorm_forward(X, self.params[gammas[i]], self.params[betas[i]], self.bn_param)
            # non-linearity
            out = np.maximum(0, a)
            relu.append(out)
            cur_layer = out
        scores = out.dot(self.params[Ws[-1]]) + self.params[bs[-1]]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (scores.shape[0], -1))
        loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))

        # Add regularization to the loss.
        W_sum = 0
        for i in range(self.num_layers):
            W_sum += np.sum(self.params[Ws[i]] * self.params[Ws[i]])
        loss += 0.5 * self.reg * W_sum
        #start backprop
        sigma[range(N), y] -= 1
        # dout = sigma
        # dout, dw, db = affine_backward(dout, self.cache[-1])
        # dw /= N
        # # Add regularization to the weights
        # dw += self.reg * self.params[Ws[-1]]
        # grads[Ws[-1]] = dw
        #
        # db /= N
        # grads[bs[-1]] = db
        #
        # for i in reversed(range(0, self.num_layers - 1)):
        #     dout, dw, db = affine_relu_backward(dout, self.cache[i])
        #     # Add regularization to the weights
        #     dw /= N
        #     dw += self.reg * self.params[Ws[i]]
        #     grads[Ws[i]] = dw
        #
        #     db /= N
        #     grads[bs[i]] = db


        grad = sigma
        ones = np.ones(N)
        for i in reversed(range(1, self.num_layers)):
            dW = np.dot(relu[i-1].T, grad)

            dW /= N
            # Add regularization to gradient
            dW += self.reg * self.params[Ws[i]]
            grads[Ws[i]] = dW

            db = ones.dot(grad)
            db /= N
            grads[bs[i]] = db

            dh = np.dot(grad, self.params[Ws[i]].T)

            # intermediate gradient
            drelu = np.zeros(affine[i-1].shape)
            drelu[np.where(affine[i-1] > 0)] = 1
            grad = drelu * dh

        dW = np.dot(X.T, grad)
        dW /= N
        # Add regularization to gradient
        dW += self.reg * self.params[Ws[0]]
        grads[Ws[0]] = dW

        db = ones.dot(grad)
        db /= N
        grads[bs[0]] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
