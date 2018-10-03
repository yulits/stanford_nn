from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.cache = []
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        layer2_height = (input_dim[1] + 2 * self.conv_param['pad'] - filter_size) // self.conv_param['stride'] + 1
        layer2_width = (input_dim[2] + 2 * self.conv_param['pad'] - filter_size) // self.conv_param['stride'] + 1
        layer2_height = (layer2_height - self.pool_param['pool_height']) // self.pool_param['stride'] + 1
        layer2_width = (layer2_width - self.pool_param['pool_width']) // self.pool_param['stride'] + 1

        self.params['W2'] = weight_scale * np.random.randn(num_filters * layer2_height * layer2_width, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # # pass conv_param to the forward pass for the convolutional layer
        # # Padding and stride chosen to preserve the input spatial size
        # filter_size = W1.shape[2]
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        #
        # # pass pool_param to the forward pass for the max-pooling layer
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        N = X.shape[0]
        self.cache = []
        out, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        self.cache.append(cache)
        out, cache = affine_relu_forward(out, W2, b2)
        self.cache.append(cache)
        scores, cache = affine_forward(out, W3, b3)
        self.cache.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (scores.shape[0], -1))
        loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))

        # Add regularization to the loss.
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        # start backprop
        sigma[range(N), y] -= 1

        dout = sigma

        dout, dw, db = affine_backward(dout, self.cache[2])
        dw /= N
        # Add regularization to the weights
        dw += self.reg * W3
        grads['W3'] = dw

        db /= N
        grads['b3'] = db

        dout, dw, db = affine_relu_backward(dout, self.cache[1])

        # Add regularization to the weights
        dw /= N
        dw += self.reg * W2
        grads['W2'] = dw

        db /= N
        grads['b2'] = db

        dout, dw, db = conv_relu_pool_backward(dout, self.cache[0])

        # Add regularization to the weights
        dw /= N
        dw += self.reg * W1
        grads['W1'] = dw

        db /= N
        grads['b1'] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def just_loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """


        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # # pass conv_param to the forward pass for the convolutional layer
        # # Padding and stride chosen to preserve the input spatial size
        # filter_size = W1.shape[2]
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        #
        # # pass pool_param to the forward pass for the max-pooling layer
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        self.cache = []
        out, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        self.cache.append(cache)
        out, cache = affine_relu_forward(out, W2, b2)
        self.cache.append(cache)
        scores, cache = affine_forward(out, W3, b3)
        self.cache.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (scores.shape[0], -1))
        loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))

        # Add regularization to the loss.
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        return loss, None