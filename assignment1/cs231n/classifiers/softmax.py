import numpy as np
from random import shuffle, seed

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)

    for j in range(num_classes):
      sigma = np.exp(scores[j]) / np.sum(np.exp(scores))

      if j == y[i]:
        loss += - 1 * np.log(sigma)
        dW[:, y[i]] += X[i] * (sigma - 1)
      else:
        loss += - 0 * np.log(sigma)
        dW[:, j] += X[i] * sigma
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to gradient
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  sigma = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (num_train, -1))

  loss = np.mean(-np.log(sigma[range(y.shape[0]), y]))
  #
  sigma[range(y.shape[0]), y] -= 1

  dW = np.transpose(X).dot(sigma)
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to gradient
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

