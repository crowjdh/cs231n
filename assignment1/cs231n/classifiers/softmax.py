import numpy as np
from random import shuffle
from past.builtins import xrange

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

  train_count = X.shape[0]
  class_count = W.shape[1]

  all_scores = X.dot(W)
  all_scores -= all_scores.max()

  loss = 0.0
  for i in xrange(train_count):
    yi = y[i]
    scores = all_scores[i]
    score_sum = np.exp(scores).sum()
    p = lambda idx: np.exp(scores[idx]) / score_sum

    prob = p(yi)# np.exp(scores[yi]) / np.exp(scores).sum()
    loss += -np.log(prob)

    for j in xrange(class_count):
      pj = p(j)
      dW[:, j] += (pj - (j == yi)) * X[i]

  loss /= train_count
  loss += reg * np.sum(W * W)
  dW /= train_count
  dW += reg * 2 * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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

  train_count = X.shape[0]
  class_count = W.shape[1]

  all_scores = X.dot(W)
  all_scores -= all_scores.max()

  sum_all_scores = np.transpose([np.exp(all_scores).sum(axis=1)])
  probabilities = np.exp(all_scores) / sum_all_scores
  losses = -np.log(probabilities)
  loss = losses[np.arange(losses.shape[0]), y].sum()
  loss /= train_count
  loss += reg * np.sum(W * W)

  probabilities[np.arange(probabilities.shape[0]), y] -= 1
  dW = X.T.dot(probabilities)

  dW /= train_count
  dW += reg * 2 * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

