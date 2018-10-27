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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Compute loss and  gradient
  for i in range(num_train):	# For each image in training
    scores = X[i].dot(W)	# Calculate scores, s = Wx
    shift_scores = scores - np.max(scores)

    loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
    loss += loss_i

    for j in xrange(num_classes):	# For each class
      softmax_output = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))

      if j == y[i]:
        dW[:,j] += (softmax_output - 1) * X[i,:] 
      else:
        dW[:,j] += softmax_output * X[i,:] 

  # Average loss and gradient over batch
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient
  loss += reg * np.sum(W * W)
  dW += 2*reg*W


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

  # Dimensions
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)

  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1,1)

  # Calculate loss only from correct class
  data_loss_i = -np.log(softmax_output[range(num_train), list(y)])
  data_loss = np.sum(data_loss_i)

  reg_loss = 0.5 * reg * np.sum(W * W)

  loss = data_loss + reg_loss

  loss /= num_train

  # Gradient
  dS = softmax_output.copy()
  dS[range(num_train), list(y)] -= 1
  dW = np.dot(X.T, dS)
  dW += 2*reg*W
  dW /= num_train


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

