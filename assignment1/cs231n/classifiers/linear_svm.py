import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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

  # Initialise gradient and loss to zero
  dW = np.zeros(W.shape)
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Compute loss and  gradient
  for i in xrange(num_train):	# For each image in training
    scores = X[i].dot(W)	# Calculate scores, s = Wx
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):	# For each class

      if j == y[i]:	# No loss computed if correctly classified
        continue

      # Calculate margin, delta = 1
      margin = scores[j] - correct_class_score + 1

      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:] 
        dW[:,y[i]] -= X[i,:] 

  # Average loss and gradient over batch
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  # Initialize the gradient and loss as zero
  loss = 0.0
  dW = np.zeros(W.shape)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # Dimensions
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_score = scores[range(num_train), list(y)].reshape(-1,1)

  margins = np.maximum(0, scores - correct_class_score + 1)

  # Zero-out margins associated with correct class scores
  margins[range(num_train), list(y)] = 0	

  data_loss = np.sum(margins) / num_train	# Sum margins & average over batch
  reg_loss = 0.5 * reg * np.sum(W * W)

  loss = data_loss + reg_loss

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # Create coefficient matrix of size (500 X 10)
  # Determine which elements have margins < 0
  # Sum number of classes that didn't meet margin condition
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  # Multiply coefficient matrix with transpose of input
  # This selects inputs to be added to gradient matrix based on margin conditions
  dW = np.dot(X.T, coeff_mat)

  # Average gradient over batch
  dW /= num_train

  # Add regularization to the loss and gradient
  dW += 2*reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
