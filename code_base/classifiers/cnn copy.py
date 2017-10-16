from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    
    C, H, W = input_dim
    ConvPad = filter_size -1
    PoolStride = 2
    Ho_Conv1 = 1 + (H + ConvPad - filter_size)
    Wo_Conv1 = 1 + (W + ConvPad - filter_size)
    Ho_Pool1 = 1 + (Ho_Conv1 - 2) / PoolStride
    Wo_Pool1 = 1 + (Wo_Conv1 - 2) / PoolStride
    
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.random.normal(0,weight_scale,num_filters)
    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters*Ho_Pool1*Wo_Pool1,hidden_dim))
    self.params['b2'] = np.random.normal(0,weight_scale,hidden_dim)
    self.params['W3'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    self.params['b3'] = np.random.normal(0,weight_scale,num_classes)
    
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
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    
    scores = None

    conv_out, conv_cache = conv_forward(X, W1, b1, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward(relu_out, pool_param)
    affine_out, affine_cache = affine_forward(pool_out, W2, b2) #[[[FLATTEN??]]]
    relu_outII, relu_cacheII = relu_forward(affine_out)
    scores, out_cache = affine_forward(relu_outII, W3, b3)
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
    ############################################################################
    loss, dout = softmax_loss(X, y)
    dx_out, grads['W3'], grads['b3'] = affine_backward(dout, out_cache)
    dreluII = relu_backward(dx_out, relu_cacheII)
    dx_affine, grads['W2'], grads['b2'] = affine_backward(dreluII, affine_cache)
    dpool = max_pool_backward(dx_affine, pool_cache)
    drelu = relu_backward(dpool, relu_cache)
    dx, grads['W1'], grads['b1'] = conv_backward(drelu, conv_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
