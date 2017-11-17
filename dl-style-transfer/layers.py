from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import math


def batch_norm(input, phase_train, decay=0.9, custom_inits=None, scope='BN'):
    """Creates a batch normalization layer.

    Used to stabilize distribution of outputs from a layer. Typically used right before a non-linearity. Works on
    n-dimensional data.

    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Ref.: https://arxiv.org/pdf/1502.03167.pdf

    Args:
        input:         Tensor,  shaped [batch, features...] allowing it do be used with Conv or FC layers of any shape.
        phase_train:   Boolean tensor, true indicates training phase.
        decay:         Float. The decay to use for the exponential moving average.
        custom_inits:  Dict from strings to functions that take a shape and return a tensor. These functions
                       are used to initialize the corresponding variable. If a variable is not in the dict, then it is
                       initialized with the default initializer for that variable. If None, then default initial
        scope:         String or VariableScope to use as the variable scope.
    Returns:
        normed, vars:  `normed` is a tensor of the batch-normalized features, and has same shape as `input`.
                       `vars` is a dict of the variables.
    """
    x_shape = input.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    # Check to ensure the minimum shape is met and there are no unknown dims in important places
    if None in x_shape[1:] or len(x_shape) < 2:
        raise ValueError("`input.shape` must be [batch, ..., features].")

    with tf.variable_scope(scope):
        # Define default initializers for beta and gamma. These are functions from shape to tensor.
        inits = {
            'Beta': lambda shape: tf.constant(0., shape=shape),
            'Gamma': lambda shape: tf.constant(1., shape=shape),
        }
        if custom_inits is not None:
            inits.update(custom_inits)  # Overwrite default inits with `custom_inits`

        beta = tf.get_variable('Beta', initializer=inits['Beta']([x_shape[-1]]))  # Learned mean
        gamma = tf.get_variable('Gamma', initializer=inits['Gamma']([x_shape[-1]]))  # Learned std. dev.

        # Get mean and variance over batch and spatial dims
        batch_mean, batch_var = tf.nn.moments(input, list(range(len(x_shape) - 1)), name='Moments')

        # We want to figure out mean and variance of whole dataset while training, so we use a moving average
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            """Updates the moving average and returns the batch mean and variance."""
            ema_apply_op = ema.apply([batch_mean, batch_var])  # Update moving average

            # There is no dependency in the graph for batch_mean on updating the average, so we add one.
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(
            phase_train,
            mean_var_with_update,  # If training, use batch mean and var
            lambda: (ema.average(batch_mean), ema.average(batch_var)))  # If inference, use averaged mean
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3, name='Normalized')
    return normed, {'Beta': beta, 'Gamma': gamma}


bn = batch_norm


def dropout(input, phase_train, keep_prob=0.75, scope='Dropout'):
    """Creates a dropout layer.

    Used to regularize Conv and FC layers by preventing co-adaptation of neurons. Works on n-dimensional data.
    Performs train-time scaling rather then inference-time scaling.

    Ref.: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

    Args:
        input:           Tensor, shaped [batch, features...] allowing it do be used with Conv or FC layers of any shape.
        phase_train: Boolean tensor, true indicates training phase. If false, no neurons are dropped.
        keep_prob:   Float, probability of dropping a neuron.
        scope:       String or VariableScope to use as the variable scope.
    Returns:
        A tensor with the neurons dropped.
    """
    with tf.variable_scope(scope):
        def train():
            # Figure out `shape` tensor for the dropout function
            dims = tf.unstack(tf.shape(input))
            dims[1:-1] = [1] * (len(dims) - 2)  # set all spatial dims to 1 so that they are all dropped together
            shape = tf.stack(dims)  # example: [batch, 1, 1, features]
            return tf.nn.dropout(input, keep_prob, shape, name='Train')

        def test():
            return tf.identity(input, name='Inference')

        return tf.identity(tf.cond(phase_train, train, test), name='Dropped')


drop = dropout


def convolutional(input, num_features, size=3, activation=tf.nn.relu, phase_train=None, custom_inits=None, scope='Conv'):
    """"Creates a convolutional Layer.

    Works on n spatial dimensions, as long as 1<=n<=3. Optionally performs batch normalization and also intelligently
    initializes the weights by using a xavier initializer by default.

    Args:
        input:        Tensor, shaped [batch, spatial..., features]. Can have 1<=n<=3 spatial dimensions.
        num_features: The size of the feature dimension. This is the number of filters/neurons the layer will use.
        size:         The size of each convolutional filter.
        activation:   The activation function to use. If `None`, the raw scores are returned.
        phase_train:  If not `None`, then the scores will be put through a batch norm layer before getting fed into the
                      activation function. In that case, this will be a scalar boolean tensor indicating if the model
                      is currently being trained or if it is inference time.
        custom_inits: The initializer to use for the weights of the kernel, and any other variables such as those in the
                      batch norm layer, if `phase_train` has been specified. If `None` then default parameters are used.
        scope:        String or VariableScope to use as the variable scope.
    Returns:
        output, vars: `output` is a tensor of the output of the layer.
                      `vars` is a dict of the variables, including those in the batch norm layer if present.
    """
    input_shape = input.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    num_spatial = (len(input_shape) - 2)
    if num_spatial < 1 or num_spatial > 3 or (None in input_shape[1:]):
        raise ValueError("`input.shape` must be [batch, spatial..., features], with 1<=n<=3 spatial dims.")

    if not input.dtype.is_floating:
        raise ValueError("`input` must be floating point.")

    with tf.variable_scope(scope):
        # Figure out `kernel_shape`
        kernel_shape = [size]*num_spatial
        kernel_shape += [input_shape[-1], num_features]  # example: [size, size, input_features, num_features]
        kernel_shape = kernel_shape

        # Define default initializers for the kernel and bias. These are functions from shape to tensor.
        inits = {
            'Kernel': lambda shape: xavier_initializer(shape=shape),
            'Bias': lambda shape: xavier_initializer(shape=shape),
        }
        if custom_inits is not None:
            inits.update(custom_inits)  # Overwrite default inits with `custom_inits`

        kernel = tf.get_variable('Kernel', initializer=inits['Kernel'](kernel_shape))
        vars = {'Kernel': kernel}

        convolved = tf.nn.convolution(input, kernel, padding="SAME", name='Conv')

        # Do batch norm?
        if phase_train is not None:
            scores, bn_vars = batch_norm(convolved, phase_train, custom_inits=inits.get('BN'))
            vars['BN'] = bn_vars
        else:
            # If we aren't doing batch norm, we need a bias!
            bias = tf.get_variable('Bias', initializer=inits['Bias']([num_features]))
            vars['Bias'] = bias
            scores = convolved + bias

        # Do activation function?
        if activation is not None:
            result = activation(scores)
        else:
            result = scores

        return tf.identity(result, 'Output'), vars


conv = convolutional


def fully_connected(input, num_features, activation=tf.nn.relu, phase_train=None, custom_inits=None, scope='FC'):
    """Creates a fully connected (dense) layer.

    Optionally performs batch normalization and also intelligently initializes weights. Will flatten `input` correctly.

    Args:
        input:        Tensor of shape `[batch, features]` that this FC layer uses as its input features.
        num_features: The number of features that the layer will output.
        activation:   The activation function to use. If `None`, the raw scores are returned.
        phase_train:  If not `None`, then the scores will be put through a batch norm layer before getting fed into the
                      activation function. In that case, this will be a scalar boolean tensor indicating if the model
                      is currently being trained or if it is inference time.
        custom_inits: The initializer to use for the weights, and any other variables such as those in the batch norm
                      layer, if `phase_train` has been specified. If `None` then default parameters are used.
        scope:        String or VariableScope to use as the variable scope.
    Returns:
        output, vars: `output` is a tensor of the output of the layer.
                      `vars` is a dict of the variables, including those in the batch norm layer if present.
    """
    input_shape = input.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    if len(input_shape) < 2 or (None in input_shape[1:]):
        raise ValueError("`input` must have shape [batch, features...]")

    with tf.variable_scope(scope):

        # Define default initializers for the weights. These are functions from shape to tensor.
        inits = {
            'Weights': lambda shape: xavier_initializer(shape=shape),
            'Bias': lambda shape: xavier_initializer(shape=shape),
        }
        if custom_inits is not None:
            inits.update(custom_inits)  # Overwrite default inits with `custom_inits`

        # Flatten all but batch dims
        if len(input_shape) > 2:
            input_shape = [input_shape[0], np.prod(input_shape[1:])]
            input = tf.reshape(input, [-1, input_shape[1]])

        weights = tf.get_variable('Weights', initializer=inits['Weights']([input_shape[-1], num_features]))
        vars = {'Weights': weights}

        matmul = tf.matmul(input, weights)

        # Do batch norm?
        if phase_train is not None:
            scores, bn_vars = batch_norm(matmul, phase_train, custom_inits=inits.get('BN'))
            vars['BN'] = bn_vars
        else:
            # If we aren't doing batch norm, we need a bias!
            bias = tf.get_variable('Bias', initializer=inits['Bias']([num_features]))
            vars['Bias'] = bias
            scores = matmul + bias

        # Do activation function?
        if activation is not None:
            result = activation(scores)
        else:
            result = scores

        return tf.identity(result, 'Output'), vars


fc = fully_connected


def xavier_initializer(shape, uniform=True, dtype=tf.float32, name='Xavier-Initializer'):
    """Outputs a random tensor initialized with the Xavier initializer. Ensures a variance of 1.

    This was already implemented in tf.contrib.layers, but it has been re-implemented here for simplification.
    Ref.: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Args:
        shape:  A tensor describing the shape of the output tensor.
        uniform: If a uniform or normal distribution should be used. Uniform performs slightly better empirically.
        dtype:   The type of tensor. Should be floating point.
        name:    The name of the initializer
    Returns:
        A tensor generated with the Xavier initializer.
    """
    shape = tf.convert_to_tensor(shape)  # Allow us to accept non-tensor inputs

    dims = shape.shape[0]
    if dims is 0:
        raise ValueError("`shape` cannot be a scalar!")
    elif dims is 1:
        # Need to clarify if this is correct, but it is how TensorFlow did it.
        n_avg = shape[0]
    else:
        # n_out is the number of neurons/filters/features
        n_out = shape[-1]
        # n_in is the number of weights associated with a neuron. The following code works both for Conv and FC layers.
        n_in = tf.reduce_prod(shape[:-1])
        n_avg = tf.cast(n_in + n_out, dtype) / 2.0

    if uniform:
        limit = tf.sqrt(3.0 / n_avg)
        return tf.random_uniform(shape, -limit, limit, dtype=dtype, name=name)
    else:
        stddev = tf.sqrt(1.3 / n_avg)  # This corrects for the fact that we are using a truncated normal distribution.
        return tf.truncated_normal(shape, stddev=stddev, dtype=dtype, name=name)


def k_sparse_transform(input_volume, k=2, method=1, name="K-Sparse"):
    """
    Takes an input volume and calculates the norm of each matrix along the feature axis. The top k of these
    are declared the "winners". The others lose and are set to zero.

    Args:
        input_volume: The 3d tensor to be transformed
        k: the number of winners to declare
        method: the type of norm to calculate
        name: the name of the returned tensor
    Returns:
        A tensor the same shape as input_volume with the top k matrices preserved and the rest set to zero
    """
    # Calculate the norms for all Activation maps in the volume using method
    norms = tf.norm(input_volume, ord=method, axis=(1, 2))
    z = tf.zeros(tf.shape(input_volume))
    # Get the top k indices
    _, top = tf.nn.top_k(norms, k=k)
    # Mask indices not in the top k
    top_mask = [i in top for i in range(tf.shape(input_volume)[-1])]
    # Set not top values to zero. Tile 1D mask to three dimensions
    return tf.where(tf.tile(top_mask, tf.concat(tf.shape(input_volume)[:-1], [1])), input_volume, z, name=name)


def k_competitive_activator(input_volume, alpha=0, k=2, method=1, name="K-Competitive"):
    """
    Takes an input volume and calculates the norm of each matrix along the feature axis. The top k of these
    are declared the "winners". The others lose and are set to zero. An amplification term is added to the winners
    proportional to alpha.

    Args:
        input_volume: The 3d tensor to be transformed
        k: the number of winners to declare
        alpha: Coefficient on the amplification term
        method: the type of norm to calculate
        name: the name of the returned tensor
    Returns:
        A tensor the same shape as input_volume with the top k matrices amplified and the rest set to zero
    """
    # Take the tan of the volume
    tan_vol = tf.tanh(input_volume)
    norms = tf.norm(tan_vol, ord=method, axis=(1, 2))
    tans = tf.tanh(norms)
    pos = tf.where(tf.greater(tans, 0), tans, tf.zeros(tf.shape(tans)))
    neg = tf.where(tf.less(tans, 0), tans, tf.zeros(tf.shape(tans)))

    _, top_pos = tf.nn.top_k(pos, k=math.ceil(k/2))
    _, top_neg = tf.nn.top_k(-neg, k=math.floor(k/2))

    top_pos_mask = [i in top_pos for i in range(tf.shape(input_volume)[-1])]
    top_neg_mask = [i in top_neg for i in range(tf.shape(input_volume)[-1])]

    pos_vol = tf.where(tf.tile(top_pos_mask, tf.concat(tf.shape(tan_vol)[:-1], [1])), tan_vol, z)
    neg_vol = tf.where(tf.tile(top_neg_mask, tf.concat(tf.shape(tan_vol)[:-1], [1])), tan_vol, z)

    pos_adder = tf.reduce_sum(pos_vol, axis=2)
    neg_adder = tf.reduce_sum(neg_vol, axis=2)

    return tf.add(pos_vol + alpha * pos_adder, neg_vol + alpha * neg_adder, name=name)


