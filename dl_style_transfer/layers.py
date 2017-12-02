from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import math


def batch_norm(x, phase_train, decay=0.9, custom_inits=None, scope=None):
    """Creates a batch normalization layer.
    Used to stabilize distribution of outputs from a layer. Typically used right before a non-linearity. Works on
    n-dimensional data.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Ref.: https://arxiv.org/pdf/1502.03167.pdf
    Args:
        x:            Tensor, shaped `[batch, features...]` allowing it do be used with Conv or FC layers of any shape.
        phase_train:  Boolean tensor, true indicates training phase.
        decay:        Float. The decay to use for the exponential moving average.
        custom_inits: Dict from strings to functions that take a shape and return a tensor. These functions
                      are used to initialize the corresponding variable. If a variable is not in the dict, then it is
                      initialized with the default initializer for that variable. If `None` use default initializers.
        scope:        String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        normed, vars: `normed` is a tensor of the batch-normalized features, and has same shape as `input`.
                      `vars` is a dict of the variables.
    """
    x = tf.convert_to_tensor(x)
    x_shape = x.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    # Check to ensure the minimum shape is met and there are no unknown dims in important places
    if None in x_shape[1:] or len(x_shape) < 2:
        raise ValueError("`x.shape` must be [batch, ..., features].")

    with tf.variable_scope(scope, default_name='BN'):
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
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(x_shape) - 1)), name='Moments')

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
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name='Normalized')
    return normed, {'Beta': beta, 'Gamma': gamma}


bn = batch_norm


def dropout(x, phase_train, keep_prob=0.75, scope=None):
    """Creates a dropout layer.
    Used to regularize Conv and FC layers by preventing co-adaptation of neurons. Works on n-dimensional data.
    Performs train-time scaling rather then inference-time scaling.
    Ref.: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    Args:
        x:           Tensor, shaped `[batch, features...]` allowing it do be used with Conv or FC layers of any shape.
        phase_train: Boolean tensor, true indicates training phase. If false, no neurons are dropped.
        keep_prob:   Float, probability of dropping a neuron.
        scope:       String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        A tensor with the neurons dropped.
    """
    with tf.variable_scope(scope, default_name='Dropout'):
        def train():
            # Figure out `shape` tensor for the dropout function
            dims = tf.unstack(tf.shape(x))
            dims[1:-1] = [1] * (len(dims) - 2)  # set all spatial dims to 1 so that they are all dropped together
            shape = tf.stack(dims)  # example: [batch, 1, 1, features]
            return tf.nn.dropout(x, keep_prob, shape, name='Train')

        def test():
            return tf.identity(x, name='Inference')

        return tf.identity(tf.cond(phase_train, train, test), name='Dropped')


drop = dropout


def convolutional(x, num_features, size=3, activation=tf.nn.leaky_relu, phase_train=None, custom_inits=None, scope=None):
    """"Creates a convolutional Layer.
    Works on n spatial dimensions, as long as 1<=n<=3 due to limitations in `tf.nn.convolution`. Optionally performs
    batch normalization and also intelligently initializes the weights via a xavier initializer by default.
    Args:
        x:            Tensor, shaped `[batch, spatial..., features]`. Can have 1<=n<=3 spatial dimensions.
        num_features: The size of the feature dimension. This is the number of filters/neurons the layer will use.
        size:         The size of each convolutional filter.
        activation:   The activation function to use. If `None`, the raw scores are returned.
        phase_train:  If not `None`, then the scores will be put through a batch norm layer before getting fed into the
                      activation function. In that case, this will be a scalar boolean tensor indicating if the model
                      is currently being trained or if it is inference time.
        custom_inits: The initializer to use for the weights of the kernel, and any other variables such as those in the
                      batch norm layer, if `phase_train` has been specified. If `None` then default parameters are used.
        scope:        String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        output, vars: `output` is a tensor of the output of the layer.
                      `vars` is a dict of the variables, including those in the batch norm layer if present.
    """
    input_shape = x.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    num_spatial = (len(input_shape) - 2)
    if num_spatial < 1 or num_spatial > 3 or (None in input_shape[1:]):
        raise ValueError("`x.shape` must be [batch, spatial..., features], with 1<=n<=3 spatial dims.")

    if not x.dtype.is_floating:
        raise ValueError("`x` must be floating point.")

    with tf.variable_scope(scope, default_name='Conv'):
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

        convolved = tf.nn.convolution(x, kernel, padding="SAME", name='Conv')

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


def _upscale(x, factor=2, scope=None):
    """Upscales the shape of a N-Dimensional tensor by duplicating adjacent entries.
    Args:
        x:      The tensor to upscale. Only the spatial dimensions (all but first and last dim) are upscaled.
        factor: The factory by which to upscale. This should be an integer.
        scope:  String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        The upscaled tensor. Shape is multiplied by `factor` on each spatial dimension (all but first and last dim).
    """
    x = tf.convert_to_tensor(x)
    with tf.variable_scope(scope, default_name='Upscale'):
        num_spatial = len(x.shape) - 2
        # Iterate over spatial dims in reverse order
        for dim in range(num_spatial, 0, -1):
            x_shape = x.shape.as_list()

            x = tf.expand_dims(x, dim + 1)

            tile_multiples = [1] * len(x.shape)
            tile_multiples[dim + 1] = factor
            x = tf.tile(x, tile_multiples)

            x_shape[0] = -1
            x_shape[dim] *= factor
            x = tf.reshape(x, x_shape)
        return x


def pool(x, compute_mask=True, pool_type="MAX", size=2, scope=None):
    """Creates a pooling layer.
    Will work on N-Dimensional data. Can also compute a mask to indicate the selected pooling indices for max pooling.
    Args:
        x:            The tensor to perform pooling on. Should have shape `[batch, ..., features]` where the middle dims
                      are spatial dimensions.
        compute_mask: Whether or not to compute a mask that indicates which indices were selected from `x`. Should only
                      be `True` when `pool_type` is "MAX".
        pool_type:    The type of pooling to use. Will be passed directly to `tf.nn.pool`. Should be "MAX" or "AVG".
        size:         The size of the pooling window to use.
        scope:        String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        A tensor of the pooled result if `compute_mask` is false, otherwise a tuple of `(pooled,mask)` where `mask` is
        a mask on `x` to identify which indices were selected in the max pooling operation.
    """

    x = tf.convert_to_tensor(x)
    if pool_type is not "MAX" and compute_mask:
        raise ValueError("`compute_mask` cannot be `True` if `pool_type` is not \"MAX\"")
    with tf.variable_scope(scope, default_name='Pool'):
        window_shape = [size]*(len(x.shape)-2)  # 2 for all spatial dims
        pooled = tf.nn.pool(x, window_shape=window_shape, pooling_type=pool_type, strides=window_shape, padding="SAME")

        if compute_mask:
            upscaled_pool = _upscale(pooled)
            mask = tf.equal(x, upscaled_pool)
            mask = tf.cast(mask, tf.float32, name='Mask')
            return pooled, mask
        else:
            return pooled


def unpool(x, mask, factor=2, scope=None):
    """Creates an unpooling layer.
    Unpooling takes `x` and upscales it, putting zeros in all locations except the indices selected in `mask`. Will work
    on N-Dimensional data. The method is described in detail in the SegNet paper by Badrinarayanan et. al.
    Ref.: https://arxiv.org/abs/1511.00561
    Args:
        x:      The tensor to perform unpooling on. Should have shape `[batch, ..., features]` where the middle
                dims are spatial dimensions.
        mask:   A boolean tensor that indicates which indices in the output should be non-zero. Same shape as the
                output tensor.
        factor: The factor by which to upscale `x` when unpooling.
        scope:  String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        A tensor for the result of the unpooling. Will have the same shape as `x` but the spatial dims will be
        multiplied by `factor`.
    """
    x = tf.convert_to_tensor(x)
    with tf.variable_scope(scope, default_name='Unpool'):
        upscaled = _upscale(x, factor)
        output = tf.multiply(upscaled, mask, name='Unpooled')  # Force all but the indices selected in `mask` to 0
        return output


def fully_connected(x, num_features, activation=tf.nn.leaky_relu, phase_train=None, custom_inits=None, scope=None):
    """Creates a fully connected (dense) layer.
    Optionally performs batch normalization and also intelligently initializes weights. Will flatten `input` correctly.
    Args:
        x:            Tensor, shaped `[batch, features...]` allowing it do be used with Conv or FC layers of any shape.
        num_features: The number of features that the layer will output.
        activation:   The activation function to use. If `None`, the raw scores are returned.
        phase_train:  If not `None`, then the scores will be put through a batch norm layer before getting fed into the
                      activation function. In that case, this will be a scalar boolean tensor indicating if the model
                      is currently being trained or if it is inference time.
        custom_inits: The initializer to use for the weights, and any other variables such as those in the batch norm
                      layer, if `phase_train` has been specified. If `None` then default parameters are used.
        scope:        String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        output, vars: `output` is a tensor of the output of the layer.
                      `vars` is a dict of the variables, including those in the batch norm layer if present.
    """
    input_shape = x.shape.as_list()  # Don't deal with a TensorShape and instead use a list

    if len(input_shape) < 2 or (None in input_shape[1:]):
        raise ValueError("`x` must have shape [batch, features...]")

    with tf.variable_scope(scope, default_name='FC'):

        # Define default initializers for the weights. These are functions from shape to tensor.
        inits = {
            'Weights': lambda shape: xavier_initializer(shape=shape),
            'Bias': lambda shape: xavier_initializer(shape=shape),
        }
        if custom_inits is not None:
            inits.update(custom_inits)  # Overwrite default inits with `custom_inits`

        # Flatten all but batch dims
        if len(input_shape) > 2:
            x = tf.reshape(x, [-1, np.prod(input_shape[1:])])

        weights = tf.get_variable('Weights', initializer=inits['Weights']([x.shape.as_list()[-1], num_features]))
        vars = {'Weights': weights}

        matmul = tf.matmul(x, weights)
        matmul = tf.reshape(matmul, [-1] + input_shape[1:-1] + [num_features])  # Reshape back into the original shape

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
        shape:   A tensor describing the shape of the output tensor.
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


def k_competitive(x, phase_train, k, alpha=0, epsilon=0.0001, scope=None):
    """Creates a k-competitive layer.
    A K-Competitive layer encourages neurons in the network to specialize by only allowing the `k` most active neurons
    to fire. It then takes the "energy" (L1 norm) of the weaker neurons and adds them to the active neurons. This allows
    the gradients of the weaker neurons to still be updated even when not selected as the strongest. This method works
    on n-dimensional data, with n>=0 spatial dimensions.
    Ref.: https://arxiv.org/abs/1705.02033
    Args:
        x:           Tensor, shaped `[batch, features...]` allowing it do be used with Conv or FC layers of any shape.
        phase_train: Boolean tensor, true indicates training phase.
        k:           The number of neurons to select as the strongest during training. This can be a float from (0,1) to
                     signify a fraction of neurons, or an int where `1<=k<=x.shape[-1]`
        alpha:       Sets the intensity of the energy hyper-parameter.
        epsilon:     A small number to perturb the divisor, avoiding division by zero.
        scope:       String or VariableScope to use as the scope. If `None`, use default naming scheme.
    Returns:
        A tensor the same shape as input_volume with the top k matrices preserved and the rest set to zero
    """
    x = tf.convert_to_tensor(x)

    with tf.variable_scope(scope, default_name='K-Comp'):
        if k <= 0:
            raise ValueError('`k` should be a float from (0,1) or an int larger than zero.')
        elif k < 1:
            k = math.ceil(k*x.shape[-1])  # Calculate how many neurons to drop if `k` is fraction.

        def train():
            # Get the shape and element-wise absolute value of the input volume
            shape = x.shape.as_list()
            ab = tf.abs(x)
            # Calculate the base energy by taking the reduced sum of the feature dimension (L1 norm)
            energy = tf.reduce_sum(ab, axis=-1)
            # Find the top k indices in the feature dimension for each spatial dimension
            _, ind = tf.nn.top_k(ab, k)
            # Produce and apply a mask marking the winning indices
            mask = tf.reduce_sum(tf.one_hot(ind, shape[-1], on_value=1.0, off_value=0.0, dtype=tf.float32), -2)
            masked = x * mask
            # Get the signs of `masked`. Add epsilon to avoid dividing by zero.
            signs = masked / (tf.abs(masked) + epsilon)
            # Calculate and add the energy term
            energy_term = tf.expand_dims(energy, -1) * signs
            return masked + alpha*energy_term

        def test():
            return x

        return tf.identity(tf.cond(phase_train, train, test), name='Output')


k_comp = k_competitive
