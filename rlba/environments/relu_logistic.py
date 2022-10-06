import functools

# import Jax and related packages
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
import optax
import chex
from chex import assert_shape

import numpy as np
import matplotlib.pyplot as plt

from typing import Iterator, Sequence, Optional

from rlba.types import (
    Array,
    ArraySpec,
    BoundedArraySpec,
    DiscreteArraySpec,
    NestedArray,
    NestedArraySpec,
    NestedDiscreteArraySpec,
)

class ReLULogisticNetwork(hk.Module):
    """A neural network with ReLU hidden layers and logistic output layer."""
    def __init__(self, 
                layer_dims: Sequence[int], 
                weight_var: float = 1.0, 
                bias_var:float = 1.0, 
                name:Optional[str] = None
                ) -> None:

        """
        Parameters:
            layer_dims: array_like made of integers
                encodes the number of nodes in the input layer and in the
                hidden layer
            weight_var: float
                unnormalized variance of entries in the weight matrix 
            bias_var: float
                unnormalized variance of entries in the bias vector
            name: str or None
                name of the neural network that is being created
        """

        super(ReLULogisticNetwork, self).__init__(name=name)

        self.layer_dims = layer_dims
        self.weight_var = weight_var
        self.bias_var = bias_var

    def __call__(self, 
                x: chex.Array
                 ) -> chex.Array:
        """
        x: array_like
           the last dimension x.shape[-1] should be equal to the number of nodes
           in the input layer 
        """
        assert_shape(x, (..., self.layer_dims[0]))
        

        # loop over the hidden layers
        for i in range(1, len(self.layer_dims)):
            # neural network weights and biases initializer

            """
            notice that the variance for weights are 
            self.weight_var/self.layer_dims[i-1] which is 
            normalized with respect to the number of input nodes
            """
            w_init = hk.initializers.RandomNormal(self.weight_var/self.layer_dims[i-1])
            b_init = hk.initializers.RandomNormal(self.bias_var)

            w = hk.get_parameter('W_'+str(i), shape=[self.layer_dims[i], \
                                                    self.layer_dims[i-1]],\
                                                    dtype=x.dtype, init=w_init)

            b = hk.get_parameter('b_'+str(i), shape=[self.layer_dims[i]],\
                                dtype=x.dtype, init=b_init)

            x = jnp.tensordot(x, w, axes=((-1), (1))) + b
            x = jax.nn.relu(x)

        w_init = hk.initializers.RandomNormal(self.weight_var/self.layer_dims[-1])
        w = hk.get_parameter('W_'+str(len(self.layer_dims)), \
                            shape=[1, self.layer_dims[-1]],\
                            dtype=x.dtype, init=w_init)

        x = jnp.tensordot(x, w, axes=((-1), (1)))
        x = jnp.exp(x)
        x = jnp.squeeze(jnp.divide(x, 1+x), axis=-1)
        return x

"""a wrapper for ReLULogisticNetwork."""
def relu_logistic_fn(x:chex.Array, 
                     layer_dims: Sequence[int], 
                     weight_var: float = 1, 
                     bias_var: float = 1, 
                     name: Optional[str] = None):
    """
    Parameters:
        x: array_like
           the last dimension x.shape[-1] should be equal to the number of nodes
           in the input layer 
        layer_dims: array_like made of integers
            encodes the number of nodes in the input layer and in the
            hidden layer
        weight_var: float
            unnormalized variance of entries in the weight matrix 
        bias_var: float
            unnormalized variance of entries in the bias vector
        name: str or None
            name of the neural network that is being created
    """
  
    model = ReLULogisticNetwork(layer_dims, weight_var, bias_var, name=name)
    return model(x)

def get_opt_fns(layer_dims: Sequence[int], 
                weight_var: float = 1.0, 
                bias_var:float = 1.0, 
                name:Optional[str] = None):
    """
    Parameters:
        layer_dims: array_like made of integers
            encodes the number of nodes in the input layer and in the
            hidden layer
        weight_var: float
            unnormalized variance of entries in the weight matrix 
        bias_var: float
            unnormalized variance of entries in the bias vector
        name: str or None
            name of the neural network that is being created
    """

    # default name for the model
    if name is None:
        name = 'model'

    # initialize the model and prameters
    forward_partial = functools.partial(relu_logistic_fn, \
                                        layer_dims=layer_dims,\
                                        weight_var=weight_var,\
                                        bias_var=bias_var, \
                                        name=name)

    # "pure model function"
    model_fn = hk.transform(forward_partial)

    # a dictionary recording the prior variances at different layers for weights
    # and biases
    prior_var_dict = {}
    for i in range(1, len(layer_dims)):
        prior_var_dict['W_'+str(i)] = weight_var/layer_dims[i-1]
        prior_var_dict['b_'+str(i)] = bias_var

    prior_var_dict['W_'+str(len(layer_dims))] = weight_var/layer_dims[-1]

    @jax.jit
    def loss_fn(params: dict, 
                x: chex.Array, 
                y: chex.Array, 
                weights: Optional[Sequence[float]] = None, 
                reg_params: dict = None) -> float:

        """
        Parameters:
            params: dict
                dictionary of weights and biases for the model
            x: array_like
                features
            y: array_like
                labels for features
            weights: None or array_like
                weights for weighing likelihood of observed data
            reg_params: dictionary
                values act as means to regularize against
        """
        
        n_samples = len(y)

        # compute the loss associated with the log likelihood
        if n_samples > 0:
            y_hat = model_fn.apply(params, None, x)
            # vector encoding the negative log-likelihood
            loss_vec = -jnp.multiply(y, jnp.log(y_hat))-\
                jnp.multiply(1-y, jnp.log(1-y_hat))
            # reweigh the vector by random weights
            if weights is not None:
                loss_vec = jnp.multiply(weights, loss_vec)
            # average
            loss_ll = jnp.mean(loss_vec)
        else:
            loss_ll = 0

        # quadratic loss (from prior) for all weights and biases
        loss_prior = 0
        for key in prior_var_dict:
            if reg_params is None:
                weight_diff = params[name][key]
            else:
                weight_diff = params[name][key] - reg_params[name][key]
                loss_prior = loss_prior + \
                    jnp.sum(jnp.square(weight_diff))/(2*prior_var_dict[key])

        # normalize the loss_prior based on number of samples,
        # since the log-likelihood part is an average
        loss_prior = loss_prior/n_samples
        loss = loss_ll + loss_prior

        return loss

    grad_loss_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    return grad_loss_fn, model_fn

def opt_pert_loss(hk_keys: hk.PRNGSequence, 
                  grad_loss_fn, 
                  model_fn, 
                  n_epochs: int,
                  learning_rate: float,
                  x: chex.Array,
                  y: chex.Array,
                  perturb: bool = False,
                  weights: Optional[Sequence[float]] = None, 
                  reg_params: Optional[dict] = None, 
                  init_params: Optional[dict] = None,
                  verbose: bool = False):
    
    """
    Parameters:
        hk_keys: hk.PRNGSequence
            pseudo-random key sequence used for statistically safe random number
            generation
        grad_loss_fn: jax compiled function
            provides value of the loss function along with the gradient for the
            inputs
        model_fn: output of hk.transform
            forward function that predicts y given x
        n_epochs: int
            number of epochs we train our model for
        learning_rate: float
            non-negative number for the optimizer
        x: array_like
            features
        y: array_like
            labels for features
        perturb: bool
            perturb is True if the data is to be perturbed
        weights: None or array_like
            weights for weighing likelihood of observed data
        reg_params: dictionary
            values act as means to regularize against
        init_params: dictionary
            initialization values for the model
        verbose: bool
            if true, loss is printed once every 10 steps
    """

    n_samples = len(y)

    if perturb:
        if weights is None:
            weights = jax.random.exponential(next(hk_keys), shape=[n_samples])

        if reg_params is None:
            reg_params = model_fn.init(next(hk_keys), x)
    else:
        weights = None
        reg_params = None

    if init_params is None:
        init_params = model_fn.init(next(hk_keys), x)

    # Construct a simple Adam optimiser using the transforms in optax.
    # You could also just use the `optax.adam` alias, but we show here how
    # to do so manually so that you may construct your own `custom` optimiser.
    opt_init, opt_update = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-learning_rate)
    )

    opt_state = opt_init(init_params)
    params = init_params

    for i in range(n_epochs):
        loss, grads = grad_loss_fn(params, x, y, weights, reg_params)

    if verbose and i%10 == 9:
        print('For step {}, the current loss is {}'.format(i, loss))

    # transform the gradient
    updates, opt_state = opt_update(grads, opt_state, params)
    # actual update
    params = optax.apply_updates(params, updates)

    def trained_model(xx):
        yy = model_fn.apply(params, None, xx)
        return yy

    return trained_model, params, weights, reg_params

class ReLULogisticBandit(object):
    """The environment ReLULogisticBandit."""
    def __init__(self, 
                seed: int, 
                n_contexts: int, 
                n_actions: int, 
                layer_dims: Sequence[int], 
                weight_var: float=1., 
                bias_var: float=1.,
                feature_var: float=1.):
        assert len(layer_dims) >= 1

        # set the random seed (key) for Jax/Haiku
        self._hk_keys = hk.PRNGSequence(jax.random.PRNGKey(seed))

        # record the properties
        self._n_contexts = n_contexts
        self._n_actions = n_actions

        self._layer_dims = layer_dims
        self._weight_var = weight_var
        self._bias_var = bias_var
        self._feature_var = feature_var
        self._input_dim = layer_dims[0]

        self._action_spec = DiscreteArraySpec(n_actions, name="action spec")
        self._observation_spec: ArraySpec = {
            'reward': BoundedArraySpec(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=1,
                name="reward",
            ),
            'context': BoundedArraySpec(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=self._n_contexts-1,
                name="context",
            ),
            'feature': BoundedArraySpec(
                shape=(self._n_actions, self._input_dim),
                dtype=float,
                minimum=0,
                maximum=jnp.inf,
                name="feature",
            ), 
        }

        self._reset()

    def _reset(self):
        self._context = None
        self._prev_context = None
        
        # construct features
        self._feature = jnp.sqrt(self._feature_var)*jax.random.normal(
            next(self._hk_keys), (self._n_contexts, self._n_actions, 
            self._input_dim))

        # initialize the model and prameters
        forward_partial = functools.partial(relu_logistic_fn, \
                                            layer_dims=self._layer_dims,\
                                            weight_var=self._weight_var,\
                                            bias_var=self._bias_var, \
                                            name="env_model")

        self._model_fn = hk.transform(forward_partial)

        self._params = self._model_fn.init(next(self._hk_keys), self._feature)

        # self.model can evaluate the prediction
        self._model = lambda x: self._model_fn.apply(self._params, None, x)

        # expected reward for each context-action pair
        # self._exp_reward.shape == (n_contexts, n_actions)
        self._exp_reward = self._model(self._feature)
        self._optimal_exp_reward = jnp.max(self._exp_reward, axis=1, keepdims=True)

        self._sample_new_context()

    def _sample_new_context(self):
        self._context = int(jax.random.randint(
                next(self._hk_keys), shape=[1], minval=0,
                maxval=self._n_contexts)[0])
        
    def _validate_action(self, action: NestedArray) -> int:
        try:
            action = int(action)
        except TypeError:
            TypeError("Action does not seem to be convertible to an int")
        if action >= self._action_spec.num_values:
            raise ValueError("action is larger than number of available arms.")
        return action

    def _validate_context(self, context: int) -> int:
        if context >= self._n_contexts:
            raise ValueError("context is larger than number of available contexts.")
        return context

    def _get_context(self):
        return self._context

    def _get_feature(self, context=None):
        assert self._feature is not None, "Please reset the environment first"
        if context is None:
            return self._feature
        else:
            context = self._validate_context(context)
            return self._feature[context, :, :]

    def step(self, action: NestedArray):
        """Step the environment according to the action and returns an `observation`.
        Args:
          action: an integer corresponding to the arm index.
        Returns:
          An `Observation` A NumPy array of bools. Must conform to the
              specification returned by `observation_spec()`.
        """
        action = self._validate_action(action)
        r_mean = self._exp_reward[self._context, action]
        inst_reward = int(jax.random.bernoulli(next(self._hk_keys), r_mean))

        self._prev_context = self._get_context()
        self._sample_new_context()
        context = self._get_context()
        feature = self._get_feature(context)

        return {'reward': inst_reward, 'context': context, 
                'feature': feature}

    def expected_reward(self, action: NestedArray):
        assert self._exp_reward is not None, "Please reset the environment first"
        action = self._validate_action(action)
        context = self._prev_context
        return float(self._exp_reward[context, action])

    def optimal_expected_reward(self):
        context = self._prev_context
        return float(self._optimal_exp_reward[context])

    def close(self):
        """Frees any resources used by the environment.
        Implement this method for an environment backed by an external process.
        This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```
        or via a context manager
        ```python
        with Env(...) as env:
          # Use env.
        ```
        """
        pass

    @property
    def observation_spec(self) -> NestedArraySpec:
        """Defines the observations provided by the environment.
        Returns:
          An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return self._observation_spec

    @property
    def action_spec(self) -> NestedDiscreteArraySpec:
        """Defines the actions that should be provided to `step`.
        Returns:
          A `DiscereteArray` spec, or a nested dict, list or tuple of `DiscreteArray` specs.
        """
        return self._action_spec

    def __enter__(self):
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Allows the environment to be used in a with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        self.close()