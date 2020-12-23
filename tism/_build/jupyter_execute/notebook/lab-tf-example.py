
Lab: `tensoflow` 範例
================

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

## Linear Regression

# define a function to generate x and y
def generate_linear_reg_data(
    n_sample, weight, intercept = 0, sd_residual = 1,
    dtype = tf.float64, seed = None):
    weight = tf.constant(weight, dtype = dtype)
    weight = tf.reshape(weight, shape = (-1, 1))
    n_feature = weight.shape[0]
    x = tf.random.normal(shape = (n_sample, n_feature),
                         seed = seed, dtype = dtype)
    e = tf.random.normal(shape = (n_sample, 1),
                         seed = seed, dtype = dtype)
    y = intercept + x @ weight + e
    return x, y

# run generate_data
n_sample = 10000
weight_true = [-1, 2, 0]
dtype = tf.float64

x, y = generate_linear_reg_data(
    n_sample = n_sample, weight = weight_true,
    intercept = 0, sd_residual = 1,
    dtype = dtype, seed = 48)

# start optimization
n_feature = len(weight_true)
learning_rate = .1
epochs = 500
tol = 10**(-4)

optimizer = tf.optimizers.SGD(learning_rate = learning_rate)

intercept = tf.Variable(tf.zeros((), dtype = dtype), 
                        name = "intercept")
weight = tf.Variable(tf.zeros((n_feature, 1), dtype = dtype), 
                     name = "weight")

for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        y_hat = intercept + x @ weight
        loss_value = tf.reduce_mean((y - y_hat)**2)
    gradients = tape.gradient(loss_value, [intercept, weight])
    optimizer.apply_gradients(zip(gradients, [intercept, weight]))
    #print(weight)
    if (tf.reduce_max(
            [tf.reduce_mean(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

print("intercept", intercept.numpy())
print("weight", weight.numpy())

## Logistic Regression

# define a function to generate x and y
def generate_logistic_reg_data(
    n_sample, weight, intercept = 0, 
    dtype = tf.float64, seed = None):
    weight = tf.constant(weight, dtype = dtype)
    weight = tf.reshape(weight, shape = (-1, 1))
    n_feature = weight.shape[0]
    x = tf.random.normal(shape = (n_sample, n_feature),
                         seed = seed, dtype = dtype)
    logits = intercept + x @ weight
    y = tfd.Bernoulli(logits=logits, dtype=dtype).sample()
    return x, y

# run generate_data
n_sample = 10000
weight_true = [-1, 2, 0]
dtype = tf.float64

x, y = generate_logistic_reg_data(
    n_sample = n_sample, 
    weight = weight_true,intercept = 0, 
    dtype = dtype, seed = 48)

# define a tf.Module to collect parameters
class LinearModel(tf.Module):
  def __init__(self, n_feature, dtype = tf.float64):
    super().__init__()
    self.weight = tf.Variable(tf.zeros((n_feature, 1), 
                                       dtype = dtype), 
                              name = "weight")
    self.intercept = tf.Variable(tf.zeros((), dtype = dtype), 
                                 name = "intercept")
  def __call__(self, x):
    return self.intercept + x @ self.weight

n_feature = len(weight_true)
learning_rate = .5
epochs = 500
tol = 10**(-4)

linear_model = LinearModel(n_feature, dtype)
optimizer = tf.optimizers.SGD(learning_rate = learning_rate)

for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        logits = linear_model(x)
        loss_value = - tf.reduce_mean(
            tfd.Bernoulli(logits=logits).log_prob(y))
    gradients = tape.gradient(
        loss_value, linear_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, linear_model.trainable_variables))
    if (tf.reduce_max(
            [tf.reduce_mean(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

print("intercept", linear_model.intercept.numpy())
print("weight", linear_model.weight.numpy())

## Factor Analysis

def generate_fa_data(n_sample, n_factor, n_item, 
                     ld, psi = None, rho = None, 
                     dtype = tf.float64):
    if (n_item % n_factor) != 0:
        n_item = n_factor * (n_item // n_factor)
    loading = np.zeros((n_item, n_factor))
    item_per_factor = (n_item // n_factor)
    for i in range(n_factor):
        for j in range(i * item_per_factor,
                       (i + 1) * item_per_factor):
            loading[j, i] = ld
    loading = tf.constant(loading, dtype = dtype)
    if rho is None:
        cor = tf.eye(n_factor, dtype = dtype)
    else:
        unit = tf.ones((n_factor, 1), dtype = dtype)
        identity = tf.eye(n_factor, dtype = dtype)
        cor = rho * (unit @ tf.transpose(unit)) + (1 - rho) * identity
    if psi is None:
        uniqueness = 1 - tf.linalg.diag_part(loading @ cor @ tf.transpose(loading))
    else:
        uniqueness = psi * tf.ones((n_item, ), dtype = dtype)
    
    mean = tf.zeros(n_item, dtype = dtype)
    cov = loading @ cor @ tf.transpose(loading) + tf.linalg.diag(uniqueness)
    dist_x = tfd.MultivariateNormalTriL(
        loc = mean, scale_tril = tf.linalg.cholesky(cov))
    x = dist_x.sample(n_sample)
    return x

n_sample = 10000
n_factor = 4
n_item = 12
ld = .7
dtype = tf.float64

x = generate_fa_data(n_sample, n_factor, 
                     n_item, ld,
                     dtype = dtype)
sample_mean = tf.reduce_mean(x, axis = 0)
sample_cov = tf.transpose(x - sample_mean) @ (x - sample_mean) / n_sample

sample_mean

# define a tf.Module to coollect parameters
class FactorModel(tf.Module):
  def __init__(self, n_item, n_factor, 
               dtype = tf.float64):
    super().__init__()
    self.intercept = tf.Variable(
        tf.zeros(n_item, dtype = dtype), name = "intercept")
    self.loading = tf.Variable(
        tf.random.uniform((n_item, n_factor), dtype = dtype), 
        name = "loading")
    self.uniqueness = tf.Variable(
        tf.fill(n_item, value = tf.constant(.2, dtype = dtype)), 
        name = "uniqueness")
  def __call__(self):
      model_mean = self.intercept
      model_cov = self.loading @ tf.transpose(self.loading) + tf.linalg.diag(self.uniqueness)
      return model_mean, model_cov

learning_rate = .5
epochs = 500
tol = 10**(-4)

factor_model = FactorModel(n_item, n_factor, dtype)
optimizer = tf.optimizers.SGD(learning_rate = learning_rate)

for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        model_mean, model_cov = factor_model()
        mvn = tfd.MultivariateNormalTriL(
            loc = model_mean, 
            scale_tril = tf.linalg.cholesky(model_cov))
        loss_value = - tf.reduce_mean(mvn.log_prob(x))
    gradients = tape.gradient(
        loss_value, factor_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, factor_model.trainable_variables))
    if (tf.reduce_max(
            [tf.reduce_mean(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

print("intercept", factor_model.intercept.numpy())
print("loading", factor_model.loading.numpy())
print("uniqueness", factor_model.uniqueness.numpy())

## Two-Parameter Logistic Model

def generate_2pl_data(n_sample, n_factor, n_item, 
                      alpha, beta, rho, 
                      dtype = tf.float64):
    if (n_item % n_factor) != 0:
        n_item = n_factor * (n_item // n_factor)
    item_per_factor = (n_item // n_factor)
    intercept = tf.fill((n_item,), value = tf.constant(alpha, dtype = dtype))
    loading = np.zeros((n_item, n_factor))
    for i in range(n_factor):
        for j in range(i * item_per_factor,
                       (i + 1) * item_per_factor):
            loading[j, i] = ld
    loading = tf.constant(loading, dtype = dtype)
    if rho is None:
        cor = tf.eye(n_factor, dtype = dtype)
    else:
        unit = tf.ones((n_factor, 1), dtype = dtype)
        identity = tf.eye(n_factor, dtype = dtype)
        cor = rho * (unit @ tf.transpose(unit)) + (1 - rho) * identity
    dist_eta = tfd.MultivariateNormalTriL(
        loc = tf.zeros(n_factor, dtype = dtype), scale_tril = tf.linalg.cholesky(cor))
    eta = dist_eta.sample(n_sample)
    logits = intercept + eta @ tf.transpose(loading)
    x = tfd.Bernoulli(logits=logits, dtype=dtype).sample()
    return x

n_sample = 10000
n_factor = 5
n_item = 25
alpha = .2
beta = .7 
rho = 0
dtype = tf.float64
x = generate_2pl_data(n_sample, n_factor, n_item, 
                      alpha, beta, rho, 
                      dtype = dtype)

class TwoPLModel(tf.Module):
    def __init__(self, n_item, n_factor, 
                 dtype = tf.float64):
        super().__init__()
        self.dtype = dtype
        self.intercept = tf.Variable(
            tf.zeros(n_item, dtype = self.dtype), name = "intercept")
        self.loading = tf.Variable(
            tf.random.uniform((n_item, n_factor), dtype = self.dtype), 
            name = "loading")
    def __call__(self, x):
        n_sample = len(x)
        joint_prob = tfd.JointDistributionSequential([
            tfd.Independent(
                tfd.Normal(
                    loc = tf.zeros((n_sample, n_factor), dtype=self.dtype),
                    scale = 1.0), 
                reinterpreted_batch_ndims=1),
            lambda eta: tfd.Independent(
                tfd.Bernoulli(
                    logits= self.intercept + eta @ tf.transpose(self.loading), 
                    dtype=self.dtype), 
                reinterpreted_batch_ndims=1)])             
        joint_prob._to_track=self
        return joint_prob

two_pl_model = TwoPLModel(n_item, n_factor)
joint_prob = two_pl_model(x)

def target_log_prob_fn(*eta):
    return joint_prob.log_prob(eta + (x,))

hmc=tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn = target_log_prob_fn,
    step_size = .015,
    num_leapfrog_steps=3)
current_state = joint_prob.sample()[:-1]
kernel_results = hmc.bootstrap_results(current_state)

@tf.function(autograph=False,
             experimental_compile=True)
def one_e_step(current_state, kernel_results):
    next_state, next_kernel_results = hmc.one_step(
        current_state=current_state,
        previous_kernel_results=kernel_results)
    return next_state, next_kernel_results

optimizer=tf.optimizers.RMSprop(learning_rate=.01)

@tf.function(autograph=False, 
             experimental_compile=True)
def one_m_step(current_state):
    with tf.GradientTape() as tape:
        loss_value = -tf.reduce_mean(
            target_log_prob_fn(*current_state))
    gradients = tape.gradient(loss_value, two_pl_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, two_pl_model.trainable_variables))
    return loss_value

import time
num_warmup_start = 1
num_warmup_iter = 1
num_iters = 1
num_accepted = 0
loss_history = np.zeros([num_iters])
tStart = time.time()
# Run warm-up stage.
for t in range(num_warmup_start):
    current_state, kernel_results = one_e_step(
        current_state, kernel_results)
    num_accepted += kernel_results.is_accepted.numpy().prod()
    if t % 500 == 0:
        print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(
            t, num_accepted / (t + 1)))
num_accepted = 0  # reset acceptance rate counter

# Run training.
for t in range(num_iters):
    for _ in range(num_warmup_iter):
        current_state, kernel_results = one_e_step(current_state, kernel_results)
    loss_value = one_m_step(current_state)
    num_accepted += kernel_results.is_accepted.numpy().prod()
    loss_history[t] = loss_value.numpy()
    if t % 50 == 0:
        print("Iteration: {:>4} Acceptance Rate: {:.3f} Loss: {:.3f}".format(
            t, num_accepted / (t + 1), loss_history[t]))
tEnd = time.time()

print(tEnd - tStart)
print(np.around(two_pl_model.trainable_variables[0].numpy(), decimals=2))
print(np.around(two_pl_model.trainable_variables[1].numpy(), decimals=2))

