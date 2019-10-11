import os
import gym
import numpy as np
import tensorflow as tf

from keras.initializers import random_uniform
# from utils import plotLearning


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, actions_number):
        self.memory_size = max_size
        self.memory_center = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, actions_number))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_center % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.memory_center += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_center, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class Actor(object):
    def __init__(self, learning_rate, action_numbers, name, input_dimensions, session, function1_dimensions,
                 function2_dimensions, action_bound, batch_size=64, checkpoint_dir='tmp/ddpg'):
        self.learning_rate = learning_rate
        self.action_numbers = action_numbers
        self.input_dimensions = input_dimensions
        self.name = name
        self.function1_dimensions = function1_dimensions
        self.function2_dimensions = function2_dimensions
        self.session = session
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.checkpoint_dir = checkpoint_dir
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoint_dir, name+'_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input= tf.placeholder(tf.float32, shape=[None, *self.input_dimensions], name='inputs')
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.action_numbers])
            f1 = 1 / np.sqrt(self.function1_dimensions)
            dense1 = tf.layers.dense(self.input, units=self.function1_dimensions,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.function2_dimensions)
            dense2 = tf.layers.dense(layer1_activation, units=self.function2_dimensions,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)

            f3 = 0.003
            mu = tf.layers.dense(layer1_activation, units=self.action_numbers, activation='tanh',
                                 kernel_initializer=random_uniform(-f3, f3),
                                 bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.session.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.session.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.saver.save(self.session, self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.saver.restore(self.session, self.checkpoint_file)


class Critic(object):
    def __init__(self, learning_rate, action_numbers, name, input_dimensions, session, function1_dimensions,
                 function2_dimensions, batch_size=64, checkpoint_dir='tmp/ddpg'):
        self.learning_rate = learning_rate
        self.action_numbers = action_numbers
        self.input_dimensions = input_dimensions
        self.name = name
        self.function1_dimensions = function1_dimensions
        self.function2_dimensions = function2_dimensions
        self.session = session
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):

            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dimensions], name='inputs')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.action_numbers], name='actions')
            self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='targets')

            f1 = 1 / np.sqrt(self.function1_dimensions)
            dense1 = tf.layers.dense(self.input, units=self.function1_dimensions,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.function2_dimensions)
            dense2 = tf.layers.dense(layer1_activation, units=self.function2_dimensions,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)

            action_in = tf.layers.dense(self.actions, units=self.function2_dimensions, activation='relu')

            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1, kernal_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.session.run(self.q, feed_dict={self.input: inputs, self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.session.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions,
                                                          self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.session.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.saver.save(self.session, self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.saver.restore(self.session, self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, beta, input_dimensions, tau, environment, gamma=0.99, actions_number=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dimensions, actions_number)
        self.batch = batch_size
        self.session = tf.Session()
        self.actor = Actor(alpha, actions_number, 'Actor', input_dimensions, self.session,
                           layer1_size, layer2_size, environment.action_space.high)
        self.critic = Critic(beta, actions_number, 'Critic', input_dimensions, self.session,
                             layer1_size, layer2_size)
        self.target_actor = Actor(alpha, actions_number, 'TargetActor', input_dimensions, self.session,
                                  layer1_size, layer2_size, environment.action_space.high)
        self.target_critic = Critic(beta, actions_number, 'TargetCritic', input_dimensions, self.session,
                                    layer1_size, layer2_size)

        self.noise = OUActionNoise(mu=np.zeros(actions_number))

        self.update_critic = [self.target_critic.params[i].
                              assign(tf. multiply(self.critic.params[i], self.tau)
                              + tf.multiply(self.target_critic.params[i], 1. - self.tau))
                              for i in range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].
                             assign(tf.multiply(self.actor.params[i], self.tau)
                             + tf.multiply(self.target_actor.params[i], 1. - self.tau))
                             for i in range(len(self.target_actor.params))]
        self.session.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.session.run(self.update_critic)
            self.target_actor.session.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.session.run(self.update_critic)
            self.target_actor.session.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.memory_center < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        critic_value = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value[j] * done[j])

        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        action_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, action_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


if __name__ == '__main__':
    environment = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.001, input_dimensions=[3], tau=0.001, environment=environment, batch_size=64,
                  layer1_size=400, layer2_size=300, actions_number=1)

    score_history = []
    np.random.seed(0)
    for i in range(1000):
        obs = environment.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = environment.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print('episode', i, 'score %.2f' % score,
              "100 game average %.2f" % np.mean(score_history[-100:]))

    filename = 'pendulum.png'
    # plotLearning(score_history, filename, window=100)

