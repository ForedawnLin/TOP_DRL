#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, random
import matplotlib.pyplot as plt
import os

np.random.seed(10703)
tf.set_random_seed(10703)
random.seed(10703)

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, networkname, trianable):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		if environment_name == 'CartPole-v0':
			self.nObservation = 4
			self.nAction = 2
			self.learning_rate = 0.0001
			self.architecture = [16, 32, 8]

		if environment_name == 'MountainCar-v0':
			self.nObservation = 2
			self.nAction = 3
			self.learning_rate = 0.0002
			self.architecture = [16, 32, 16]

		kernel_init = tf.random_uniform_initializer(-0.5, 0.5)
		bias_init = tf.constant_initializer(0)
		self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')
		with tf.variable_scope(networkname):
			layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer1', trainable=trianable)
			layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer2', trainable=trianable)
			layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer3', trainable=trianable)
			self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init, bias_initializer=bias_init, name='output', trainable=trianable)

		self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')

		if trianable == True:
			self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)

			self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		with tf.variable_scope(networkname, reuse=True):
			self.w1 = tf.get_variable('layer1/kernel')
			self.b1 = tf.get_variable('layer1/bias')
			self.w2 = tf.get_variable('layer2/kernel')
			self.b2 = tf.get_variable('layer2/bias')
			self.w3 = tf.get_variable('layer3/kernel')
			self.b3 = tf.get_variable('layer3/bias')
			self.w4 = tf.get_variable('output/kernel')
			self.b4 = tf.get_variable('output/bias')



	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.

		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		pass

	def load_model_weights(self,weight_file):
		# Helper function to load model weights.
		pass


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory = []
		self.is_burn_in = False
		self.memory_max = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		index = np.random.randint(len(self.memory), size=batch_size)
		batch = [self.memory[i] for i in index]
		return batch

	def append(self, transition):
		# Appends transition to the memory.
		self.memory.append(transition)
		if len(self.memory) > self.memory_max:
			self.memory.pop(0)



class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, sess, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.epsilon = 0.05

		if environment_name == 'CartPole-v0':
			self.gamma = 0.99
		if environment_name == 'MountainCar-v0':
			self.gamma = 1.0
		self.max_episodes = 10001
		self.batch_size = 32
		self.render = render

		self.qNetwork = QNetwork(environment_name, 'q', trianable=True)
		self.tNetwork = QNetwork(environment_name, 't', trianable=False)
		self.replay = Replay_Memory()

		self.env = gym.make(environment_name)
		self.env.seed(1)

		self.init = tf.global_variables_initializer()

		self.sess = sess
		tf.summary.FileWriter("logs/", self.sess.graph)
		self.sess.run(self.init)
		self.saver = tf.train.Saver(max_to_keep=200)

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		rnd = np.random.rand()
		if rnd <= self.epsilon:
			return np.random.randint(len(q_values))
		else:
			return np.argmax(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values)

	def network_assign(self):
		self.sess.run(tf.assign(self.tNetwork.w1, self.qNetwork.w1))
		self.sess.run(tf.assign(self.tNetwork.b1, self.qNetwork.b1))
		self.sess.run(tf.assign(self.tNetwork.w2, self.qNetwork.w2))
		self.sess.run(tf.assign(self.tNetwork.b2, self.qNetwork.b2))
		self.sess.run(tf.assign(self.tNetwork.w3, self.qNetwork.w3))
		self.sess.run(tf.assign(self.tNetwork.b3, self.qNetwork.b3))
		self.sess.run(tf.assign(self.tNetwork.w4, self.qNetwork.w4))
		self.sess.run(tf.assign(self.tNetwork.b4, self.qNetwork.b4))

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		reward_log = []
		test_reward_log = []
		test_episode = []
		if not self.replay.is_burn_in:
			self.burn_in_memory()
		for episode in np.arange(self.max_episodes):
			state = self.env.reset()
			if self.render:
				self.env.render()
			is_terminal = False
			rewardi = 0.0
			if episode % 10 == 0:
				self.network_assign()
			while not is_terminal:
				observation = np.array(state).reshape(1, -1)
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
				action = self.epsilon_greedy_policy(q_values)
				nextstate, reward, is_terminal, debug_info = self.env.step(action)
				if self.render:
					self.env.render()
				self.replay.append([state, action, reward, nextstate, is_terminal])
				state = nextstate
				rewardi = rewardi+reward

				batch = self.replay.sample_batch(self.batch_size)
				batch_observation = np.array([trans[0] for trans in batch])
				batch_action = np.array([trans[1] for trans in batch])
				batch_reward = np.array([trans[2] for trans in batch])
				batch_observation_next = np.array([trans[3] for trans in batch])
				batch_is_terminal = np.array([trans[4] for trans in batch])
				q_batch = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation})
				q_batch_next = self.sess.run(self.tNetwork.output, feed_dict={self.tNetwork.input: batch_observation_next})
				q_batch_nextq = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation_next})
				batch_actionq = np.argmax(q_batch_nextq,axis=1)
		
				y_batch = batch_reward+self.gamma*(1-batch_is_terminal)*q_batch_next[np.arange(self.batch_size),batch_actionq]

				targetQ = q_batch.copy()
				targetQ[np.arange(self.batch_size), batch_action] = y_batch
				_, train_error = self.sess.run([self.qNetwork.opt, self.qNetwork.loss], feed_dict={self.qNetwork.input: batch_observation, self.qNetwork.targetQ: targetQ})
			reward_log.append(rewardi)
			print(episode, rewardi)
			if episode % 100 == 0:
				test_reward = self.test()
				test_reward_log.append(test_reward/20.0)
				test_episode.append(episode)
				save_path = self.saver.save(self.sess, "../model/model_{}.ckpt".format(episode))
				print("Model saved in path: %s" % save_path)
			train_episode = np.arange(episode+1)
			np.savez('../data/training_log.npz', test_episode=test_episode, test_reward_log=test_reward_log,
				 reward_log=reward_log, train_episode=train_episode)
		print(test_reward_log)
		train_episode = np.arange(self.max_episodes)
		np.savez('../data/training_log.npz', test_episode=test_episode, test_reward_log=test_reward_log,
				 reward_log=reward_log, train_episode=train_episode)

	def test(self, model_file=None, no=20, stat=False):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# uncomment this line below for videos
		# self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)
		if model_file is not None:
			self.saver.restore(self.sess, model_file)
		reward_list = []
		cum_reward = 0.0
		for episode in np.arange(20):
			episode_reward = 0.0
			state = self.env.reset()
			if self.render:
				self.env.render()
			is_terminal = False
			while not is_terminal:
				observation = np.array(state).reshape(1, -1)
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
				action = self.greedy_policy(q_values)
				nextstate, reward, is_terminal, debug_info = self.env.step(action)
				if self.render:
					self.env.render()
				state = nextstate
				episode_reward = episode_reward+reward
				cum_reward = cum_reward+reward
			reward_list.append(episode_reward)
		if stat:
			return cum_reward, reward_list
		else:
			return cum_reward


	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		state = self.env.reset()
		for i in np.arange(self.replay.burn_in):
			action = self.env.action_space.sample()
			nextstate, reward, is_terminal, debug_info = self.env.step(action)
			self.replay.append([state, action, reward, nextstate, is_terminal])
			if is_terminal:
				state = self.env.reset()
			else:
				state = nextstate
		self.replay.is_burn_in = True

	def test_final_model(self, model_no):
		# test the performance of the final model with 100 episodes.
		_, t_reward = self.test(model_file="../model/model_{}.ckpt".format(model_no), no=100, stat = True)
		t_reward = np.array(t_reward)
		print('final model performance(mean,std):')
		print(np.mean(t_reward),np.std(t_reward))
		

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--test',dest='test',type=int,default=0)
	parser.add_argument('--test_final',dest='test_final',type=int,default=0)
	parser.add_argument('--model_no',dest='model_file_no',type=str)
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	model_path = '../model/'
	data_path = '../data/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	agent = DQN_Agent(environment_name, sess, render=args.render)
	if args.train == 1:
		agent.train()
	if args.test == 1:
		print(agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no))/20.0)
	if args.test_final == 1:
		agent.test_final_model(args.model_file_no)
	sess.close()

if __name__ == '__main__':
	main(sys.argv)

