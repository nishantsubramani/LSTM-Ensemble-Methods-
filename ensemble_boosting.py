
"""Based on the TensorFlow RNN Tutorial

Configs: Many Different Configs (can read about in get_config and their class definitions)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import shutil
import gnumpy as gpu
import numpy as np
import tensorflow as tf
import skflow as sk
import pandas as pd
from sklearn.preprocessing import normalize
import os
import operator
import csv

import updated_reader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
		"model", "small",
		"A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class PTBModel(object):
	"""The PTB model."""

	def __init__(self, is_training, config):
		#self._case_weight = case_weight
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size
		self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell, output_keep_prob=config.keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

		self._initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device("/gpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size])
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)

			#inputs = Tensor(Batch_size, num_steps, hidden_size)

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.input_keep_prob)

		# Simplified version of tensorflow.models.rnn.rnn.py's rnn().
		# This builds an unrolled LSTM for tutorial purposes only.
		# In general, use the rnn() or state_saving_rnn() from rnn.py.
		#
		# The alternative version of the code below is:
		#
		# from tensorflow.models.rnn import rnn
		# inputs = [tf.squeeze(input_, [1])
		#           for input_ in tf.split(1, num_steps, inputs)]
		# outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.concat(1, outputs), [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
		softmax_b = tf.get_variable("softmax_b", [vocab_size])
		self._logits = logits = tf.matmul(output, softmax_w) + softmax_b
		self._probabilities = probabilities = tf.nn.softmax(logits)
		
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(self._targets, [-1])],
				[tf.ones([batch_size * num_steps])])
		#sess = tf.InteractiveSession()
		#loss.eval()
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		#cost.eval()
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
																			config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars))

	def assign_lr(self, session, lr_value):
		session.run(tf.assign(self.lr, lr_value))
	
	@property
	def case_weight(self):
		return self._case_weight
	
	@property
	def input_data(self):
		return self._input_data

	@property
	def targets(self):
		return self._targets

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def logits(self):
		return self._logits
	
	@property
	def probabilities(self):
		return self._probabilities
	


class SmallConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class SmallExtraConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 20
	keep_prob = 1.0
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class SmallDropConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 0.5
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class SmallExtraDropConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 20
	keep_prob = 0.5
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class MediumConfig(object):
	"""Medium config."""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	input_keep_prob = 1.0
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000

class LargeConfig(object):
	"""Large config."""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	input_keep_prob = 1.0
	lr_decay = 1 / 1.15
	batch_size = 20
	vocab_size = 10000

class TestConfig(object):
	"""Tiny config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class NishantConfig1(object):
	"""Nishant config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 4
	num_layers = 2
	num_steps = 20
	hidden_size = 150
	max_epoch = 3
	max_max_epoch = 10
	keep_prob = 1.0
	input_keep_prob = 1.0
	lr_decay = 0.4
	batch_size = 20
	vocab_size = 10000

class NishantConfig2(object):
	"""Nishant config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 4
	num_layers = 2
	num_steps = 20
	hidden_size = 150
	max_epoch = 3
	max_max_epoch = 10
	keep_prob = 0.5
	input_keep_prob = 1.0
	lr_decay = 0.4
	batch_size = 20
	vocab_size = 10000

class DougSmall(object):
	'''Doug config, for testing'''
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 3
	num_layers = 2
	num_steps = 10
	hidden_size = 50
	max_epoch = 3
	max_max_epoch = 10
	keep_prob = 1.0
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class DougDropout(object):
	'''Doug dropout config, for testing'''
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 3
	num_layers = 2
	num_steps = 10
	hidden_size = 50
	max_epoch = 3
	max_max_epoch = 10
	keep_prob = 0.5
	input_keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

def run_epoch(session, m, data, eval_op, verbose=False):
	"""Runs the model on the given data."""
	epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = m.initial_state.eval()
	#i = 0

	#funny_dropout_probs = [0.1, 0.01, 0.0005]
	for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
																										m.num_steps)):
		#print(step)
		#print(x)
		#x = tf.nn.dropout([x], tf.Variable(tf.int32(0.5)))
		#print(x)
		#print(y)

		#eval_op.input_keep_prob = funny_dropout_probs[i%3]
		cost, state, _ = session.run([m.cost, m.final_state, eval_op],
																 {m.input_data: x,
																	m.targets: y,
																	m.initial_state: state})
		#print(cost)
		costs += cost
		iters += m.num_steps
		#i+=1
		#print(step)
		#print(epoch_size)
		if verbose and step % (epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
						(step * 1.0 / epoch_size, gpu.exp(costs / iters),
						 iters * m.batch_size / (time.time() - start_time)))

	return gpu.exp(costs / iters)

def run_epoch_boost(session, m, data, eval_op, weights, epoch_num, verbose=False, forward = True):
	"""Runs the model on the given data."""
	epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = m.initial_state.eval()
	#i = 0

	#funny_dropout_probs = [0.1, 0.01, 0.0005]
	for step, (x, y) in enumerate(reader.ptb_iterator_boost(data, m.batch_size,
																										m.num_steps, weights, epoch_num, forward)):
		#print(step)
		#print(x)
		#x = tf.nn.dropout([x], tf.Variable(tf.int32(0.5)))
		#print(x)
		#print(y)

		#eval_op.input_keep_prob = funny_dropout_probs[i%3]
		cost, state, _ = session.run([m.cost, m.final_state, eval_op],
																 {m.input_data: x,
																	m.targets: y,
																	m.initial_state: state})
		#print(cost)
		costs += cost
		iters += m.num_steps
		#i+=1
		#print(step)
		#print(epoch_size)
		if verbose and step % (epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
						(step * 1.0 / epoch_size, gpu.exp(costs / iters),
						 iters * m.batch_size / (time.time() - start_time)))

	return gpu.exp(costs / iters)


def get_config():
	if FLAGS.model == "small":
		return SmallConfig()
	elif FLAGS.model == 'smalldrop':
		return SmallDropConfig()
	elif FLAGS.model == 'smallextra':
		return SmallExtraConfig()
	elif FLAGS.model == 'smallextradrop':
		return SmallExtraDropConfig()
	elif FLAGS.model == "medium":
		return MediumConfig()
	elif FLAGS.model == "large":
		return LargeConfig()
	elif FLAGS.model == "test":
		return TestConfig()
	elif FLAGS.model == "nishant1":
		return NishantConfig1()
	elif FLAGS.model == "nishantdrop":
		return NishantConfig2()
	elif FLAGS.model == 'doug':
		return DougSmall()
	elif FLAGS.model == 'dougdrop':
		return DougDropout()
	else:
		raise ValueError("Invalid model: %s", FLAGS.model)

def output_training_set_error_for_boosting(session, m, data, eval_op, verbose=False):
	#Make sure m = PTBModel(is_training=False, config=eval_config, case_weight = None)

	"""Runs the model on the given data."""
	#epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
	#start_time = time.time()
	costs = []
	#iters = 0
	state = m.initial_state.eval()

	for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
																										m.num_steps)):
		cost, state, _ = session.run([m.cost, m.final_state, eval_op],
																 {m.input_data: x,
																	m.targets: y,
																	m.initial_state: state})
		costs.append(cost/m.num_steps)
		#iters += m.num_steps
		#print(gpu.shape(probs))

		# if verbose and step % (epoch_size // 10) == 10:
		#   print("%.3f perplexity: %.3f speed: %.0f wps" %
		#         (step * 1.0 / epoch_size, gpu.exp(costs / iters),
		#          iters * m.batch_size / (time.time() - start_time)))

	return(costs)

def output_test_set_probs(session, m, data, eval_op, verbose=False, partition = False):
	"""Runs the model on the given data."""
	#epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
	#start_time = time.time()
	#costs = []
	#iters = 0
	state = m.initial_state.eval()
	full_probs = []
	for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
																										m.num_steps)):
		cost, state, probs, _ = session.run([m.cost, m.final_state, m.probabilities, eval_op],
																 {m.input_data: x,
																	m.targets: y,
																	m.initial_state: state})
		#costs.append(cost/m.num_steps)
		#iters += m.num_steps
		#print(np.shape(probs))
		if partition:
				full_probs.append(probs[0,y[0,0]])
		else:
				full_probs.append(probs)

		# if verbose and step % (epoch_size // 10) == 10:
		#   print("%.3f perplexity: %.3f speed: %.0f wps" %
		#         (step * 1.0 / epoch_size, gpu.exp(costs / iters),
		#          iters * m.batch_size / (time.time() - start_time)))

	return(full_probs)

#Normalization function that is very slow. Use faster methods (numpy or otherwise) methods if you see this function called.
def naive_normalization(x):
	new_x = []

	for entry in x:
		new_x.append(float(entry)/sum(x))
	return(new_x)

def classify_ensemble(data, probs, batch_size, num_steps):
	epoch_size = ((len(data) // batch_size) - 1) // num_steps
	start_time = time.time()
	costs = 0.0
	iters = 0
	# for i in range(len(probs)):
	#   probs[i] = probs[i] / np.sum(probs[i])
	#probs = tf.nn.softmax(probs)
	# print(np.sum(probs[0]))
	# print(np.sum(probs[50]))
	#print(len(probs[0]))
	for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size,
																										num_steps)):
		# print(step)
		# print(x)
		# print(y)
		# print(probs)
		#print(probs[0])
		cost = -1 * gpu.log(probs[step][0,y[0,0]])
		#print(cost)
		'''
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(y, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float64)])

		print(loss)
		cost = tf.reduce_sum(loss) / batch_size
		print(cost)
		'''
		costs += cost
		iters += num_steps
		
		if step % (epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
						(step * 1.0 / epoch_size, gpu.exp(costs / iters),
						 iters * batch_size / (time.time() - start_time)))

	return gpu.exp(costs / iters)	

def baseline(data_path, num_ensembles, model_name, train = True, random_training_order = False, reverse_order = False):
		algo_name = 'baseline'
		raw_data = reader.ptb_raw_data(data_path)
		train_data, valid_data, test_data, _, word_to_id = raw_data

		if reverse_order:
				train_data.reverse()
				valid_data.reverse()
				test_data.reverse()
				algo_name = 'reverse ' + algo_name

		if random_training_order:
				train_sentence_list = reader.get_sentence_list(train_data, word_to_id['<eos>'], reverse_order)
				perm = range(len(train_sentence_list))
				np.random.shuffle(perm)
				train_data = []
				for idx in perm:
						train_data += train_sentence_list[idx]


		FLAGS.model = model_name

		config = get_config()
		eval_config = get_config()
		eval_config.batch_size = 1
		eval_config.num_steps = 1

		full_test_set_logits = []
		full_train_set_logits = []
		for i in range(len(test_data)):
				full_test_set_logits.append(np.zeros((1,eval_config.vocab_size)))
		for i in range(len(train_data)):
				full_train_set_logits.append(np.zeros((1,eval_config.vocab_size)))
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
		for iii in range(num_ensembles):
			with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as session:
						initializer = tf.random_uniform_initializer(-config.init_scale,
																										config.init_scale)
						with tf.variable_scope("model", reuse=None, initializer=initializer):
								sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
								m = PTBModel(is_training=True, config=config)
								sess.close()
						
						with tf.variable_scope("model", reuse=True, initializer=initializer):
								mvalid = PTBModel(is_training=False, config=config)
								mtest = PTBModel(is_training=False, config=eval_config)
								m2 = PTBModel(is_training=False, config=eval_config)

						tf.initialize_all_variables().run()
						saver = tf.train.Saver()

						if random_training_order:
								new_folder = 'random training order' + algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						else:
								new_folder = algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'
						if not os.path.exists(checkpoint_dir):
								os.makedirs(checkpoint_dir)
						

						if train:
								for i in range(config.max_max_epoch):
										lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
										m.assign_lr(session, config.learning_rate * lr_decay)

										print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
										train_perplexity = run_epoch(session, m, train_data, m.train_op, verbose=False)
										print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
										
										if (i+1)%5 == 0 or (i+1) == config.max_max_epoch:
												valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
												print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

										if (i+1) % 5 == 0 or (i+1) == config.max_max_epoch:
												saver.save(session, checkpoint_dir + 'model.ckpt', global_step = i+1)
						else:
								ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
								if ckpt and ckpt.model_checkpoint_path:
										saver.restore(session, ckpt.model_checkpoint_path)


			#case_scores = output_training_set_error_for_boosting(session, m2, train_data, tf.no_op())
				
			#print('Got past case_score computation')

			# score = sum(case_scores)
			# norm = len(case_scores) * -1 * gpu.log(float(1.0) / eval_config.vocab_size)
			# epsilon_t =  (1 - (norm - score) / norm) / 2.0
			# alpha_t = 0.5 * gpu.log((1 - epsilon_t)/ epsilon_t)
			# alpha_t_list.append(alpha_t)

			#print('simple math done')
			#print(epsilon_t)

			# train_case_weights = gpu.sqrt((1 - epsilon_t)/ epsilon_t) * np.multiply(train_case_weights, np.asarray(case_scores))
			# train_case_weights = naive_normalization(train_case_weights)

			#print('case_weights have been modified')
						test_set_probs = output_test_set_probs(session, mtest, test_data, tf.no_op())
						train_set_probs = output_test_set_probs(session, m2, train_data, tf.no_op())
						for i in range(len(test_set_probs)):
							#test_set_probs[i] = test_set_probs[i] * alpha_t
								full_test_set_logits[i] += test_set_probs[i]

						for i in range(len(train_set_probs)):
								full_train_set_logits[i] += train_set_probs[i]

						#print('test set computation for this classifier has been completed')
						# test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
						# print("Test Perplexity: %.3f" % test_perplexity)
		
		#alpha_t_sum = sum(alpha_t_list)
		for i in range(len(full_test_set_logits)):
				full_test_set_logits[i] = full_test_set_logits[i] / num_ensembles

		for i in range(len(full_train_set_logits)):
				full_train_set_logits[i] = full_train_set_logits[i] / num_ensembles


		test_ensemble_perplexity = classify_ensemble(test_data, full_test_set_logits, 1, 1)
		train_ensemble_perplexity = classify_ensemble(train_data, full_train_set_logits, 1, 1)
		print('test_ppl:' + str(test_ensemble_perplexity))
		print('train ppl: ' + str(train_ensemble_perplexity))

#AdaBoost Inspired Mini Batch Sampling
def ABIMBS(data_path, num_ensembles, model_name, forward = True, train = True, random_training_order = False):
		
		if forward:
				algo_name = 'FABIMBS'
		else:
				algo_name = 'BABIMBS'

		raw_data = reader.ptb_raw_data(data_path)
		train_data, valid_data, test_data, _, word_to_id = raw_data

		if random_training_order:
				train_sentence_list = reader.get_sentence_list(train_data, word_to_id['<eos>'])
				perm = range(len(train_sentence_list))
				np.random.shuffle(perm)
				train_data = []
				for idx in perm:
						train_data += train_sentence_list[idx]

		case_weight_length = len(train_data)-1
		train_case_weights = np.repeat(1.0/case_weight_length, case_weight_length).tolist()
		
		FLAGS.model = model_name
		config = get_config()
		eval_config = get_config()
		eval_config.batch_size = 1
		eval_config.num_steps = 1

		alpha_t_list = []
		full_test_set_logits = []
		for i in range(len(test_data)-1):
				full_test_set_logits.append(np.zeros((1,eval_config.vocab_size)))

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
		
		for iii in range(num_ensembles):
				with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as session:
						initializer = tf.random_uniform_initializer(-config.init_scale,
																							config.init_scale)
						with tf.variable_scope("model", reuse=None, initializer=initializer):
								sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
								m = PTBModel(is_training=True, config=config)
								sess.close()

						with tf.variable_scope("model", reuse=True, initializer=initializer):
								mvalid = PTBModel(is_training=False, config=config)
								mtest = PTBModel(is_training=False, config=eval_config)
								#config.batch_size = 1
								m2 = PTBModel(is_training=False, config=eval_config)

						tf.initialize_all_variables().run()
						saver = tf.train.Saver()
						
						if random_training_order:
								new_folder = 'random training order' + algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						else:
								new_folder = algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						
						checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'
						if not os.path.exists(checkpoint_dir):
								os.makedirs(checkpoint_dir)
						
						if train:
								data_len = len(train_data)

								if iii == 0:
										if forward:
												train_case_weights = train_case_weights[0:(data_len-config.num_steps)]
										else:
												train_case_weights = train_case_weights[(config.num_steps - 1):(data_len - 1)]

								train_case_weights = normalize(np.asarray(train_case_weights).reshape(1,-1), norm = 'l1')

								np.savetxt(checkpoint_dir + 'train_case_weights.out', train_case_weights, delimiter = ',')  
								
								for i in range(config.max_max_epoch):
										lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
										m.assign_lr(session, config.learning_rate * lr_decay)

										print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
										train_perplexity = run_epoch_boost(session, m, train_data, m.train_op, train_case_weights, i, verbose = False, forward = forward)
										print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
										valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
										print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

										if (i+1) % 5 == 0 or (i+1) == config.max_max_epoch:
												saver.save(session, checkpoint_dir + 'model.ckpt', global_step = i+1)
						else:
								ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
								if ckpt and ckpt.model_checkpoint_path:
										saver.restore(session, ckpt.model_checkpoint_path)

						case_scores = output_training_set_error_for_boosting(session, m2, train_data, tf.no_op())

						score = sum(case_scores)
						norm = len(case_scores) * -1 * gpu.log(float(1.0) / eval_config.vocab_size)
						epsilon_t =  (1 - (norm - score) / norm) / 2.0
						alpha_t = 0.5 * gpu.log((1 - epsilon_t)/ epsilon_t)

						if random_training_order:
								alpha_filepath = 'simple-examples/ckpt/' + 'random training order' + algo_name + '_' + model_name + '/' + 'alpha_t.out'
						else:
								alpha_filepath = 'simple-examples/ckpt/' + algo_name + '_' + model_name + '/' + 'alpha_t.out'
						
						if iii == 0:
								shutil.rmtree(alpha_filepath, ignore_errors = True)
								
						with open(alpha_filepath, 'ab') as f:
								f.write(str(alpha_t))
								f.write(',')
						alpha_t_list.append(alpha_t)

						if forward:
								case_scores = case_scores[0:(data_len-config.num_steps)]
						else:
								case_scores = case_scores[(config.num_steps - 1):(data_len - 1)]

						train_case_weights = gpu.sqrt((1 - epsilon_t)/ epsilon_t) * np.multiply(train_case_weights, np.asarray(case_scores))

						test_set_probs = output_test_set_probs(session, mtest, test_data, tf.no_op())
						for i in range(len(test_set_probs)):
								test_set_probs[i] = test_set_probs[i] * alpha_t
								full_test_set_logits[i] += test_set_probs[i]

						test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
						print("Test Perplexity: %.3f" % test_perplexity)
		
		alpha_t_sum = sum(alpha_t_list)
		for i in range(len(full_test_set_logits)):
			full_test_set_logits[i] = full_test_set_logits[i] / alpha_t_sum

		ensemble_perplexity = classify_ensemble(test_data, full_test_set_logits, 1, 1)
		print(ensemble_perplexity)

#AdaBoost Inspired Sentence Sampling
def ABISS(data_path, num_ensembles, model_name, method = 'stddev', train = True, random_training_order = False):
		
		algo_name = method + ' ' + 'ABISS'

		raw_data = reader.ptb_raw_data(data_path)
		train_data, valid_data, test_data, _, word_to_id = raw_data

		eos_id = word_to_id['<eos>']
		case_weight_length = len(train_data)-1
		train_case_weights = np.repeat(1.0/case_weight_length, case_weight_length).tolist()

		train_sentence_list = reader.get_sentence_list(train_data, eos_id)
		if random_training_order:
				perm = range(len(train_sentence_list))
				np.random.shuffle(perm)
				train_data = []
				for idx in perm:
						train_data += train_sentence_list[idx]
				train_sentence_list = reader.get_sentence_list(train_data, eos_id)
		num_sent = len(train_sentence_list)
		train_sentence_weights = np.repeat(1.0/num_sent, num_sent).tolist()

		new_train_data = reader.weighted_sentence_selection(train_sentence_list, train_sentence_weights, random_training_order)
	
		FLAGS.model = model_name

		config = get_config()
		eval_config = get_config()
		eval_config.batch_size = 1
		eval_config.num_steps = 1

		alpha_t_list = []
		full_test_set_logits = []
		for i in range(len(test_data)-1):
				full_test_set_logits.append(np.zeros((1,eval_config.vocab_size)))
		
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)

		for iii in range(num_ensembles):
				with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as session:
						initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

						with tf.variable_scope("model", reuse=None, initializer=initializer):
								sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
								m = PTBModel(is_training=True, config=config)
								sess.close()


						with tf.variable_scope("model", reuse=True, initializer=initializer):
								mvalid = PTBModel(is_training=False, config=config)
								mtest = PTBModel(is_training=False, config=eval_config)
								#config.batch_size = 1
								m2 = PTBModel(is_training=False, config=eval_config)

						tf.initialize_all_variables().run()
						saver = tf.train.Saver()
						
						if random_training_order:
								new_folder = 'random training order' + algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						else:
								new_folder = algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						
						checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'
						if not os.path.exists(checkpoint_dir):
								os.makedirs(checkpoint_dir)

						if train:
								np.savetxt(checkpoint_dir + 'train_case_weights.out', train_case_weights, delimiter = ',')
								np.savetxt(checkpoint_dir + 'train_sentence_weights.out', train_sentence_weights, delimiter = ',')

								for i in range(config.max_max_epoch):
										lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
										m.assign_lr(session, config.learning_rate * lr_decay)

										print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
										train_perplexity = run_epoch(session, m, new_train_data, m.train_op, verbose=False)
										print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
										valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
										print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
										
										if (i+1) % 5 == 0 or (i+1) == config.max_max_epoch:
												saver.save(session, checkpoint_dir + 'model.ckpt', global_step = i+1)
						else:
								ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
								if ckpt and ckpt.model_checkpoint_path:
										saver.restore(session, ckpt.model_checkpoint_path)

						case_scores = output_training_set_error_for_boosting(session, m2, train_data, tf.no_op())

						score = sum(case_scores)
						norm = len(case_scores) * -1 * gpu.log(float(1.0) / eval_config.vocab_size)
						epsilon_t =  (1 - (norm - score) / norm) / 2.0
						alpha_t = 0.5 * gpu.log((1 - epsilon_t)/ epsilon_t)
						alpha_t_list.append(alpha_t)

						if iii == 0:
								shutil.rmtree('simple-examples/ckpt/' + algo_name + '_' + model_name + '/' + 'alpha_t.out', ignore_errors = True)

						with open('simple-examples/ckpt/' + algo_name + '_' + model_name + '/' + 'alpha_t.out', 'ab') as f:
								f.write(str(alpha_t))
								f.write(',')
						
						train_case_weights = gpu.sqrt((1 - epsilon_t)/ epsilon_t) * np.multiply(train_case_weights, np.asarray(case_scores))
						train_case_weights = np.ravel(normalize(np.asarray(train_case_weights).reshape(1,-1), norm = 'l1'))

						if method == 'stddev':
								new_train_case_weights = reject_outliers(train_case_weights)
						elif method == 'sqrt':
								new_train_case_weights = sqrt_norm(train_case_weights)
			
						
						start_idx = 0

						for i in range(len(train_sentence_list)):
								this_sentence_length = len(train_sentence_list[i])
								sentence_tokens = [v for v in new_train_case_weights[start_idx:(this_sentence_length+start_idx)] if np.isfinite(v)]
								if len(sentence_tokens) == 0:
										this_sentence_weights[i] = 0
								else:
										train_sentence_weights[i] = np.mean([v for v in new_train_case_weights[start_idx:(this_sentence_length+start_idx)] if np.isfinite(v)])


								start_idx += this_sentence_length

						train_sentence_weights = np.ravel(normalize(np.asarray(train_sentence_weights).reshape(1,-1), norm = 'l1'))

						new_train_data = reader.weighted_sentence_selection(train_sentence_list, train_sentence_weights, random_training_order)

						test_set_probs = output_test_set_probs(session, mtest, test_data, tf.no_op())

						for i in range(len(test_set_probs)):
								test_set_probs[i] = test_set_probs[i] * alpha_t
								full_test_set_logits[i] += test_set_probs[i]
						
						test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
						print("Test Perplexity: %.3f" % test_perplexity)
		
		alpha_t_sum = np.sum(alpha_t_list)
		for i in range(len(full_test_set_logits)):
				full_test_set_logits[i] = full_test_set_logits[i] / alpha_t_sum

		ensemble_perplexity = classify_ensemble(test_data, full_test_set_logits, 1, 1)
		print(ensemble_perplexity)

'''
Function that assumes that sentences that start with the same word behave similarly. Models are trained on smaller data in which sentence weights for sentences starting with a specific token are averaged for a mean unigram weight.
The highest unigram weights are chosen and all sentences that start with these tokens are chosen in the next dataset for this ensemble method. Method doesn't provide positive results.
'''
def unigram_partition(data_path, num_ensembles, model_name, method = 'none', train = True, random_training_order = False, reverse_order = False):
		
		algo_name = 'unigram' + '_' + 'partition'

		raw_data = reader.ptb_raw_data(data_path)
		train_data, valid_data, test_data, _, word_to_id = raw_data

		if reverse_order:
				train_data.reverse()
				valid_data.reverse()
				test_data.reverse()
				algo_name = 'reverse ' + algo_name

		eos_id = word_to_id['<eos>']
		case_weight_length = len(train_data)-1
		train_case_weights = np.repeat(1.0/case_weight_length, case_weight_length).tolist()

		train_sentence_list = reader.get_sentence_list(train_data, eos_id, reverse_order)
		if random_training_order:
				#train_sentence_list = reader.get_sentence_list(train_data, eos_id)
				perm = range(len(train_sentence_list))
				np.random.shuffle(perm)
				train_data = []
				for idx in perm:
						train_data += train_sentence_list[idx]
				train_sentence_list = reader.get_sentence_list(train_data, eos_id, reverse_order)
		num_sent = len(train_sentence_list)
		train_sentence_weights = np.repeat(1.0/num_sent, num_sent).tolist()

		new_train_data = train_data
	
		FLAGS.model = model_name

		config = get_config()
		eval_config = get_config()
		eval_config.batch_size = 1
		eval_config.num_steps = 1

		alpha_t_list = []
		full_test_set_logits = []
		for i in range(len(test_data)-1):
				full_test_set_logits.append(np.zeros((1,eval_config.vocab_size)))
		
		sentence_starters = []
		id_to_sentence_num_dict = {}
		for i in range(num_sent):
				if reverse_order:
						desired_id = train_sentence_list[i][-1]
				else:
						desired_id = train_sentence_list[i][0]
				if desired_id in id_to_sentence_num_dict:
						id_to_sentence_num_dict[desired_id].append(i)
				else:
						id_to_sentence_num_dict[desired_id] = [i]
						sentence_starters.append(desired_id)


		id_to_model = {}
		for idx in sentence_starters:
				id_to_model[idx] = [1]

		id_to_weight = {}


		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)

		for iii in range(num_ensembles):
				with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as session:
						initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

						with tf.variable_scope("model", reuse=None, initializer=initializer):
								sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
								m = PTBModel(is_training=True, config=config)
								sess.close()


						with tf.variable_scope("model", reuse=True, initializer=initializer):
								mvalid = PTBModel(is_training=False, config=config)
								mtest = PTBModel(is_training=False, config=eval_config)
								
								m2 = PTBModel(is_training=False, config=eval_config)

						tf.initialize_all_variables().run()
						saver = tf.train.Saver()
						if iii > 0:
								np.savetxt(checkpoint_dir + 'test_set_probs_no_alpha.out', np.squeeze(test_set_probs_no_alpha), delimiter = ',')

						if random_training_order:
								new_folder = 'random training order ' + algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						else:
								new_folder = algo_name + '_' + model_name + '/' + 'ensemble' + str(iii + 1)
						
						checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'
						if not os.path.exists(checkpoint_dir):
								os.makedirs(checkpoint_dir)

						if iii > 0:
								for k,v in id_to_sentence_num_dict.items():
										total_wt = 0
										num_sentences = len(v)
										for sent_idx in v:
												total_wt += train_sentence_weights[sent_idx]
										id_to_weight[k] = total_wt/num_sentences

								sorted_id_to_weight = sorted(id_to_weight.items(), key = operator.itemgetter(1), reverse = True)
								np.savetxt(checkpoint_dir + 'sorted_id_to_weight', sorted_id_to_weight, delimiter = ',')
								new_train_data = []
								i = 0
								sent_included = 0                      
								while sent_included < np.floor(num_sent/2):
										start_key = sorted_id_to_weight[i][0]
										sentence_additions = id_to_sentence_num_dict[start_key]
										for idx in sentence_additions:
												new_train_data += train_sentence_list[idx]
												sent_included += 1
										id_to_model[start_key].append(iii + 1)
										i += 1

						

						if train:

								np.savetxt(checkpoint_dir + 'train_case_weights.out', train_case_weights, delimiter = ',')
								np.savetxt(checkpoint_dir + 'train_sentence_weights.out', train_sentence_weights, delimiter = ',')

								for i in range(config.max_max_epoch):
										lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
										m.assign_lr(session, config.learning_rate * lr_decay)

										print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
										train_perplexity = run_epoch(session, m, new_train_data, m.train_op, verbose=False)
										print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
										valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
										print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
										
										if (i+1) % 5 == 0 or (i+1) == config.max_max_epoch:
												saver.save(session, checkpoint_dir + 'model.ckpt', global_step = i+1)
						else:
								ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
								if ckpt and ckpt.model_checkpoint_path:
										saver.restore(session, ckpt.model_checkpoint_path)

						if train:

								case_scores = output_training_set_error_for_boosting(session, m2, train_data, tf.no_op())
						

								score = sum(case_scores)
								norm = len(case_scores) * -1 * gpu.log(float(1.0) / eval_config.vocab_size)
								epsilon_t =  (1 - (norm - score) / norm) / 2.0
								alpha_t = 0.5 * gpu.log((1 - epsilon_t)/ epsilon_t)
								alpha_t_list.append(alpha_t)

								if iii == 0:
										shutil.rmtree('simple-examples/ckpt/' + 'random training order ' + algo_name + '_' + model_name + '/' + 'alpha_t.out', ignore_errors = True)

								with open('simple-examples/ckpt/' + 'random training order ' + algo_name + '_' + model_name + '/' + 'alpha_t.out', 'ab') as f:
										f.write(str(alpha_t))
										f.write(',')
								
								train_case_weights = gpu.sqrt((1 - epsilon_t)/ epsilon_t) * np.multiply(train_case_weights, np.asarray(case_scores))
								train_case_weights = np.ravel(normalize(np.asarray(train_case_weights).reshape(1,-1), norm = 'l1'))
								
								if method == 'stddev':
										new_train_case_weights = reject_outliers(train_case_weights)
								elif method == 'sqrt':
										new_train_case_weights = sqrt_norm(train_case_weights)
								else:
										new_train_case_weights = train_case_weights
					
								start_idx = 0

								for i in range(len(train_sentence_list)):
										
										this_sentence_length = len(train_sentence_list[i])
										
										sentence_tokens = [v for v in new_train_case_weights[start_idx:(this_sentence_length+start_idx)] if np.isfinite(v)]
										if len(sentence_tokens) == 0:
												this_sentence_weights[i] = 0
										else:
												train_sentence_weights[i] = np.mean([v for v in new_train_case_weights[start_idx:(this_sentence_length+start_idx)] if np.isfinite(v)])
										


										start_idx += this_sentence_length


								train_sentence_weights = np.ravel(normalize(np.asarray(train_sentence_weights).reshape(1,-1), norm = 'l1'))

								for k,v in id_to_sentence_num_dict.items():
										total_wt = 0
										num_sentences = len(v)
										for sent_idx in v:
												total_wt += train_sentence_weights[sent_idx]
										id_to_weight[k] = total_wt/num_sentences

						test_set_probs = output_test_set_probs(session, mtest, test_data, tf.no_op(), partition = True)
						test_set_probs_no_alpha = test_set_probs
						np.savetxt(checkpoint_dir + 'test_set_probs_no_alpha.out', np.squeeze(test_set_probs_no_alpha), delimiter = ',')

						train_set_probs = output_test_set_probs(session, m2, train_data, tf.no_op(), partition = True)
						train_set_probs_no_alpha = train_set_probs
						
						np.savetxt(checkpoint_dir + 'train_set_probs_no_alpha.out', np.squeeze(train_set_probs_no_alpha), delimiter = ',')            
					
						test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
						print("Test Perplexity: %.3f" % test_perplexity)
		
		if random_training_order:
				new_folder = 'random training order ' + algo_name + '_' + model_name
		else:
				new_folder = algo_name + '_' + model_name
						
		checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'
		with open(checkpoint_dir + 'id_to_model.out', 'w') as f:
				for k,v in id_to_model.items():
						f.write(str(k) + ',' + ','.join(str(id_to_model[k])) + '\n')

		with open(checkpoint_dir + 'id_to_sent_num.out', 'w') as ff:
				for k,v in id_to_sentence_num_dict.items():
						ff.write(str(k) + ',' + ','.join(str(id_to_sentence_num_dict[k])) + '\n')

		print('Test PPL: ' + str(evaluate_unigram_partition(data = test_data, batch_size = 1, num_steps = 1, num_ensembles = num_ensembles, eos_id = eos_id, fp = 'simple-examples/ckpt/random training order unigram_partition_small/', probs_fn = 'test_set_probs_no_alpha.out')))
		print('Train PPL: ' + str(evaluate_unigram_partition(data = train_data, batch_size = 1, num_steps = 1, num_ensembles = num_ensembles, eos_id = eos_id, fp = 'simple-examples/ckpt/random training order unigram_partition_small/', probs_fn = 'train_set_probs_no_alpha.out')))

#Scaling function that continues taking square roots of the class weights until the largest weight < k * smallest weight.
def sqrt_norm(data, k=10):
		#print(np.shape(data))
		for i in range(np.shape(data)[0]):
				if data[i] == 0:
						data[i] += 10**-30
		smallest_idx = np.argmin(data)
		largest_idx = np.argmax(data)
		num_iterations = 0
		while(abs(data[largest_idx]) > (k*abs(data[smallest_idx]) + 10**-10)):
				#print(data[largest_idx])
				#print(data[smallest_idx])
				data = np.sqrt(data)
				num_iterations += 1
				#print(np.shape(data))
				#print(num_iterations)
		return(data)

#Scaling function that effectively removes any class weights that are outside m standard deviations of the mean and treats them as outliers that don't interfere with calculations
def reject_outliers(data, m=3):
	data = np.array(data)
	outlier_idx = np.where(abs(data - gpu.mean(data)) >= m * np.std(data))
	data[outlier_idx] = np.inf
	return(data)

#Helper function that evaluates the performance of the unigram_partition method
def evaluate_unigram_partition(data, batch_size, num_steps, num_ensembles, eos_id, fp = 'simple-examples/ckpt/random training order unigram_partition_small/', probs_fn = 'test_set_probs_no_alpha.out'):
		epoch_size = ((len(data) // batch_size) - 1) // num_steps
		start_time = time.time()
		costs = 0.0
		iters = 0
		
		full_probs = []
		for i in range(num_ensembles):
				print(i)
				#full_probs[i] = np.loadtxt(fp + 'ensemble' + str(i+1) + '/test_set_probs_no_alpha.out', delimiter = ',')
				full_probs.append(np.asarray(pd.read_csv(fp + 'ensemble' + str(i+1) + '/' + probs_fn, delimiter = ',', header = None)))
				print(np.shape(full_probs[i]))
				# for ii in range(len(full_probs[i])):
				#     full_probs[i][ii] = full_probs[i][ii] / np.sum(full_probs[i][ii])

		print('reading in probs done')
		id_to_model = {}
		with open(fp + 'id_to_model.out', 'rb') as f:
				csv_reader = csv.reader(f, delimiter = ',', quotechar = '|')
				for row in csv_reader:
						row_list = [x for x in row if (x != '[' and x != ']' and x != '' and x != ' ')]
						#print(row_list)
						#row_list = row.split(',')
						row_list = [int(i) for i in row_list]
						id_to_model[row_list[0]] = row_list[1:len(row_list)]

		print('reading in id_to_model done')
		#print(id_to_model[1344])
		#probs = tf.nn.softmax(probs)
		# print(np.sum(probs[0]))
		# print(np.sum(probs[50]))
		#print(len(probs[0]))
		next_is_start_of_sentence = True
		flaggg = True
		#sent_list = reader.get_sentence_list(data = data, eos_id = eos_id)
		for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size,
																										num_steps)):
				if next_is_start_of_sentence:
						x = x[0,0]
						if x in id_to_model:
								models_included = id_to_model[x]
								coef = 1
						else:
								models_included = [1,2,3,4,5,6,7,8,9]
								coef = 1
				if x == eos_id:
						#cost = -1 * gpu.log(full_probs[0][step])
						models_included = [1]
						coef = 1
						next_is_start_of_sentence = True 
				else:
						next_is_start_of_sentence = False
						#coef = 0.5
						#models_included = id_to_model[x]
				probs = 0
				denom = 0
				for m in models_included:
						if m == 1:
								probs += full_probs[m-1][step]
								denom += 1
						else:
								#coef = 0.5
								probs += coef*full_probs[m-1][step]
								denom += coef
				probs = probs / float(denom)
				cost = -1 * gpu.log(probs)

				# print(step)
				# print(x)
				# print(y)
				# print(probs)
				#print(probs[0])
				#cost = -1 * gpu.log(probs[step][0,y[0,0]])
				#print(cost)
				'''
				loss = tf.nn.seq2seq.sequence_loss_by_example(
						[logits],
						[tf.reshape(y, [-1])],
						[tf.ones([batch_size * num_steps], dtype=tf.float64)])

				print(loss)
				cost = tf.reduce_sum(loss) / batch_size
				print(cost)
				'''
				costs += cost
				iters += num_steps
		
				if step % (epoch_size // 10) == 10:
						print("%.3f perplexity: %.3f speed: %.0f wps" %
								(step * 1.0 / epoch_size, gpu.exp(costs / iters),
								iters * batch_size / (time.time() - start_time)))

		return gpu.exp(costs / iters) 

#General method to just run 1 LSTM on PTB
def run_one_model(data_path = 'simple-examples/data', model_name = 'small', train = True, random_training_order = False, reverse_order = False, algo_name = 'hp_baseline', perc = 100, random_token_start = False, ensemble_fn = None, new_lr = None, num_epochs = None):
	#algo_name = 'hp_baseline'
	raw_data = reader.ptb_raw_data(data_path, perc = perc, random = random_token_start, ensemble_fn = ensemble_fn)
	train_data, valid_data, test_data, vocab_size, word_to_id = raw_data

	if reverse_order:
		train_data.reverse()
		valid_data.reverse()
		test_data.reverse()
		algo_name = 'reverse ' + algo_name

	if random_training_order:
		train_sentence_list = reader.get_sentence_list(train_data, word_to_id['<eos>'], reverse_order)
		perm = range(len(train_sentence_list))
		np.random.shuffle(perm)
		train_data = []
		for idx in perm:
			train_data += train_sentence_list[idx]


	FLAGS.model = model_name

	config = get_config()
	
	eval_config = get_config()

	if num_epochs != None:
		config.max_max_epoch = num_epochs
		eval_config.max_max_epoch = num_epochs
	
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	if new_lr != None:
		config.learning_rate = new_lr
		eval_config.learning_rate = new_lr
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
	
	with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as session:
		
		#if ensemble_fn == None or ensemble_fn == 'ensemble0':
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
			m = PTBModel(is_training=True, config=config)
			#sess.close()
		
		with tf.variable_scope("model", reuse=True, initializer=initializer):
			mvalid = PTBModel(is_training=False, config=config)
			mtest = PTBModel(is_training=False, config=eval_config)
			#m2 = PTBModel(is_training=False, config=eval_config)
		
		#else:
			
		tf.initialize_all_variables().run()

		saver = tf.train.Saver()

		if random_training_order:
			new_folder = 'random training order' + algo_name + '_' + model_name + '/'
		else:
			new_folder = algo_name + '_' + model_name + '/'

		if ensemble_fn != None:
			if ensemble_fn != 'ensemble0':
				checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/' + 'ensemble0/'
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(session, ckpt.model_checkpoint_path)
			new_folder += ensemble_fn + '/'
				
		
		checkpoint_dir = 'simple-examples/ckpt/' + new_folder + '/'

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		

		if train:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, train_data, m.train_op, verbose=False)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				
				
				valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

				if (i+1) == config.max_max_epoch:
					saver.save(session, checkpoint_dir + 'model.ckpt', global_step = i+1)
		else:
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(session, ckpt.model_checkpoint_path)

		#print('test set computation for this classifier has been completed')
		test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
		print("Test Perplexity: %.3f" % test_perplexity)
		full_probs = output_test_set_probs(session, mtest, test_data, tf.no_op(), partition = True)
		np.savetxt(checkpoint_dir + 'test_set_probs.out', np.squeeze(full_probs), delimiter = ',')
		


		return(full_probs)

#Ensemble method that uses a pretrained starting point for the weights of the network to build highly specific models on smaller data. Does not work well either.
def naive_pretrain_ensemble(data_path, num_ensembles, model_name, train = True, random_training_order = False, reverse_order = False, new_lr = None):
		for i in range(num_ensembles):
				if i == 0:
						output_probs = np.array(run_one_model(data_path = data_path, model_name = model_name, train = False, random_training_order = random_training_order, reverse_order = reverse_order, algo_name = 'pretrain_test', perc = 100, random_token_start = False, ensemble_fn = 'ensemble' + str(i)))
				else:
						output_probs = np.add(output_probs, np.array(run_one_model(data_path = data_path, model_name = model_name, train = train, random_training_order = random_training_order, reverse_order = reverse_order, algo_name = 'pretrain_test', perc = 25, random_token_start = True, ensemble_fn = 'ensemble' + str(i), new_lr = new_lr, num_epochs = 1)))
						#output_probs += run_one_model(data_path = data_path, model_name = model_name, train = train, random_training_order = random_training_order, reverse_order = reverse_order, algo_name = 'pretrain_test', perc = 50, random_token_start = True, ensemble_fn = 'ensemble' + str(i), new_lr = new_lr)
		output_probs = output_probs
		output_probs = output_probs / float(num_ensembles)
		ensemble_test_perplexity = gpu.exp(-1 * sum(np.log(output_probs)) / float(len(output_probs)))
		print("Ensemble Test Perplexity: %.3f" % ensemble_test_perplexity)