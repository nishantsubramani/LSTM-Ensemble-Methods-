from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import gnumpy as gpu
import numpy as np
import tensorflow as tf

def _read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
	data = _read_words(filename)
	#print(data)
	counter = collections.Counter(data)
	#print(counter)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id


def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None, perc = 100, random = False, ensemble_fn = None,):
	"""Load PTB raw data from data directory "data_path".

	Reads PTB text files, converts strings to integer ids,
	and performs mini-batching of the inputs.

	The PTB dataset comes from Tomas Mikolov's webpage:

	http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

	Args:
		data_path: string path to the directory where simple-examples.tgz has
			been extracted.

	Returns:
		tuple (train_data, valid_data, test_data, vocabulary)
		where each of the data objects can be passed to PTBIterator.
	"""

	if perc == 100:
		train_path = os.path.join(data_path, "ptb.train.txt")
		valid_path = os.path.join(data_path, "ptb.valid.txt")
		test_path = os.path.join(data_path, "ptb.test.txt")

		word_to_id = _build_vocab(os.path.join(data_path, "ptb.train.txt"))
	
	else:
		if random:
			train_path = os.path.join(data_path, "ptb.train.txt")
			valid_path = os.path.join(data_path, "ptb.valid.txt")
			test_path = os.path.join(data_path, "ptb.test.txt")
			word_to_id = _build_vocab(os.path.join(data_path, "ptb.train.txt"))
			start_frac = ((100 - perc) / 100.0)
			train_start_idx = np.random.randint(0, np.ceil(start_frac * 929000), 1)
			
			_create_subset(train_path, os.path.join(data_path, ensemble_fn + 'train.txt'), 929000*perc / 100.0, train_start_idx)
			train_path = os.path.join(data_path, ensemble_fn + 'train.txt')
			# word_to_id = _build_vocab(train_path)

			# _create_unk_subset(valid_path, os.path.join(data_path, ensemble_fn + 'valid.txt'), 100, word_to_id)
			# _create_unk_subset(test_path, os.path.join(data_path, ensemble_fn + 'test.txt'), 100, word_to_id)

			# #train_path = os.path.join(data_path, ensemble_fn + 'train.txt')
			# valid_path = os.path.join(data_path, ensemble_fn + 'valid.txt')
			# test_path = os.path.join(data_path, ensemble_fn + 'test.txt')
		
		else:
			_create_percent_partition(perc = perc)
			train_path = os.path.join(data_path, 'first.' + str(perc) + "perc.ptb.train.txt")
			valid_path = os.path.join(data_path, 'first.' + str(perc) + "perc.ptb.valid.txt")
			test_path = os.path.join(data_path, 'first.' + str(perc) + "perc.ptb.test.txt")

			word_to_id = _build_vocab(os.path.join(data_path, 'first.' + str(perc) + "perc.ptb.train.txt"))

	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	vocabulary = len(word_to_id)
	#print(vocabulary)
	return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps):
	"""Iterate on the raw PTB data.

	This generates batch_size pointers into the raw PTB data, and allows
	minibatch iteration along these pointers.

	Args:
		raw_data: one of the raw data outputs from ptb_raw_data.
		batch_size: int, the batch size.
		num_steps: int, the number of unrolls.

	Yields:
		Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
		The second element of the tuple is the same data time-shifted to the
		right by one.

	Raises:
		ValueError: if batch_size or num_steps are too high.
	"""
	raw_data = np.array(raw_data, dtype=np.int32)

	data_len = len(raw_data)
	batch_len = data_len // batch_size
	data = np.zeros([batch_size, batch_len], dtype=np.int32)
	for i in range(batch_size):
		data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

	epoch_size = (batch_len - 1) // num_steps

	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

	for i in range(epoch_size):
		x = data[:, i*num_steps:(i+1)*num_steps]
		y = data[:, i*num_steps+1:(i+1)*num_steps+1]
		yield (x, y)

def ptb_iterator_boost(raw_data, batch_size, num_steps, weights = False, seed = 1, forward = True):
	"""Iterate on the raw PTB data.

	This generates batch_size pointers into the raw PTB data, and allows
	minibatch iteration along these pointers.

	Args:
		raw_data: one of the raw data outputs from ptb_raw_data.
		batch_size: int, the batch size.
		num_steps: int, the number of unrolls.

	Yields:
		Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
		The second element of the tuple is the same data time-shifted to the
		right by one.

	Raises:
		ValueError: if batch_size or num_steps are too high.
	"""
	raw_data = np.array(raw_data, dtype=np.int32)

	data_len = len(raw_data)
	batch_len = data_len // batch_size
	data = np.zeros([batch_size, batch_len], dtype=np.int32)
	
	epoch_size = (batch_len - 1) // num_steps

	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

	if type(weights) is bool:
		weights = np.repeat(float(1)/data_len, data_len)


	#print(data_len)
	#print(num_steps)
	starting_locations = data_len - num_steps
	#print(len(starting_locations))

	# if forward:
	# 	weights = weights[0:(data_len-num_steps)]
	# else:
	# 	weights = weights[(num_steps - 1):(data_len - 1)]
	# 	#print(len(weights))
	
	# weights = naive_normalization(weights)
	
	np.random.seed(seed)
	weights = weights.ravel()

	full_boost_data = np.random.choice(a = starting_locations, size = epoch_size * batch_size, replace = True, p = weights)
	for i in range(epoch_size):
			#b = gpu.random.choice(a = starting_locations, size = batch_size, replace = True, p = weights)
			x = np.zeros([batch_size, num_steps], dtype=np.int32)
			y = np.zeros([batch_size, num_steps], dtype=np.int32)

			for j in range(batch_size):
					x[j,:] = raw_data[full_boost_data[i*batch_size + j]:(full_boost_data[i*batch_size + j] + num_steps)]
					y[j,:] = raw_data[(full_boost_data[i*batch_size + j] + 1):(full_boost_data[i*batch_size + j] + num_steps + 1)]
					# x[j,:] = raw_data[b[j]:(b[j] + num_steps)]
					# y[j,:] = raw_data[(b[j] + 1):(b[j] + num_steps + 1)]
			yield (x, y)




def get_sentence_list(data, eos_id, reverse = False):
		sentence_list = []
		new_sentence = []
		for i in range(len(data)):
				new_word = data[i]
				if (reverse and new_word == eos_id and len(new_sentence) > 0):
						sentence_list.append(new_sentence)
						new_sentence = []
						new_sentence.append(new_word)
				elif reverse == False and new_word == eos_id:
						new_sentence.append(new_word)
						sentence_list.append(new_sentence)
						new_sentence = []
				else:
						new_sentence.append(new_word)
		if len(sentence_list) > 0:
				sentence_list.append(new_sentence)
		return(sentence_list)

def weighted_sentence_selection(sentence_list, weights, random_order = True):
		#np.random.seed(seed)
		new_data = []
		num_sentences = len(sentence_list)
		sentence_indices = np.random.choice(a = num_sentences, size = np.ceil(num_sentences/2), replace = True, p = weights)

		if random_order:
				for i in range(num_sentences):
						new_data += sentence_list[i]
				for sentence_idx in sentence_indices:
						new_data += sentence_list[sentence_idx]
		else:
				new_idx = np.concatenate((range(num_sentences),sentence_indices))
				new_idx = np.sort(new_idx)
				for idx in new_idx:
						new_data += sentence_list[idx]
		#print(new_data)
		return(new_data)


def naive_normalization(x):
	new_x = []

	for entry in x:
		new_x.append(float(entry)/sum(x))
	return(new_x)

def _create_subset(read_fn, write_fn, num_tokens, start_idx = 0):
	start = False
	with open(write_fn, "w") as f1:
		with open(read_fn, "r") as f2:
			for line in f2:
				if(num_tokens <= 0):
					break
				else:
					words = line.split()
					if start_idx <= 0:
						start = True
					if start:
						num_tokens -= len(words) + 1
						f1.write(' ' + line + ' \n')
					else:
						start_idx -= len(words) + 1
						#print(start_idx)


def _create_unk_subset(read_fn, write_fn, num_tokens, word_to_id, start_idx = 0):
	start = False
	with open(write_fn, "w") as f1:
		with open(read_fn, "r") as f2:
			for line in f2:
				if(num_tokens <= 0):
					break
				else:
					if start_idx <= 0:
						start = True
					words = line.split()
					line_len = len(words)
					if start:
						for i in range(line_len):
							if words[i] not in word_to_id:
								words[i] = '<unk>'
						num_tokens -= line_len + 1
						f1.write(' ' +  ' '.join(words) + ' \n')
					else:
						start_idx -= line_len + 1

def _create_percent_partition(input_train_fn = 'simple-examples/data/ptb.train.txt', input_valid_fn = 'simple-examples/data/ptb.valid.txt', input_test_fn = 'simple-examples/data/ptb.test.txt', perc = 100, random_start = False):
	if random_start:
		train_start = np.random.random_integers(0, 929000 - np.ceil(929000*perc/100.0))
		new_train_fn = 'simple-examples/data/rdm.' + str(int(perc)) + '.' + str(train_start) + '.perc.ptb.train.txt'
		_create_subset(read_fn = input_train_fn, write_fn = new_train_fn, num_tokens = 929000*perc / 100.0, start_idx = train_start)
		
		word_to_id = _build_vocab(new_train_fn)
		vocabulary = len(word_to_id)
		valid_start = np.random.random_integers(0, 73000 - np.ceil(73000*perc/100.0))
		test_start = np.random.random_integers(0, 82000 - np.ceil(82000*perc/100.0))
		new_valid_fn = 'simple-examples/data/rdm.' + str(int(perc)) + '.' + str(valid_start) + '.perc.ptb.valid.txt'
		new_test_fn = 'simple-examples/data/rdm.' + str(int(perc)) + '.' + str(valid_start) + '.perc.ptb.test.txt'
		
		_create_unk_subset(read_fn = input_valid_fn, write_fn = new_valid_fn, num_tokens = 73000*perc / 100.0, word_to_id = word_to_id, start_idx = valid_start)
		_create_unk_subset(read_fn = input_test_fn, write_fn = new_test_fn, num_tokens = 82000*perc / 100.0, word_to_id = word_to_id, start_idx = test_start)
	else:
		new_train_fn = 'simple-examples/data/first.' + str(int(perc)) + 'perc.ptb.train.txt'
		_create_subset(input_train_fn, new_train_fn, 929000*perc / 100.0)
		
		word_to_id = _build_vocab(new_train_fn)
		vocabulary = len(word_to_id)
		new_valid_fn = 'simple-examples/data/first.' + str(int(perc)) + 'perc.ptb.valid.txt'
		new_test_fn = 'simple-examples/data/first.' + str(int(perc)) + 'perc.ptb.test.txt'
		
		_create_unk_subset(input_valid_fn, new_valid_fn, 73000*perc / 100.0, word_to_id)
		_create_unk_subset(input_test_fn, new_test_fn, 82000*perc / 100.0, word_to_id)

	return(new_train_fn, new_valid_fn, new_test_fn, vocabulary, word_to_id)


# _create_percent_partition(perc = 5)
# _create_percent_partition(perc = 10)
# _create_percent_partition(perc = 15)
# _create_percent_partition(perc = 20)

#_create_subset('simple-examples/data/ptb.train.txt', 'simple-examples/data/15perc.ptb.train.txt', 0.15*929000)

#_create_subset('simple-examples/data/ptb.valid.txt', 'simple-examples/data/med.ptb.valid.txt', 73000/929000 * 100000)

#_create_subset('simple-examples/data/ptb.test.txt', 'simple-examples/data/med.ptb.test.txt', 82000/929000 * 100000)


