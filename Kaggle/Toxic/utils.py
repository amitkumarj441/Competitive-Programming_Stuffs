import re
import pandas as pd
import numpy as np
import tensorflow as tf
import collections
import tqdm
import time

def review_to_wordlist(review):
  words = review.lower().split()
  return words

def review_to_wordlist_char(review):
  words = review.split()
  return words

def review_to_sentences(review, tokenizer, remove_stopwords=False):
  raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
  return sentences

def load_vocab(vocab_dir):
  f = open(vocab_dir, 'r')
  index2word = f.readlines()
  index2word = map(lambda x: x.strip(), index2word)
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  print "%d words loadded from vocab file" % len(index2word)
  return index2word, word2index

def load_char(char_dir):
  f = open(char_dir, 'r')
  index2char = f.readlines()
  index2char = map(lambda x: x.strip(), index2char)
  char2index = dict([(char, idx) for idx, char in enumerate(index2char)])
  print "%d char loadded from vocab file" % len(index2char)
  return index2char, char2index

def load_glove(pretrain_dir, vocab):
  embedding_dict = {}
  f = open(pretrain_dir,'r')
  for row in f:
    values = row.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embedding_dict[word] = vector
  f.close()
  vocab_size = len(vocab)
  embedding = np.zeros((vocab_size, 300))
  for idx, word in enumerate(vocab):
    word_vector = embedding_dict.get(word)
    if word_vector is not None:
      embedding[idx] = word_vector
  return embedding

def load_fasttext(pretrain_dir, vocab):
  embedding_dict = {}
  f = open(pretrain_dir, 'r')
  for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
    values = row.split(" ")
    word = values[0]
    vector = np.array([float(num) for num in values[1:-1]])
    embedding_dict[word] = vector
  f.close()
  vocab_size = len(vocab)
  embedding = np.zeros((vocab_size, 300))
  for idx, word in enumerate(vocab):
    word_vector = embedding_dict.get(word)
    if word_vector is not None:
      embedding[idx] = word_vector
  return embedding

def build_vocab(sentences, max_words=None):
  word_count = collections.Counter()
  for sentence in sentences:
    for word in sentence.split():
      word_count[word] += 1

  print "the dataset has %d different words totally" % len(word_count)
  if not max_words:
    max_words = len(word_count)
  filter_out_words = len(word_count) - max_words
  word_dict = word_count.most_common(max_words)
  index2word = ["<unk>"] + [word[0] for word in word_dict]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])

  print "%d words filtered out of the vocabulary and %d words in the vocabulary" % \
      (filter_out_words, len(index2word))
  return index2word, word2index

def build_vocab_char(sentences, max_char=None):
  char_count = collections.Counter()
  for sentence in sentences:
    for ch in sentence:
      char_count[ch] += 1

  print "the dataset has %d different chars totally" % len(char_count)
  if not max_char:
    max_char = len(char_count)
  filter_out_char = len(char_count) - max_char
  char_dict = char_count.most_common(max_char)
  index2char = ["<unk>"] + [ch[0] for ch in char_dict]
  char2index = dict([(char, idx) for idx, char in enumerate(index2char)])

  print "%d char filtered out of the vocabulary and %d char in the vocabulary" % \
      (filter_out_char, len(index2char))
  return index2char, char2index

def vectorize(data, word_dict, verbose=True):
  reviews = []
  for idx, line in enumerate(data):
    seq_line = [word_dict[w] if w in word_dict else 1 for w in line]
    reviews.append(seq_line)

    if verbose and (idx % 10000 == 0):
      print("Vectorization: processed {}".format(idx))
  return reviews

def vectorize_char(data, char_dict, verbose=True):
  reviews = []
  for idx, line in enumerate(data):
    char_line = []
    for word in line:
      char_line.append([char_dict[ch] if ch in char_dict else 1 for ch in word])
    reviews.append(char_line)

    if verbose and (idx % 10000 == 0):
      print("Vectorization: processed {}".format(idx))
  return reviews

def padding_data_for_rnn(sentences):
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=True):
  """
    batch all data index into a list
  """
  idx_list = np.arange(n_data)
  if shuffle:
    np.random.shuffle(idx_list)
  batch_index = []
  num_batches = int(np.ceil(float(n_data) / batch_size))
  for idx in range(num_batches):
    start_idx = idx * batch_size
    batch_index.append(idx_list[start_idx: min(start_idx + batch_size, n_data)])
  return batch_index

def get_batches(sentences, labels, batch_size, max_len=None, shuffle=True):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=shuffle)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    lab_batch = [labels[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, lab_batch))
  return all_batches

def get_batches_with_char(sentences, chars, labels, batch_size, max_len=None, shuffle=True):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=shuffle)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    char_batch = [chars[t] for t in minibatch]
    lab_batch = [labels[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    ch = map(lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, 10), char_batch)
    ch = tf.keras.preprocessing.sequence.pad_sequences(ch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ch, lab_batch))
  return all_batches

def get_batches_with_fe(sentences, labels, ex_features, batch_size, max_len=None, shuffle=True):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=shuffle)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    ex_batch = ex_features[minibatch]
    lab_batch = [labels[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ex_batch, lab_batch))
  return all_batches

def get_batches_with_charfe(sentences, chars, labels, ex_features, batch_size, max_len=None, shuffle=True):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=shuffle)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    char_batch = [chars[t] for t in minibatch]
    ex_batch = ex_features[minibatch]
    lab_batch = [labels[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    ch = map(lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, 10), char_batch)
    ch = tf.keras.preprocessing.sequence.pad_sequences(ch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ch, ex_batch, lab_batch))
  return all_batches

def get_test_batches(sentences, batch_size, max_len=None):
  """
    load test data
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=False)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len))
  return all_batches

def get_test_batches_with_char(sentences, chars, batch_size, max_len=None):
  """
    load test data
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=False)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    char_batch = [chars[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    ch = map(lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, 10), char_batch)
    ch = tf.keras.preprocessing.sequence.pad_sequences(ch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ch))
  return all_batches

def get_test_batches_with_fe(sentences, ex_features, batch_size, max_len=None):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=False)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    ex_batch = ex_features[minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ex_batch))
  return all_batches

def get_test_batches_with_charfe(sentences, chars, ex_features, batch_size, max_len=None):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size, shuffle=False)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    ex_batch = ex_features[minibatch]
    char_batch = [chars[t] for t in minibatch]
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
    ch = map(lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, 10), char_batch)
    ch = tf.keras.preprocessing.sequence.pad_sequences(ch, max_len)
    seq_len = [max_len] * seq.shape[0]
    all_batches.append((seq, seq_len, ch, ex_batch))
  return all_batches
  
def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def one_hot(labels, nb_classes=None):
  labels = np.array(labels).astype("int32")
  if not nb_classes:
    nb_classes = np.max(labels) + 1
  onehot_labels = np.zeros((len(labels), nb_classes)).astype("float32")
  for i in range(len(labels)):
    onehot_labels[i, labels[i]] = 1.
  return onehot_labels

def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  print "loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time)
  return model

def create_or_load_model(model, model_dir, session, name):
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print "created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time)

  global_step = model.global_step.eval(session=session)
  return model, global_step
