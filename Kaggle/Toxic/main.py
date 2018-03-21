import pandas as pd
import numpy as np
import config
import utils
import time
import os

from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
from model import TextCNN, TextRNN, TextRNNChar, TextCNNChar, TextRNNFE
from model import TextCNNFE, TextRNNCharFE, TextRCNN, TextCNNChar2
import tensorflow as tf

def add_features(file):
  df = pd.read_csv(file)
  df['total_length'] = df['comment_text'].apply(len)
  df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
  df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
  df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
  df['unique_vs_length'] = df.apply(lambda row: float(row['num_unique_words'])/float(row['total_length']),axis=1)
  return hstack([csr_matrix(np.reshape(df['caps_vs_length'].values, (-1, 1))),
                 csr_matrix(np.reshape(df['unique_vs_length'].values, (-1, 1)))]).toarray()

def main(args):
  print "loadding data and labels from dataset"
  train = pd.read_csv(args.train_dir)
  ch_train = pd.read_csv(args.chtrain_dir)
  x_train = train["comment_text"]
  x_chtrain = ch_train["comment_text"]
  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  x = []
  x_ch = []
  for line in x_train:
    if len(line) > 0:
      x.append(utils.review_to_wordlist(line.strip()))
  print "loaded %d comments from dataset" % len(x)
  for line in x_chtrain:
    if len(line) > 0:
      x_ch.append(utils.review_to_wordlist_char(line.strip()))
  print "loaded %d comments from dataset" % len(x)
  y = train[target_cols].values

  index2word, word2index = utils.load_vocab(args.vocab_dir)
  index2char, char2index = utils.load_char(args.char_dir)
  x_vector = utils.vectorize(x, word2index, verbose=False)
  x_vector = np.array(x_vector)
  char_vector = utils.vectorize_char(x_ch, char2index, verbose=False)
  char_vector = np.array(char_vector)
  print char_vector[0]

  save_dir = os.path.join(args.save_dir, args.model_type)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  if args.model_type in ["cnn", "cnnfe", "chcnn", "chcnn2"]:
    max_step = args.max_step_cnn
    max_size = args.max_size_cnn
    nb_epochs = args.nb_epochs_cnn
  elif args.model_type in ["rnn", "rnnfe", "rnnfe2", "chrnn", "chrnnfe", "rcnn"]: 
    max_step = args.max_step_rnn
    max_size = args.max_size_rnn
    nb_epochs = args.nb_epochs_rnn

  ex_features = add_features("../data/train.csv")
  nfolds = args.nfolds
  skf = KFold(n_splits=nfolds, shuffle=True, random_state=2018)
  test_prob = []
  stack_logits = np.zeros((len(x_vector), len(target_cols)))
  for (f, (train_index, test_index)) in enumerate(skf.split(x_vector)):
    x_train, x_eval = x_vector[train_index], x_vector[test_index]
    char_train, char_eval = char_vector[train_index], char_vector[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    with tf.Graph().as_default():
      config_proto = utils.get_config_proto()
      sess = tf.Session(config=config_proto)
      if args.model_type == "cnn":
        model = TextCNN(args, "TextCNN")
      elif args.model_type == "cnnfe":
        model = TextCNNFE(args, "TextCNNFE")
      elif args.model_type == "rnn":
        model = TextRNN(args, "TextRNN")
      elif args.model_type == "rnnfe":
        model = TextRNNFE(args, "TextRNNFE")
      elif args.model_type == "rcnn":
        model = TextRCNN(args, "TextRCNN")
      elif args.model_type == "attention":
        model = RNNWithAttention(args, "Attention")
      elif args.model_type == "chrnn":
        model = TextRNNChar(args, "TextRNNChar")
      elif args.model_type == "chcnn":
        model = TextCNNChar(args, "TextCNNChar")
      elif args.model_type == "chcnn2":
        model = TextCNNChar(args, "TextCNNChar2")
      elif args.model_type == "rnnfe2":
        model = TextRNNFE2(args, "TextCNNCharFE2")
      elif args.model_type == "chrnnfe":
        model = TextRNNCharFE(args, "TextCNNCharFE")
      else:
        raise ValueError("Unknown model_type %s" % args.model_type)      
      sess.run(tf.global_variables_initializer())

      if args.use_ft: 
        pretrain_dir = args.ft_dir
        print "use FastText word vector"
        embedding = utils.load_fasttext(pretrain_dir, index2word)
      if not args.use_ft:
        pretrain_dir = args.glove_dir
        print "use Glove word vector"
        embedding = utils.load_glove(pretrain_dir, index2word)
      sess.run(model.embedding_init, {model.embedding_placeholder: embedding})

      for line in model.tvars:
        print line

      print "training %s model for toxic comments classification" % (args.model_type)
      print "%d fold start training" % f
      for epoch in range(1, nb_epochs + 1):
        print "epoch %d start with lr %f" % (epoch, model.learning_rate.eval(session=sess)), "\n", "- " * 50
        loss, total_comments = 0.0, 0
        if args.model_type in ["cnn", "rnn", "rcnn"]:
          train_batch = utils.get_batches(x_train, y_train, args.batch_size, args.max_len)
          valid_batch = utils.get_batches(x_eval, y_eval, max_size, args.max_len, False)

        elif args.model_type in ["chrnn", "chcnn", "chcnn2"]:
          train_batch = utils.get_batches_with_char(x_train, char_train, y_train, args.batch_size, args.max_len)
          valid_batch = utils.get_batches_with_char(x_eval, char_eval, y_eval, max_size, args.max_len, False)

        elif args.model_type in ["rnnfe", "cnnfe", "rnnfe2"]:
          train_batch = utils.get_batches_with_fe(x_train, y_train, ex_features, args.batch_size, args.max_len)
          valid_batch = utils.get_batches_with_fe(x_eval, y_eval, ex_features, max_size, args.max_len, False)

        elif args.model_type in ["chrnnfe"]:
          train_batch = utils.get_batches_with_charfe(x_train, char_train, y_train, ex_features, args.batch_size, args.max_len)
          valid_batch = utils.get_batches_with_charfe(x_eval, char_eval, y_eval, ex_features, max_size, args.max_len, False)

        epoch_start_time = time.time()
        step_start_time = epoch_start_time
        for idx, batch in enumerate(train_batch):
          if args.model_type in ["cnn", "rnn", "rcnn"]:
            comments, comments_length, labels = batch
            _, loss_t, global_step, batch_size = model.train(sess, comments, comments_length, labels)

          elif args.model_type in ["chrnn", "chcnn", "chcnn2"]:
            comments, comments_length, chs, labels = batch
            _, loss_t, global_step, batch_size = model.train(sess, comments, comments_length, chs, labels)

          elif args.model_type in ["rnnfe", "cnnfe", "rnnfe2"]:
            comments, comments_length, exs, labels = batch
            _, loss_t, global_step, batch_size = model.train(sess, comments, comments_length, labels, exs)

          elif args.model_type in ["chrnnfe"]:
            comments, comments_length, chs, exs, labels = batch
            _, loss_t, global_step, batch_size = model.train(sess, comments, comments_length, chs, labels, exs)

          loss += loss_t * batch_size
          total_comments += batch_size

          if global_step % 200 == 0:
            print "epoch %d step %d loss %f time %.2fs"%(epoch,global_step,loss_t,time.time()-step_start_time)

          if global_step % 200 == 0:
            _ = run_valid(valid_batch, model, sess, args.model_type)
            # model.saver.save(sess, os.path.join(save_dir, "model.ckpt"), global_step=global_step)  
            step_start_time = time.time()

        epoch_time = time.time() - epoch_start_time
        sess.run(model.learning_rate_decay_op)
        print "%.2f seconds in this epoch with train loss %f" % (epoch_time, loss / total_comments)

      test_prob.append(run_test(args, model, sess))
      stack_logits[test_index] = run_valid(valid_batch, model, sess, args.model_type)

  preds = np.zeros((test_prob[0].shape[0], len(target_cols)))
  for prob in test_prob:
    preds += prob
    print prob[0]
  preds /= len(test_prob)
  print len(test_prob)
  write_predict(stack_logits, args.model_type)
  write_results(preds, args.model_type)

def run_valid(valid_data, model, sess, model_type):
  total_logits = []
  total_labels = []
  loss = 0.0
  for batch in valid_data:
    if model_type in ["cnn", "rnn", "rcnn"]:
      comments, comments_length, labels = batch
      loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels)

    elif model_type in ["chrnn", "chcnn", "chcnn2"]:
      comments, comments_length, chs, labels = batch
      loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, chs, labels)

    elif model_type in ["rnnfe", "cnnfe", "rnnfe2"]:
      comments, comments_length, exs, labels = batch
      loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels, exs)

    elif model_type in ["chrnnfe"]:
      comments, comments_length, chs, exs, labels = batch
      loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, chs, labels, exs)

    total_logits += logits_t.tolist()
    total_labels += labels
    loss += loss_t * batch_size
  auc_tf = tf.metrics.auc(labels=np.array(total_labels), predictions=np.array(total_logits))
  sess.run(tf.local_variables_initializer())
  auc_tf = sess.run(auc_tf)
  print "auc %f in %d valid comments %f" % (auc_tf[1], np.array(total_logits).shape[0], loss/len(total_labels))  
  return total_logits

def run_test(args, model, sess):
  test = pd.read_csv(args.test_dir)
  ch_test = pd.read_csv(args.chtest_dir)
  x_test = test["comment_text"]
  x_chtest = ch_test["comment_text"]
  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  x = []
  x_ch = []
  for line in x_test:
    if len(line) > 0:
      x.append(utils.review_to_wordlist(line.strip()))
  print "loaded %d comments from test dataset" % len(x)
  for line in x_chtest:
    if len(line) > 0:
      x_ch.append(utils.review_to_wordlist_char(line.strip()))
  print "loaded %d comments from dataset" % len(x)

  index2word, word2index = utils.load_vocab(args.vocab_dir)
  index2char, char2index = utils.load_char(args.char_dir)
  x_vector = utils.vectorize(x, word2index, verbose=False)
  x_vector = np.array(x_vector)
  char_vector = utils.vectorize_char(x_ch, char2index, verbose=False)
  char_vector = np.array(char_vector)

  ex_features = add_features("../data/test.csv")
  if args.model_type in ["cnn"]:
    test_batch = utils.get_test_batches(x_vector, args.max_size_cnn, args.max_len)
  elif args.model_type in ["rnn", "rcnn"]:
    test_batch = utils.get_test_batches(x_vector, args.max_size_rnn, args.max_len)
  elif args.model_type in ["chrnn"]:
    test_batch = utils.get_test_batches_with_char(x_vector, char_vector, args.max_size_rnn, args.max_len)
  elif args.model_type in ["chcnn", "chcnn2"]:
    test_batch = utils.get_test_batches_with_char(x_vector, char_vector, args.batch_size, args.max_len)
  elif args.model_type in ["rnnfe", "cnnfe", "rnnfe2"]:
    test_batch = utils.get_test_batches_with_fe(x_vector, ex_features, args.max_size_rnn, args.max_len)
  elif args.model_type in ["chrnnfe"]:
    test_batch = utils.get_test_batches_with_charfe(x_vector, char_vector, ex_features, args.max_size_rnn, args.max_len)

  total_logits = []
  for batch in test_batch:
    if args.model_type in ["cnn", "rnn", "rcnn"]:
      comments, comments_length = batch
      logits = model.get_logits(sess, comments, comments_length).tolist()
    elif args.model_type in ["chrnn", "chcnn", "chcnn2"]:
      comments, comments_length, chs = batch
      logits = model.get_logits(sess, comments, comments_length, chs).tolist()
    elif args.model_type in ["rnnfe", "cnnfe", "rnnfe2"]:
      comments, comments_length, exs = batch
      logits = model.get_logits(sess, comments, comments_length, exs).tolist()
    elif args.model_type in ["chrnnfe"]:
      comments, comments_length, chs, exs = batch
      logits = model.get_logits(sess, comments, comments_length, chs, exs).tolist()
    total_logits += logits
  return np.array(total_logits)

def write_predict(logits, model_type):
  cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  train = pd.read_csv('../data/train.csv')    
  trainid = pd.DataFrame({'id': train["id"]})
  train = pd.concat([trainid, pd.DataFrame(logits, columns=cols)], axis=1)
  train.to_csv('logits-%s.csv' % model_type, index=False)
  print train.shape

def write_results(logits, model_type):
  cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  submission = pd.read_csv('../data/sample_submission.csv')    
  submid = pd.DataFrame({'id': submission["id"]})
  submission = pd.concat([submid, pd.DataFrame(logits, columns=cols)], axis=1)
  submission.to_csv('submission-%s.csv' % model_type, index=False)
  print submission.shape

if __name__ == '__main__':
  args = config.get_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  if args.model_type in ["cnn", "chrnn", "chrnnfe", "cnnfe"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  main(args)
