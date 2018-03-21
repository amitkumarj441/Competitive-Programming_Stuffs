import argparse

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1023, help='random seed')
  #data
  parser.add_argument('--train_dir', type=str, default='../data/pre/train.csv', help="train data")
  parser.add_argument('--test_dir', type=str, default='../data/pre/test.csv', help="test data")
  parser.add_argument('--chtrain_dir', type=str, default='../data/pre/train_ch.csv', help="char train data")
  parser.add_argument('--chtest_dir', type=str, default='../data/pre/test_ch.csv', help="char test data")
  parser.add_argument('--vocab_dir', type=str, default='../data/pre/vocab.txt', help='vocab dir')
  parser.add_argument('--char_dir', type=str, default='../data/pre/char.txt', help='char dir')
  parser.add_argument('--ft_dir', type=str, default='../data/glove/crawl-300d-2M.vec', help='fasttext dir')
  parser.add_argument('--glove_dir', type=str, default='../data/glove/glove.840B.300d.txt', help='glove dir')
  parser.add_argument('--use_ft', type=bool, default=True, help='whether use glove')
  parser.add_argument('--save_dir', type=str, default='save/saves')

  # model
  parser.add_argument('--model_type', type=str, default="cnn", help='cnn,rnn,attention,chrnn,cnnfe,rnnfe')
  parser.add_argument('--nb_classes', type=int, default=6, help='class numbers')
  parser.add_argument('--max_len', type=int, default=200, help='The length of input x')
  parser.add_argument('--vocab_size', type=int, default=100002, help='data vocab size')
  parser.add_argument('--embed_size', type=int, default=300, help='dims of word embedding')
  parser.add_argument('--char_vocab_size', type=int, default=206, help='data char vocab size')
  parser.add_argument('--char_embed_size', type=int, default=100, help='dims of char embedding')
  parser.add_argument('--dropout_eb', type=float, default=0.4, help='dropout of embedding')

  #cnn
  parser.add_argument('--filter_sizes', type=list, default=[1, 2, 3, 4, 5], help='')
  parser.add_argument('--char_filter_size', type=int, default=5, help='char filter size')
  parser.add_argument('--num_filters', type=int, default=128, help='num of filters')
  parser.add_argument('--char_num_filters', type=int, default=100, help='num of char filters')
  parser.add_argument('--dropout_cnn', type=float, default=0.5, help='keep prob in cnn')
  parser.add_argument('--max_size_cnn', type=int, default=1000, help='max numbers every batch of cnn')
  parser.add_argument('--max_step_cnn', type=int, default=20000, help='max cnn train step') #tobe
  parser.add_argument('--nb_epochs_cnn', type=int, default=5, help='Number of epoch')
  
  #rnn
  parser.add_argument('--cell_type', type=str, default="gru", help='lstm or gru')
  parser.add_argument('--hidden_size', type=int, default=512, help='rnn hidden size')
  parser.add_argument('--rnn_layers', type=int, default=2, help='rnn layers')
  parser.add_argument('--max_size_rnn', type=int, default=400, help='max numbers every batch of rnn')
  parser.add_argument('--max_step_rnn', type=int, default=16000, help='max rnn train step') #tobe
  parser.add_argument('--nb_epochs_rnn', type=int, default=4, help='Number of epoch')

  parser.add_argument('--batch_size', type=int, default=128, help='example numbers every batch')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
  parser.add_argument('--lr_decay', type=float, default=0.6, help='learning rate decay rate')
  parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max norm of gradient')
  parser.add_argument('--nfolds', type=int, default=10, help='cv') 

  return parser.parse_args()
