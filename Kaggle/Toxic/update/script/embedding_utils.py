import numpy as np
import tqdm

def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []

    f = open(file_path)
    for index, line in enumerate(tqdm.tqdm(f)):
        if index == 0:
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_embedding_list_glove(file_path):
    embedding_word_dict = {}
    embedding_list = []

    f = open(file_path)
    for index, line in enumerate(tqdm.tqdm(f)):
        values = line.split()
        word = values[0]
        i = 1
        while not isfloat(values[i]):
            word = word + ' ' + values[i]
            i = i + 1
        if i > 1: print('Exceptional case: {0}'.format(word))
        try:
            coefs = np.asarray(values[i:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train
