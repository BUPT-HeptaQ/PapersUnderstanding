# pre processing: word(en, cn) -> number (not hot vector)

import os
import numpy as np
import nltk
import pickle
import argparse
import code

from collections import Counter


""" this part is used for configure """


def get_args():

    parser = argparse.ArgumentParser()

# data files
    parser.add_argument('--train_file', type=str, default=None,
                        help='training file')
    parser.add_argument('--develop_file', type=str, default=None,
                        help='development file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='test file')
    parser.add_argument('--vocab_file', type=str, default="vocab.pkl",
                        help='dictionary file')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='max number of vocabulary')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    return parser.parse_args()


""" All functions """


def load_data(in_file):
    chinese = []
    english = []

    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            english.append(["BOS"] + nltk.word_tokenize(line[0]) + ["EOS"])
            # split chinese sentences into characters
            chinese.append(["BOS"] + [C for C in line[1]] + ["EOS"])

    return english, chinese


def build_dict(sentences, max_words=10000):
    words_count = Counter()

    for sentence in sentences:
        for word in sentence:
            words_count[word] += 1

    max_word = words_count.most_common(max_words)
    total_words = len(max_word) + 1
    word_dictionary = {word[0]: index + 1 for (index, word) in enumerate(max_word)}
    word_dictionary["UNK"] = 0

    return word_dictionary, total_words


def encode(english_sentences, chinese_sentences, english_dictionary, chinese_dictionary, sort_by_len=True):
    """
    Encode the sequences.
    """
    length = len(english_sentences)
    out_english_sentences = []
    out_chinese_sentences = []

    for i in range(length):
        english_sequence = [english_dictionary[word] if word in english_dictionary
                            else 0 for word in english_sentences[i]]
        chinese_sequence = [chinese_dictionary[word] if word in chinese_dictionary
                            else 0 for word in chinese_sentences[i]]

        out_english_sentences.append(english_sequence)
        out_chinese_sentences.append(chinese_sequence)

    # sort sentences by english lengths(it is convenient to calculate batches)
    def length_argsort(sequence):
        return sorted(range(len(sequence)), key=lambda x: len(sequence[x]))

    if sort_by_len:
        sorted_index = length_argsort(out_english_sentences)
        out_english_sentences = [out_english_sentences[i] for i in sorted_index]
        out_chinese_sentences = [out_chinese_sentences[i] for i in sorted_index]

    return out_english_sentences, out_chinese_sentences


def get_mini_batches(n, mini_batch_size, shuffle=False):
    index_list = np.arange(0, n, mini_batch_size)
    mini_batches = []

    if shuffle:
        np.random.shuffle(index_list)
    for index in index_list:
        mini_batches.append(np.arange(index, min(index + mini_batch_size, n)))

    return mini_batches


def prepare_data(sequences):
    # convert mini batch of sequences into numpy matrix
    batch_sample = len(sequences)  # 128
    sequences_length = [len(sequence) for sequence in sequences]
    max_len = np.max(sequences_length)

    # indication of padding
    padding = np.zeros((batch_sample, max_len)).astype('int32')
    padding_mask = np.zeros((batch_sample, max_len)).astype('float32')

    for index, sequence in enumerate(sequences):
        padding[index, :sequences_length[index]] = sequence
        padding_mask[index, :sequences_length[index]] = 1.0

    return padding, padding_mask


def generate_examples(english_sentences, chinese_sentences, batch_size):
    mini_batches = get_mini_batches(len(english_sentences), batch_size)
    all_convert = []

    for mini_batch in mini_batches:
        # convert to numpy array
        mini_batch_english_sentences = [english_sentences[txt] for txt in mini_batch]
        mini_batch_chinese_sentences = [chinese_sentences[txt] for txt in mini_batch]

        mini_batch_english, mini_batch_english_mask = prepare_data(mini_batch_english_sentences)
        mini_batch_chinese, mini_batch_chinese_mask = prepare_data(mini_batch_chinese_sentences)

        all_convert.append((mini_batch_english, mini_batch_english_mask, mini_batch_chinese, mini_batch_chinese_mask))

    return all_convert


""" main function """


def main(args):
    train_english, train_chinese = load_data(args.train_file)
    develop_english, develop_chinese = load_data(args.develop_file)

    args.num_train = len(train_english)
    args.num_develop = len(develop_english)

    if os.path.isfile(args.vocab_file):
        english_dictionary, chinese_dictionary, english_total_words, \
            chinese_total_words = pickle.load(open(args.vocab_file, "rb"))

    else:
        english_dictionary, english_total_words = build_dict(train_english)
        chinese_dictionary, chinese_total_words = build_dict(train_chinese)
        pickle.dump([english_dictionary, chinese_dictionary, english_total_words, chinese_total_words],
                    open(args.vocab_file, "wb"))

    args.english_total_words = english_total_words
    args.chinese_total_words = chinese_total_words

    inverse_english_dictionary = {vector: key for key, vector in english_dictionary.items()}
    inverse_chinese_dictionary = {vector: key for key, vector in chinese_dictionary.items()}

    # encode the words into numbers
    train_english, train_chinese = encode(train_english, train_chinese, english_dictionary, chinese_dictionary)

    # convert the train and develop data into numpy matrices
    # batch_size * seq_length
    train_data = generate_examples(train_english, train_chinese, args.batch_size)
    develop_english, develop_chinese = encode(develop_english, develop_chinese,
                                              english_dictionary, chinese_dictionary)
    develop_data = generate_examples(develop_english, develop_chinese, args.batch_size)

    code.interact(local=locals())


if __name__ == "__main__":
    args = get_args()
    main(args)

