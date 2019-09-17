# training: feed the training data into the models, loss function, gradient descent

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import argparse

from torch.nn import Parameter
from torch.autograd import Variable
from torch import optim
from torch.nn import Module
from tqdm import tqdm
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

    # model details
    parser.add_argument('--use_cuba', type=int, default=0,
                        help='use cuda GPU or not 0|1')
    parser.add_argument('--model_file', type=str, default="model.th",
                        help='model file')
    parser.add_argument('--model', type=str, default="HingeModelCriterion",
                        help='choose the loss criterion')
    parser.add_argument('--embedding_size', type=int, default=300,
                        help='Default embedding size if embedding_file is not given')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of RNN units')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of RNN layers')

    # training details

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='Evaluation on dev set after K epochs')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer: sgd or adam (default) or rmsprop')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD')

    return parser.parse_args()


""" Modeling """

# encode English: matrix
# decode to Chinese: matrix


class encoder_decoder_model(nn.Module):
    def __init__(self, args):
        super(encoder_decoder_model, self).__init__()
        self.nhid = args.hidden_size

        self.embed_english = nn.Embedding(args.english_total_words, args.embedding_size)
        self.embed_chinese = nn.Embedding(args.chinese_total_words, args.embedding_size)

        self.encoder = nn.LSTMCell(args.embedding_size, args.hidden_size)
        self.decoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)

        self.linear = nn.Linear(self.nhid, args.chinese_total_words)
        self.linear.bias.fill_(0)
        self.linear.weight.uniform_(-0.1, 0.1)

        self.embed_english.weight.uniform_(-0.1, 0.1)
        self.embed_chinese.weight.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (Variable(weight.new(batch_size, self.nhid).zero_()),
                Variable(weight.new(batch_size, self.nhid).zero_()))

    def forward(self, english_matrix, english_matrix_mask, chinese_matrix, hidden):
        """
        :param english_matrix: B * T tensor
        :param english_matrix_mask: B * T tensor
        :param chinese_matrix: B * J tensor
        :param hidden: B * J * hidden_size vector
        :return:
        """

        english_matrix_embedded = self.embed_english(english_matrix)
        B, T, embedding_size = english_matrix_embedded.size()

        # encoder
        hiddens = []
        cells = []
        for i in range(T):
            hidden = self.encoder(english_matrix_embedded[:, i, :], hidden)
            hiddens.append(hidden[0].unsqueeze(1))
            cells.append(hidden[1].unsqueeze(1))

        hiddens = torch.cat(hiddens, 1)
        cells = torch.cat(cells, 1)

        english_matrix_lengths = english_matrix_mask.sum(1).unsqueeze(2).expand(B, 1, embedding_size)-1
        h = hiddens.gather(1, english_matrix_lengths).permute(1, 0, 2)
        c = cells.gether(1, english_matrix_lengths).permute(1, 0, 2)

        # decoder
        chinese_embedded = self.embed_chinese(chinese_matrix)
        hiddens, (h, c) = self.decoder(chinese_matrix, hx=(h, c))

        hiddens = hiddens.contiguous()
        # output layer: score of each word
        decoded = self.linear(hiddens.view(hiddens.size(0) * hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)  # (B * J) * chinese_total_words matrix

        # return B * J * chinese_total_words
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens


class language_model_criterion(nn.Module):
    def __init__(self):
        super(language_model_criterion, self).__init__()

    def forward(self, input, target, mask):
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


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


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


""" main function """


def evaluate(model, data, args, crit):
    total_develop_batches = len(data)
    correct_count = 0

    loss = 0
    total_num_words = 0

    print("total %d" % total_develop_batches)
    total_num_words = 0

    for index, (mini_batch_english, mini_batch_english_mask, mini_batch_chinese,
                mini_batch_chinese_mask) in enumerate(data):
        batch_size = mini_batch_english.shape[0]

        mini_batch_english = Variable(torch.from_numpy(mini_batch_english), volatile=True).long()
        mini_batch_english_mask = Variable(torch.from_numpy(mini_batch_english), volatile=True).long()

        hidden = model.init_hidden(batch_size)

        mini_batch_input = Variable(torch.from_numpy(mini_batch_chinese[:, :-1]), volatile=True).long()
        mini_batch_output = Variable(torch.from_numpy(mini_batch_chinese[:, 1:]), volatile=True).long()
        mini_batch_output_mask = Variable(torch.from_numpy(mini_batch_chinese_mask[:, 1:]), volatile=True)
        if args.use_cuba:
            mini_batch_english = mini_batch_english.cuda()
            mini_batch_english_mask = mini_batch_english_mask.cuda()
            mini_batch_input = mini_batch_input.cuda()
            mini_batch_output = mini_batch_output.cuda()
            mini_batch_output_mask = mini_batch_output_mask.cuda()

        mini_batch_pred, hidden = model(mini_batch_english, mini_batch_english_mask, mini_batch_input, hidden)
        num_words = torch.sum(mini_batch_output_mask).data[0]
        loss += crit(mini_batch_pred, mini_batch_output, mini_batch_output_mask).data[0] * num_words

        total_num_words += num_words

        mini_batch_pred = torch.max(mini_batch_pred.view(mini_batch_pred.size(0) *
                                                         mini_batch_pred.size(1), mini_batch_pred.size(2)), 1)[1]
        correct = (mini_batch_pred == mini_batch_output).float()

        correct_count += torch.sum(correct * mini_batch_output_mask.contiguous().
                                   view(mini_batch_output_mask.size(0) * mini_batch_output_mask(1), 1)).data[0]

    return correct_count, loss, total_num_words


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

    if os.path.isfile(args.model_file):
        model = torch.load(args.model_file)
    elif args.model == "encoder_decoder_model":
        model = encoder_decoder_model(args)

    if args.use_cuba:
        model = model.cuba()

    crit = language_model_criterion()

    learning_rate = args.learing_rate
    optimizer = getattr(optim, args.optimizer)(model.parameter(), lr=learning_rate)

    correct_count, loss, num_words = evaluate(model, develop_data, args, crit)

    loss = loss / num_words
    accuracy = correct_count / num_words

    print("develop loss %s" % loss)
    print("develop accuracy %f" % accuracy)
    print("develop total number of words %f" % num_words)

    best_accuracy = accuracy

    total_num_sentences = 0
    total_time = 0

    for epoch in range(args.num_epochs):
        np.random.shuffle(train_data)
        total_train_loss = 0
        total_num_words = 0

        for index, (mini_batch_english, mini_batch_english_mask, mini_batch_chinese,
                    mini_batch_chinese_mask) in tqdm(enumerate(train_data)):

            # convert numpy ndarray to PyTorch tensor
            # convert to PyTorch Variable
            batch_size = mini_batch_english.shape[0]
            total_num_sentences += batch_size
            hidden = model.init_hidden(batch_size)

            mini_batch_english = Variable(torch.from_numpy(mini_batch_english)).long()
            mini_batch_english_mask = Variable(torch.from_numpy(mini_batch_english_mask)).long()

            mini_batch_input = Variable(torch.from_numpy(mini_batch_chinese[:, :-1])).long()
            mini_batch_output = Variable(torch.from_numpy(mini_batch_chinese[:, 1:])).long()
            mini_batch_output_mask = Variable(torch.from_numpy(mini_batch_chinese_mask[:, 1:]))

            if args.use_cuba:
                mini_batch_english = mini_batch_english.cuda()
                mini_batch_english_mask = mini_batch_english_mask.cuda()

                mini_batch_input = mini_batch_input.cuda()
                mini_batch_output = mini_batch_output.cuda()
                mini_batch_output_mask = mini_batch_output_mask.cuda()

            mini_batch_pred, hidden = model(mini_batch_english, mini_batch_english_mask, mini_batch_input, hidden)

            # calculate loss function
            loss = crit(mini_batch_pred, mini_batch_output, mini_batch_output_mask)
            num_words = torch.sum(mini_batch_output_mask).data[0]
            total_train_loss += loss.data[0] * num_words
            total_num_words += num_words

            # update the model
            optimizer.zero_grad()  # zero the previous gradient
            loss.backword()
            optimizer.step()

    print("training loss: %f" % (total_train_loss / total_num_words))

    if (epoch + 1) % args.eval_epoch == 0:
        print("start evaluating on develop...")

        correct_count, loss, num_words = evaluate(model, develop_data, args, crit)

        loss = loss / num_words
        accuracy = correct_count / num_words

        print("develop loss %s" % loss)
        print("develop accuracy %f" % accuracy)
        print("develop total number of words %f" % num_words)

        if accuracy >= best_accuracy:
            torch.save(model, args.model_file)
            best_accuracy = accuracy
            print("model saved...")
        else:
            learning_rate *= 0.5
            optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)

        print("The best accuracy is: %f" % best_accuracy)
        print("#" * 20)

    test_english, test_chinese = load_data(args.test_file)
    args.num_test = len(test_english)
    test_english, test_chinese = encode(test_english, test_chinese, english_dictionary, chinese_dictionary)
    test_data = generate_examples(test_english, test_chinese, args.batch_size)

    correct_count, loss, num_words = evaluate(model, test_data, args, crit)
    loss = loss / num_words
    accuracy = correct_count / num_words

    print("test loss %s" % loss)
    print("test accuracy %f" % accuracy)


if __name__ == "__main__":
    args = get_args()
    main(args)
