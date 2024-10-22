# this is a plain translation model without attention, used to compare with the Attention-based one

import nltk
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as func

from collections import Counter


""" Data Pre Processing Part"""


def load_data(in_file):
    chinese = []
    english = []

    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")

            english.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            # split chinese sentences into characters
            chinese.append(["BOS"] + [c for c in line[1]] + ["EOS"])

    return english, chinese


train_file = "data/train.txt"
develop_file = "data/develop.text"
train_english, train_chinese = load_data(train_file)
develop_english, develop_chinese = load_data(develop_file)

# building the word dictionary
UNK_IDX = 0
PAD_IDX = 1


def build_dictionary(sentences, max_words=10000):
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    most_word = word_count.most_common(max_words)
    total_words = len(most_word) + 2
    word_dictionary = {word[0]: index+2 for index, word in enumerate(most_word)}
    word_dictionary["UNK"] = UNK_IDX
    word_dictionary["PAD"] = PAD_IDX

    return word_dictionary, total_words


english_dictionary, english_total_words = build_dictionary(train_english)
chinese_dictionary, chinese_total_words = build_dictionary(train_chinese)
inverse_english_dictionary = {vector: key for key, vector in english_dictionary.items()}
inverse_chinese_dictionary = {vector: key for key, vector in chinese_dictionary.items()}


# transfer all words into numbers
def encode(english_sentences, chinese_sentences, english_dictionary, chinese_dictionary, sort_by_len=True):
    """ Encode the sequences """
    length = len(english_sentences)
    output_english_sentences = [[english_dictionary.get(word, 0) for word in sentence]
                                for sentence in english_sentences]
    output_chinese_sentences = [[chinese_dictionary.get(word, 0) for word in sentence]
                                for sentence in chinese_sentences]

    # sort sentences by english lengths
    def len_argsort(sequence):
        return sorted(range(len(sequence)), key=lambda x: len(sequence[x]))

    if sort_by_len:
        sorted_index = len_argsort(output_english_sentences)
        output_english_sentences = [output_english_sentences[i] for i in sorted_index]
        output_chinese_sentences = [output_chinese_sentences[i] for i in sorted_index]

    return output_english_sentences, output_chinese_sentences


train_english, train_chinese = encode(train_english, train_chinese, english_dictionary, chinese_dictionary)
develop_english, develop_chinese = encode(develop_english, develop_chinese, english_dictionary, chinese_dictionary)


# separate all sentences into batches
def get_mini_batches(n, mini_batch_size, shuffle=True):
    index_list = np.arange(0, n, mini_batch_size)
    if shuffle:
        np.random.shuffle(index_list)
    mini_batches = []
    for index in index_list:
        mini_batches.append(np.arange(index, min(index + mini_batch_size, n)))

    return mini_batches


def prepare_data(sequences):
    lengths = [len(sequence) for sequence in sequences]
    n_samples = len(sequences)
    max_len = np.max(lengths)

    padding = np.zeros((n_samples, max_len)).astype('int32')
    padding_lengths = np.array(lengths).astype("int32")
    for index, sequence in enumerate(sequences):
        padding[index, :lengths[index]] = sequence

    return padding, padding_lengths


def generate_examples(english_sentences, chinese_sentences, batches_size):
    mini_batches = get_mini_batches(len(english_sentences), batches_size)
    all_convert = []

    for mini_batch in mini_batches:
        mini_batch_english_sentences = [english_sentences[t] for t in mini_batch]
        mini_batch_chinese_sentences = [chinese_sentences[t] for t in mini_batch]
        mini_batch_english, mini_batch_english_len = prepare_data(mini_batch_english_sentences)
        mini_batch_chinese, mini_batch_chinese_len = prepare_data(mini_batch_chinese_sentences)
        all_convert.append((mini_batch_english, mini_batch_english_len, mini_batch_chinese, mini_batch_chinese_len))

    return all_convert


batch_size = 128
train_data = generate_examples(train_english, train_chinese, batch_size)
random.shuffle(train_data)
develop_data = generate_examples(develop_english, develop_chinese, batch_size)


# plain encoder and decoder model
class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.RNN = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, english_sentences, lengths):
        sorted_len, sorted_index = lengths.sort(0, descending=True)
        english_sentences_sorted = english_sentences[sorted_index.long()]
        embedded = self.dropout(self.embed(english_sentences_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_output, hide = self.rnn(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        _, original_index = sorted_index.sort(0, descending=False)

        output = output[original_index.long()].contiguous()
        hide = hide[:, original_index.long()].contiguous()

        return output, hide[[-1]]


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.RNN = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chinese_sentences, chinese_sentences_lengths, hide):
        sorted_len, sorted_index = chinese_sentences_lengths.sort(0, descending=True)
        chinese_sentences_sorted = chinese_sentences[sorted_index.long()]
        hide = hide[:, sorted_index.long()]

        # batch_size, output_length, embed_size
        chinese_sentences_sorted = self.dropout(self.embed(chinese_sentences_sorted))
        packed_sequence = nn.utils.rnn.pack_padded_sequence(chinese_sentences_sorted,
                                                            sorted_len.long().cpu.data.numpy(), batch_first=True)
        output, hide = self.rnn(packed_sequence, hide)

        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        _, original_index = sorted_index.sort(0, descending=False)
        output_sequence = unpacked[original_index.long()].contiguous()
        # print(output_sequence.shape)

        hide = hide[:, original_index.long()].contiguous()
        outcome = func.log_softmax(self.output(output_sequence), -1)

        return outcome, hide


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, english_sentences, english_sentences_lengths, chinese_sentences, chinese_sentences_lengths):
        encoder_output, hide = self.encoder(english_sentences, english_sentences_lengths)
        output, hide = self.decoder(chinese_sentences=chinese_sentences,
                                    chinese_sentences_lengths=chinese_sentences_lengths, hide=hide)

        return output, None

    def translate(self, english_sentences, english_sentences_lengths, chinese_sentences, max_length=10):
        encoder_output, hide = self.encoder(english_sentences, english_sentences_lengths)
        predictions = []
        batch_size = english_sentences.shape[0]
        attends = []

        for i in range(max_length):
            output, hide = self.decoder(chinese_sentences=chinese_sentences,
                                        chinese_sentences_lengths=torch.ones(batch_size).
                                        long().to(chinese_sentences.device), hide=hide)
            chinese_sentences = output.max(2)[1].view(batch_size, 1)
            predictions.append(chinese_sentences)

        return torch.cat(predictions, 1), None


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input_list, target, mask):
        input_list = input_list.contiguous().view(-1, input_list.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)

        output = -input_list.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.1

english_vocab_size = len(english_dictionary)
chinese_vocab_size = len(chinese_dictionary)
embed_size = hidden_size = 100
encoder = PlainEncoder(vocab_size=english_vocab_size, hidden_size=hidden_size, dropout=dropout)
decoder = PlainDecoder(vocab_size=chinese_vocab_size, hidden_size=hidden_size, dropout=dropout)

model = PlainSeq2Seq(encoder, decoder)
model = model(device)
crit = LanguageModelCriterion()(device)
optimizer = torch.optim.Adam(model.parameters())


def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0

    with torch.no_grad():
        for it, (mini_batch_english, mini_batch_english_lengths,
                 mini_batch_chinese, mini_batch_chinese_lengths) in enumerate(data):
            mini_batch_english = torch.from_numpy(mini_batch_english).long().to(device)
            mini_batch_english_lengths = torch.from_numpy(mini_batch_english).long().to(device)

            mini_batch_input = torch.from_numpy(mini_batch_chinese[:, :-1]).long().to(device)
            mini_batch_output = torch.from_numpy(mini_batch_chinese[:, 1:]).long().to(device)

            mini_batch_chinese_lengths = torch.from_numpy(mini_batch_chinese_lengths - 1).long().to(device)
            mini_batch_chinese_lengths[mini_batch_chinese_lengths <= 0] = 1

            mini_prediction, attend = model(mini_batch_english, mini_batch_english_lengths,
                                            mini_batch_input, mini_batch_chinese_lengths)

            mini_batch_output_mask = torch.arange(mini_batch_chinese_lengths.max().item(),
                                                  device=device)[None, :] < mini_batch_chinese_lengths[:, None]
            mini_batch_output_mask = mini_batch_output_mask.float()
            loss = crit(mini_prediction, mini_batch_output, mini_batch_output_mask)

            num_words = torch.sum(mini_batch_chinese_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

    print("evaluation loss", total_loss / total_num_words)


def train(model, data, num_epochs=30):
    for epoch in range(num_epochs):
        total_num_words = total_loss = 0
        model.train()

        for it, (mini_batch_english, mini_batch_english_lengths,
                 mini_batch_chinese, mini_batch_chinese_lengths) in enumerate(data):
            mini_batch_english = torch.from_numpy(mini_batch_english).long().to(device)
            mini_batch_english_lengths = torch.from_numpy(mini_batch_english).long().to(device)

            mini_batch_input = torch.from_numpy(mini_batch_chinese[:, :-1]).long().to(device)
            mini_batch_output = torch.from_numpy(mini_batch_chinese[:, 1:]).long().to(device)

            mini_batch_chinese_lengths = torch.from_numpy(mini_batch_chinese_lengths - 1).long().to(device)
            mini_batch_chinese_lengths[mini_batch_chinese_lengths <= 0] = 1

            mini_prediction, attend = model(mini_batch_english, mini_batch_english_lengths,
                                            mini_batch_input, mini_batch_chinese_lengths)

            mini_batch_output_mask = torch.arange(mini_batch_chinese_lengths.max().item(),
                                                  device=device)[None, :] < mini_batch_chinese_lengths[:, None]
            mini_batch_output_mask = mini_batch_output_mask.float()
            loss = crit(mini_prediction, mini_batch_output, mini_batch_output_mask)

            num_words = torch.sum(mini_batch_chinese_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameter(), 5)
            optimizer.step()

            if it % 100 == 0:
                print("epoch", epoch, "iteration", it, "loss", loss.item())

        print("epoch", epoch, "training loss", total_loss / total_num_words)

        if epoch % 5 == 0:
            print("evaluating on develop...")
            evaluate(model, develop_data)


train(model, train_data, 30)

