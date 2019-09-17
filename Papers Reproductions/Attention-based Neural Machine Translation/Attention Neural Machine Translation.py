# Attention-based Neural Machine Translation
# load english and chinese data
# we use word tokenizer in nltk to separate English words, all of them are lowercase
# we use single Chinese characters as basic unit

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
    num_example = 0

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


""" Encoder Part """


# the task of encode module is input words into embedding layer and GRU layer,
# and convert them into some hidden states as context vectors afterwards
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, encode_hidden_size, decode_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.RNN = nn.GRU(embed_size, encode_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.function = nn.Linear(encode_hidden_size * 2, decode_hidden_size)

    def forward(self, text, lengths):
        sorted_len, sorted_index = lengths.sort(0, descending=True)
        text_sorted = text[sorted_index.long()]
        embedded = self.dropout(self.embed(text_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            sorted_len.long().cpu.data.numpy(), batch_first=True)
        packed_output, hide = self.RNN(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        _, original_index = sorted_index.sort(0, descending=False)
        output = output[original_index.long()].contiguous()
        hide = hide[:, original_index.long()].contiguous()

        hide = torch.cat([hide[-2], hide[-1]], dim=1)
        hide = torch.tanh(self.function(hide)).unsqueeze(0)

        return output, hide


""" Loung Attention model """


# according to the Attention-based model to implement
# according to context vectors and output hidden states, calculate the outputs
class Attention(nn.Module):
    def __init__(self, encode_hidden_size, decode_hidden_size):
        super(Attention, self).__init__()

        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size

        self.linear_input = nn.Linear(encode_hidden_size * 2, decode_hidden_size, bias=False)
        self.linear_output = nn.Linear(encode_hidden_size * 2, decode_hidden_size, decode_hidden_size)

    def forward(self, output, context, mask):
        """
        :param output: batch_size, output_len, decode_hidden_size
        :param context: batch_size, context_len, encode_hidden_size
        :param mask:
        :return:
        """
        batches_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_input = self.linear_input(context.view(batches_size*input_len, -1)).\
            view(batches_size, input_len - 1)  # batch_size, output_len, decode_hidden_size

        attend = torch.bmm(output, context_input.transpose(1, 2))  # batch_size, output_len, context_len
        attend.masked_fill(mask, -1e6)
        attend = func.softmax(attend, dim=2)  # batch_size, output_len, context_len

        context = torch.bmm(attend, context)  # batch_size, output_len, encode_hidden_size
        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batches_size * output_len, -1)
        output = torch.tanh(self.linear_output(output))
        output = output.view(batches_size, output_len, -1)

        return output, attend


""" Decoder Part """


# decoder will according to the context of translated sentences and context vectors to decide the next output word
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, encode_hidden_size, decode_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encode_hidden_size, decode_hidden_size)
        self.RNN = nn.GRU(embed_size, encode_hidden_size, batch_first=True)
        self.output = nn.Linear(decode_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, english_len, chinese_len):
        device = english_len.device
        max_english_len = english_len.max()
        max_chinese_len = chinese_len.max()

        english_mask = torch.arange(max_english_len, device=english_len.device)[None, :] < english_len[:, None]
        chinese_mask = torch.arange(max_chinese_len, device=chinese_len.device)[None, :] < chinese_len[:, None]
        mask = (1 - english_mask[:, :, None] * chinese_mask[:, None, :]).byte()

        return mask

    def forward(self, english_context_vector, english_context_vector_lengths, chinese, chinese_lengths, hide):
        sorted_len, sorted_index = chinese_lengths.sort(0, descending=True)
        chinese_sorted = chinese[sorted_index.long()]
        hide = hide[:, sorted_index.long()]

        chinese_sorted = self.dropout(self.embed(chinese_sorted))  # batch_size, output_len, embed_size

        packed_sequence = nn.utils.rnn.pack_padded_sequence(chinese_sorted,
                                                            sorted_len.long().cpu.data.numpy(), batch_first=True)
        output, hide = self.RNN(packed_sequence, hide)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        _, original_index = sorted_index(0, descending=False)
        output_sequence = unpacked[original_index.long()].contiguous()
        hide = hide[:, original_index.long()].contiguous()

        mask = self.create_mask(chinese_lengths, english_context_vector)

        output, attend = self.attention(output_sequence, english_context_vector, mask)
        output = func.log_softmax(self.out(output), -1)

        return output, hide, attend


""" Sequence to Sequence model """


# building the "Sequence to Sequence (Seq2Seq)" model and put encoder, attention, decoder together
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, english, english_lengths, chinese, chinese_lengths):
        encoder_output, hide = self.encoder(english, english_lengths)
        output, hide, attend = self.decoder(english_context_vector=encoder_output,
                                            english_context_vector_length=english_lengths,
                                            chinese=chinese, chinese_lengths=chinese_lengths, hide=hide)

        return output, attend

    def translate(self, english, english_lengths, chinese, max_length=100):
        encoder_output, hide = self.encoder(english, english_lengths)
        predictions = []
        batches_size = english.shape[0]
        attends = []

        for i in range(max_length):
            output, hide, attend = self.decoder(english_context_vector=encoder_output,
                                                english_context_vector_length=english_lengths,
                                                chinese=chinese,
                                                chinese_lengths=torch.ones(batches_size).long().to(chinese.device),
                                                hide=hide)
            chinese = output.max(2)[1].view(batches_size, 1)
            predictions.append(chinese)
            attends.append(attend)

        return torch.cat(predictions, 1), torch.cat(attends, 1)


""" training data part """


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

english_vocab_size = len(english_dictionary)
chinese_vocab_size = len(chinese_dictionary)
embed_size = hidden_size = 100
dropout = 0.2

encoder = Encoder(vocab_size=english_vocab_size, embed_size=embed_size,
                  encode_hidden_size=hidden_size, decode_hidden_size=hidden_size,
                  dropout=dropout)

decoder = Decoder(vocab_size=chinese_vocab_size, embed_size=embed_size,
                  encode_hidden_size=hidden_size, decode_hidden_size=hidden_size,
                  dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model(device)
crit = LanguageModelCriterion()(device)
optimizer = torch.optim.Adam(model.parameter(), lr=0.001)


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


train(model, train_data, num_epochs=30)


def translate_develop(i):
    model.eval()

    english_sent = " ".join([inverse_english_dictionary[word] for word in develop_english[i]])
    print(english_sent)
    chinese_sent = " ".join([inverse_chinese_dictionary[word] for word in develop_chinese[i]])
    print(chinese_sent)

    sent = nltk.word_tokenize(english_sent.lower())
    bos = torch.tensor([[chinese_dictionary["BOS"]]]).long().to(device)
    mini_batch_english = torch.tensor([[english_dictionary.get(word, 0) for word in sent]]).long().to(device)
    mini_batch_english_length = torch.tensor([len(sent)]).long().to(device)

    translation, attention = model.translate(mini_batch_english, mini_batch_english_length, bos)
    translation = [inverse_chinese_dictionary[i] for ii in translation.data.cpu().numpy().reshape(-1)]

    translations = []
    for word in translation:
        if word != "EOS":
            translations.append(word)
        else:
            break
    print(" ".join(translations))


for i in range(100, 120):
    translate_develop()
    print()

