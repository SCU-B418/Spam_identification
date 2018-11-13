# _*_ encoding:utf-8 _*_
# _*_ author: 汪嘉伟 _*_
# _*_ 2018/7/29 15:18 _*_

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, word_hidden_dim, char_embedding_dim, char_hidden_dim, batch_size, char_text, text_vocab, tag_vocab, use_gpu=True):
        super(LSTMTagger, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.batch_size = batch_size
        self.text_vocab = text_vocab
        self.tag_vocab = tag_vocab
        self.use_gpu = use_gpu
        self.char_text = char_text
        self.word_embeddings = nn.Embedding(len(text_vocab.stoi), word_embedding_dim)
        if text_vocab:
            self.word_embeddings.weight.data.copy_(text_vocab.vectors)
        if use_gpu:
            self.word_embeddings.cuda()
        self.char_embedding = nn.Embedding(len(char_text.vocab.stoi), char_embedding_dim)
        if use_gpu:
            self.char_embedding.cuda()
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, bidirectional=True)  #char_lstm is used to get the character embedding of words
        self.lstm = nn.LSTM(word_embedding_dim + char_hidden_dim*2, word_hidden_dim+char_hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear((word_hidden_dim + char_hidden_dim)*2, len(tag_vocab.stoi)-2)

        self.char_hidden = self.init_char_hidden()

    def init_char_hidden(self):
        if self.use_gpu:
            return (torch.randn(2, 1, self.char_hidden_dim).cuda(),
                torch.randn(2, 1, self.char_hidden_dim).cuda())
        return (torch.randn(2, 1, self.char_hidden_dim),
                torch.randn(2, 1, self.char_hidden_dim))

    def forward(self, batch_sentences, lengths):
        word_embeds = self.word_embeddings(batch_sentences)  # The word embedding of the sentence in batch
        tensor_char_batch_embeds = self.get_char_embeds(batch_sentences)  # Get the character embedding of each word
        char_word_concentrate_embeds = torch.cat((word_embeds, tensor_char_batch_embeds), dim=2)
        char_word_concentrate_embeds = pack_padded_sequence(char_word_concentrate_embeds, lengths, batch_first=True)
        lstm_out, _ = self.lstm(char_word_concentrate_embeds)
        tag_space = self.hidden2tag(lstm_out[0])
        tag_scores = tag_space.view(-1, len(self.tag_vocab.stoi)-2)
        print('tag_scores:', tag_scores)
        return tag_scores

    def get_char_embeds(self, batch_sentences):  # Get the character embedding of each word in the batch
        words_list = [] # All the words in each sentence in batch form a list and add the list to the words_list
        char_batch_embeds = []
        for ele in batch_sentences:
            words_list_ele = []  # Add words from sentences to words_list_ele
            for e in ele:
                words_list_ele.append(self.text_vocab.itos[e.item()])
            words_list.append(words_list_ele)  # Add a list of words in each sentence as an element to the words_list

        for words in words_list:
            #print(words)
            char_sentence_embeds = []  # This list stores the character embedding of all words in the sentence
            for word in words:
                char_list = list(word)
                char_batch = self.char_text.numericalize(char_list)  # Get the tensor after splitting the word into letters, consisting of the index of each letter in the alphabetic vocabulary
                char_batch = char_batch.view(-1, len(char_list))
                if self.use_gpu:
                    char_batch = char_batch.cuda()
                char_embedding = self.char_embedding(char_batch)
                output_c, (hn, cn) = self.char_lstm(char_embedding, self.char_hidden)
                char_embed = hn.view(-1)  # Get the character embedding of the word
                char_embed_list = char_embed.tolist()
                char_sentence_embeds.append(char_embed_list)
            char_batch_embeds.append(char_sentence_embeds)


        #print(char_batch_embeds)
        tensor_char_batch_embeds = torch.tensor(char_batch_embeds)
        #print(tensor_char_batch_embeds)
        if self.use_gpu:
            tensor_char_batch_embeds = tensor_char_batch_embeds.cuda()
        return tensor_char_batch_embeds  # return the character embedding of the batch

 