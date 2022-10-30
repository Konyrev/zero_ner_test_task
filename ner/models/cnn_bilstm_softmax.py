import numpy as np

from torch import cat, FloatTensor, Tensor
from torch.nn import Embedding, Conv2d, Dropout, Linear, LSTM, Module, Parameter
from torch.nn.functional import cross_entropy, max_pool2d
from torch.nn.init import uniform
from typing import Dict

# Функция инициализации CNN
def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    uniform(input_embedding, -bias, bias)

# Функция инициализации Linear слоя
def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_layer_value_(input_lstm, index, value='weight', tag='i', reverse=False):
    weight = eval(f'input_lstm.{value}_{tag}h_l{index}' + str('_reverse' if reverse else ''))
    sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    uniform(weight, -sampling_range, sampling_range)


def init_layer_value(input_lstm, index, value='weight'):
    init_layer_value_(input_lstm, index, value, 'i')
    init_layer_value_(input_lstm, index, value, 'h')

# Функция инициализации LSTM слоя
def init_lstm(input_lstm):
    for index in range(input_lstm.num_layers):
        init_layer_value(input_lstm, index, 'weight')

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            bias.data.zero_()

            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


class CNN_BiLSTM_Softmax(Module):
    def __init__(self,
                 vocab_size: int,
                 tag_to_ix: Dict[str, int],
                 embedding_dim: int,
                 hidden_dim: int,
                 char_to_ix: Dict[str, int] = None,
                 raw_char_to_ix: Dict[str, int] = None,
                 pre_word_embeds=None,
                 char_out_dimension: int = 25,
                 char_embedding_dim: int = 25
                 ):
        '''
                vocab_size: int - размер словаря
                tag_to_ix: Dict[str, int] - словарь маппинга тегов
                embedding_dim: int - размерность эмбеддингов слов
                hidden_dim: int - размер скрытого LSTM слоя
                char_to_ix: Dict[str, int] -  словарь маппинга lowercased символов
                raw_char_to_ix: Dict[str, int] -  словарь маппинга символов
                pre_word_embeds - маппинг эмбеддинг слова -> индекс слова
                char_out_dimension: int - размерность выходного слоя символьной CNN
                char_embedding_dim: int - размерность символьных эмбеддингов
        '''

        super(CNN_BiLSTM_Softmax, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            self.char_embeds = Embedding(len(char_to_ix), self.char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            self.char_cnn3 = Conv2d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(3, char_embedding_dim),
                padding=(2, 0)
            )

            self.raw_char_embedding_dim = char_embedding_dim
            self.raw_char_embeds = Embedding(len(raw_char_to_ix), self.char_embedding_dim)
            init_embedding(self.raw_char_embeds.weight)
            self.raw_char_cnn3 = Conv2d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(3, char_embedding_dim),
                padding=(2, 0)
            )

        self.word_embeds = Embedding(vocab_size + 1, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = Parameter(FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = Dropout(0.68)

        self.lstm = LSTM(
            embedding_dim + self.out_channels * 2 + 3,
            hidden_dim,
            bidirectional=True,
            num_layers=1
        )
        init_lstm(self.lstm)

        self.hidden2tag = Linear(hidden_dim * 2, self.tagset_size)
        init_linear(self.hidden2tag)

    def __predict__(self,
                           sentences: Tensor,
                           sentences_features: Tensor,
                           chars2: Tensor,
                           raw_chars2: Tensor
    ) -> Tensor:

        chars_embeds = self.char_embeds(chars2)
        chars_embeds = chars_embeds.unsqueeze(1)
        chars_cnn_out3 = self.char_cnn3(chars_embeds)
        chars_embeds = max_pool2d(
            chars_cnn_out3,
            kernel_size=(chars_cnn_out3.size(2), 1)
        ).view(
            chars_cnn_out3.size(0),
            self.out_channels
        )

        raw_chars_embeds = self.raw_char_embeds(raw_chars2)
        raw_chars_embeds = raw_chars_embeds.unsqueeze(1)
        raw_chars_cnn_out3 = self.raw_char_cnn3(raw_chars_embeds)
        raw_chars_embeds = max_pool2d(
            raw_chars_cnn_out3,
            kernel_size=(raw_chars_cnn_out3.size(2), 1)
        ).view(
            raw_chars_cnn_out3.size(0),
            self.out_channels
        )

        embeds = self.word_embeds(sentences)
        embeds = cat((embeds, sentences_features, chars_embeds, raw_chars_embeds), 1)
        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        # Приведение к размерности не в памяти, а виртуально
        lstm_out = lstm_out.view(
            len(sentences),
            self.hidden_dim * 2
        )
        lstm_out = self.dropout(lstm_out)
        logits = self.hidden2tag(lstm_out)
        return logits

    def forward(self, data):
        sentences, sentences_features, chars2_list, raw_chars2_list, tags = data
        logits = self.__predict__(sentences, sentences_features, chars2_list, raw_chars2_list)
        scores = cross_entropy(logits, tags)
        return scores, logits