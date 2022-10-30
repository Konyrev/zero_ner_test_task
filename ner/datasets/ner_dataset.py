import numpy as np
import torch

from torch import LongTensor, Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple


class NERDataset(Dataset):
    def __init__(self,
                 data: List[List[Tuple[str, str]]],
                 word_to_id: Dict[str, int],
                 char_to_id: Dict[str, int],
                 raw_char_to_id: Dict[str, int],
                 tag_to_id: Dict[str, int]
    ):
        self.data = NERDataset.__prepare_dataset__(
            data,
            word_to_id,
            char_to_id,
            raw_char_to_id,
            tag_to_id
        )
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.raw_char_to_id = raw_char_to_id
        self.tag_to_id = tag_to_id

    @staticmethod
    def __get_word_features__(word: str):
        return np.array([float(word.isupper()), float(word.istitle()), float(word.isdigit())])

    @staticmethod
    def __prepare_dataset__(
            sentences: List[List[Tuple[str, str]]],
            word_to_id: Dict[str, int],
            char_to_id: Dict[str, int],
            raw_char_to_id: Dict[str, int],
            tag_to_id: Dict[str, int]
    ) -> List[Dict]:
        data = []
        for sentence in tqdm(sentences):
            if len(sentence) == 0:
                continue

            str_words = [word[0].lower() for word in sentence]
            words = [word_to_id[word if word in word_to_id else '<UNK>']
                     for word in str_words]
            words_features = [
                NERDataset.__get_word_features__(word[0]) for word in sentence
            ]

            chars = [[char_to_id[char] for char in word if char in char_to_id]
                     for word in str_words]
            raw_chars = [[raw_char_to_id[char] for char in word[0] if char in raw_char_to_id]
                         for word in sentence]
            tags = [tag_to_id[word[-1]] for word in sentence]
            data.append({
                'str_words': str_words,
                'words': words,
                'words_features': words_features,
                'chars': chars,
                'raw_chars': raw_chars,
                'tags': tags,
            })
        return data

    def __get_chars_ids__(chars: List[List[int]]) -> LongTensor:
        chars_length = [len(c) for c in chars]
        char_maxl = max(chars_length) if chars_length else 0
        chars_mask = np.zeros((len(chars_length), char_maxl), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c
        return LongTensor(chars_mask)

    def __getitem__(self, index: int) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        # Готовим признаки преложений: word_ids + word features
        sentence_in = self.data[index]['words']
        sentence_in = LongTensor(sentence_in)
        sentence_features = LongTensor(self.data[index]['words_features'])

        # Готовим символьные признаки chars2_mask - id lowercased символов,
        # raw_chars2_mask - id исходных символов
        chars2_mask = NERDataset.__get_chars_ids__(self.data[index]['chars'])
        raw_chars2_mask = NERDataset.__get_chars_ids__(self.data[index]['raw_chars'])

        tags = self.data[index]['tags']
        targets = LongTensor(tags)

        return sentence_in, sentence_features, chars2_mask, raw_chars2_mask, targets

    def __len__(self) -> int:
        return len(self.data)


def stack_batch_tensors(
        data: List[Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]],
        use_gpu: bool=True
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    sentences_in = torch.cat(tuple([d[0] for d in data]))
    sentences_features = torch.cat(tuple([d[1] for d in data]))

    # Выравниваем тензоры с символами, добивая нулями слева
    max_char2_length = max([d[2].shape[1] for d in data])
    chars2 = torch.cat([torch.nn.functional.pad(d[2], (0, max_char2_length - d[2].shape[1])) for d in data])

    max_raw_char2_length = max([d[3].shape[1] for d in data])
    raw_chars2 = torch.cat([torch.nn.functional.pad(d[3], (0, max_raw_char2_length - d[3].shape[1])) for d in data])

    tags = torch.cat(tuple([d[4] for d in data]))
    if use_gpu:
        return sentences_in.cuda(), sentences_features.cuda(), chars2.cuda(), raw_chars2.cuda(), tags.cuda()
    else:
        return sentences_in, sentences_features, chars2, raw_chars2, tags


def create_loader(dataset: NERDataset, shuffle: bool = True, use_gpu: bool = False) -> DataLoader:
    return DataLoader(dataset, collate_fn=lambda d: stack_batch_tensors(d, use_gpu), batch_size=10, shuffle=shuffle)