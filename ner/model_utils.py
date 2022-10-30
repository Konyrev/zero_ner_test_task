import numpy as np

from typing import Dict


def load_embeddings(path: str, word_to_id: Dict[str, int], words_dimention=50):
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), words_dimention))

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            splitted = line.strip().split()
            word = splitted[0].lower()
            if word in word_to_id:
                word_embeds[word_to_id[word]] = np.array(splitted[1:])

    return word_embeds
