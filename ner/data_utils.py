from collections import Counter
from typing import Callable, Dict, Iterator, List, Tuple

from ner.qa_types import QASpan


def load_sentences(path: str, bio_tagging: bool=False) -> List[List[Tuple[str, str]]]:
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                # Обнуляем предложение
                if len(texts[-1]) > 0:
                    texts.append([])
            else:
                if 'DOCSTART' in line.rstrip():
                    texts.append([])
                else:
                    splitted = line.split(' ')
                    tag = splitted[-1]
                    if not bio_tagging:
                        if '-' in tag:
                            tag = tag.split('-')[-1]
                    texts[-1].append((splitted[0], tag))
    return texts


def load_qa_sentences(path: str):
    sentences = []
    spans = []
    with open(path, 'r', encoding='utf-8') as f:
        current_char_pos = 0
        for line in f:
            line = line.rstrip()
            if not line:
                if len(sentences[-1]) > 0:
                    sentences.append([])
                    spans.append([])
                    current_char_pos = 0
            else:
                if 'DOCSTART' in line.rstrip():
                    sentences.append([])
                    spans.append([])
                    current_char_pos = 0
                else:
                    splitted = line.split(' ')
                    token = splitted[0]
                    sentences[-1].append(token)

                    tag = splitted[-1]
                    if tag.startswith('I-'):
                        spans[-1].append(
                            QASpan(
                                token=f"{spans[-1][-1].token} {token}",
                                label=spans[-1][-1].label,
                                start_context_char_pos=spans[-1][-1].start_context_char_pos,
                                end_context_char_pos=current_char_pos + len(token),
                            )
                        )
                    elif tag.startswith('B-'):
                        spans[-1].append(
                            QASpan(
                                token=token,
                                label=tag.split('-')[-1],
                                start_context_char_pos=current_char_pos,
                                end_context_char_pos=current_char_pos + len(token),
                            )
                        )
                    current_char_pos += len(token) + 1
    return sentences, spans


def iter_over_sentences(
        sentences: List[List[Tuple[str, str]]],
        is_lower: bool = True
) -> Iterator[str]:
    for sentence in sentences:
        for word in sentence:
            yield word[0].lower() if is_lower else word[0]


def iter_over_words(
        sentences: List[List[Tuple[str, str]]],
        lower: bool = False
) -> Iterator[str]:
    for sentence in sentences:
        for word in sentence:
            for char in word[0].lower() if lower else word[0]:
                yield char


def iter_over_tags(sentences: List[List[Tuple[str, str]]]) -> Iterator[str]:
    for sentence in sentences:
        for word in sentence:
            yield word[-1]


def create_mapping(sentences: List[List[Tuple[str, str]]],
                   iterator: Callable[[List[List[Tuple[str, str]]]], Iterator[str]],
                   use_unknown: bool = True
) -> Tuple[Dict[str, int], Dict[int, str]]:
    freqs = sorted(list(Counter(iterator(sentences)).items()), key=lambda el: (el[1], el[0]), reverse=True)
    element_to_id = {element[0]: i for i, element in enumerate(freqs)}
    if use_unknown:
        element_to_id['<UNK>'] = len(element_to_id)

    id_to_element = {id: element for element, id in element_to_id.items()}
    return element_to_id, id_to_element


