from datasets import Dataset
from transformers import BatchEncoding
from typing import Dict, List, Tuple


# Выравниваем подтокены и таргет после токенизации BERT-ом
def align_labels_with_tokens(
        labels: List[str],
        word_ids: List[int],
        tag_to_id: Dict[str, int]
) -> List[int]:
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Значит это новое слово
            current_word = word_id
            label = -100 if word_id is None else tag_to_id[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Специальный токен для начала предложения и для padding-а
            new_labels.append(-100)
        else:
            # Такое же слово, что и предыдущий токен
            label = labels[word_id]
            # Если мы в BIO кодировке, то меняем B-xxx на I-xxx
            if 'B-' in label:
                label = label.replace('B-', 'I-')
            new_labels.append(tag_to_id[label])

    return new_labels

# Приводим датасет в соответствии с токенизацией BERT-а
def tokenize_and_align_labels(
        sentences: List[List[Tuple[str, str]]],
        tokenizer,
        tag_to_id: Dict[str, int]
) -> BatchEncoding:
    tokenized_inputs = tokenizer(
        [[word for word, _ in sentence] for sentence in sentences if len(sentence) > 0],
        truncation=True,
        is_split_into_words=True
    )
    all_labels = [[label for _, label in sentence] for sentence in sentences if len(sentence) > 0]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, tag_to_id))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def create_bert_dataset(
        sentences: List[List[Tuple[str, str]]],
        tokenizer,
        tag_to_id: Dict[str, int]
) -> Dataset:
    return Dataset.from_dict(tokenize_and_align_labels(sentences, tokenizer, tag_to_id))