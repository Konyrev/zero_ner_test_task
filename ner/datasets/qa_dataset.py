from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Dict, List, Tuple

from ner.qa_types import QASpan, QAInstance


class QADataset(Dataset):
    def __init__(self,
                 sentences: List[List[str]],
                 spans: List[List[QASpan]],
                 prompt_mapper: Dict[str, str]
    ):
        self.data = QADataset.__prepare_dataset__(sentences, spans, prompt_mapper)
        self.prompt_mapper = prompt_mapper

    @staticmethod
    def __prepare_dataset__(
            sentences: List[List[str]],
            spans: List[List[QASpan]],
            prompt_mapper: Dict[str, str]
    ) -> List[QAInstance]:
        dataset = []
        for sentence, labels in zip(sentences, spans):
            for label_tag, label_name in prompt_mapper.items():
                question_prompt = f"What is the {label_name}?"

                answer_list = []
                for span in labels:
                    if span.label == label_tag:
                        answer_list.append(span)

                if len(answer_list) == 0:
                    empty_span = QASpan(
                        token="",
                        label="O",
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    instance = QAInstance(
                        context=sentence,
                        question=question_prompt,
                        answer=empty_span,
                    )
                    dataset.append(instance)
                else:
                    for answer in answer_list:
                        instance = QAInstance(
                            context=sentence,
                            question=question_prompt,
                            answer=answer,
                        )
                        dataset.append(instance)

        return dataset

    def __getitem__(self, index) -> QAInstance:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def char_to_token_boudaries(
        offset_mapping: List[List[int]],
        start_end_context_char_pos: List[int]
) -> Tuple[int, int]:
    start_context_char_pos, end_context_char_pos = start_end_context_char_pos

    special_tokens_count = 0
    res_start_token_pos = 0
    res_end_token_pos = 0
    for i, token_boudaries in enumerate(offset_mapping):

        # Встретили специальный токен BERT-а
        if token_boudaries == [0, 0]:
            special_tokens_count += 1
            continue

        if special_tokens_count == 2:
            start_token_pos, end_token_pos = token_boudaries

            if start_token_pos == start_context_char_pos:
                res_start_token_pos = i

            # Дошли до конца предложения (включительно)
            if end_token_pos == end_context_char_pos:
                res_end_token_pos = i
                break

    return res_start_token_pos, res_end_token_pos


def collate_qa_batch(batch: List[QAInstance], tokenizer: Callable) -> BatchEncoding:
    context_list = []
    question_list = []
    start_end_context_char_pos_list = []

    for instance in batch:
        context_list.append(instance.context)
        question_list.append(instance.question)
        start_end_context_char_pos_list.append(
            [
                instance.answer.start_context_char_pos,
                instance.answer.end_context_char_pos,
            ]
        )

    tokenized_batch = tokenizer(
        question_list, [' '.join(c) for c in context_list]
    )

    offset_mapping_batch = tokenized_batch["offset_mapping"].numpy().tolist()

    # Символьные граница конвертируем в границы токенов
    start_token_pos_list, end_token_pos_list = [], []
    for offset_mapping, start_end_context_char_pos in zip(
        offset_mapping_batch, start_end_context_char_pos_list
    ):
        if start_end_context_char_pos == [0, 0]:
            start_token_pos_list.append(0)
            end_token_pos_list.append(0)
        else:
            (
                start_token_pos,
                end_token_pos,
            ) = char_to_token_boudaries(
                offset_mapping=offset_mapping,
                start_end_context_char_pos=start_end_context_char_pos,
            )
            start_token_pos_list.append(start_token_pos)
            end_token_pos_list.append(end_token_pos)

    tokenized_batch["start_positions"] = LongTensor(start_token_pos_list)
    tokenized_batch["end_positions"] = LongTensor(end_token_pos_list)

    # Сохраним еще исходный батч
    tokenized_batch["instances"] = batch

    return tokenized_batch


def create_qa_loader(
        dataset: QADataset,
        tokenizer: Callable,
        shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, collate_fn=lambda d: collate_qa_batch(d, tokenizer), batch_size=50, shuffle=shuffle)