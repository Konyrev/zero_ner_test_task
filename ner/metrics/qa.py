import numpy as np

from torch import no_grad, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List
from tqdm import tqdm

from ner.metrics.ner import calculate_metrics as calculate_token_metrics
from ner.qa_types import QASpan


def get_top_valid_spans(
    context_list: List[str],
    question_list: List[str],
    prompt_mapper: Dict[str, str],
    inputs: BatchEncoding,
    outputs: BatchEncoding,
    offset_mapping_batch: Tensor,
    n_best_size: int = 1,
    max_answer_length: int = 100,
) -> List[List[QASpan]]:
    """
    Из output-а QuestionAnswering модели получаем QASpan-ы
    https://huggingface.co/docs/transformers/tasks/question_answering
    Args:
        context_list: List[str] - контексты (предложенния)
        question_list: List[str] - тексты вопросов
        prompt_mapper: Dict[str, str] - маппинг тегов на текст
        inputs: BatchEncoding - то, что пришло на вход модели
        outputs: BatchEncoding - результата работы модели
        offset_mapping_batch: Tensor - маппинг спана на позицию в тексте
        n_best_size: int - сколько возращать ответов на один вопрос
        max_answer_length: int - максимальная длина ответа
    """

    batch_size = len(offset_mapping_batch)

    inv_prompt_mapper = {v: k for k, v in prompt_mapper.items()}

    top_valid_spans_batch = []
    for i in range(batch_size):
        context = context_list[i]

        offset_mapping = offset_mapping_batch[i].cpu().numpy()
        mask = inputs["token_type_ids"][i].bool().cpu().numpy()
        offset_mapping[~mask] = [0, 0]
        offset_mapping = [
            (span if span != [0, 0] else None) for span in offset_mapping.tolist()
        ]

        start_logits = outputs["start_logits"][i].cpu().numpy()
        end_logits = outputs["end_logits"][i].cpu().numpy()

        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        top_valid_spans = []

        for start_index, end_index in zip(start_indexes, end_indexes):
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            if (end_index < start_index) or (
                end_index - start_index + 1 > max_answer_length
            ):
                continue
            if start_index <= end_index:
                start_context_char_char, end_context_char_char = offset_mapping[
                    start_index
                ]
                span = QASpan(
                    token=context[start_context_char_char:end_context_char_char],
                    label=inv_prompt_mapper[
                        question_list[i].split(r"What is the ")[-1].rstrip(r"?")
                    ],
                    start_context_char_pos=start_context_char_char,
                    end_context_char_pos=end_context_char_char,
                )
                top_valid_spans.append(span)
        top_valid_spans_batch.append(top_valid_spans)
    return top_valid_spans_batch


def calculate_metrics(
        model: Module,
        dataloader: DataLoader,
        id_to_tag: Dict[int, str],
        use_gpu: bool = False
):
    model.eval()

    epoch_loss = []
    gold_labels_per_sentence = []
    predict_labels_per_sentence = []

    with no_grad():
        for inputs in tqdm(dataloader):
            instances_batch = inputs.pop("instances")

            context_list, question_list = [], []
            for instance in instances_batch:
                context_list.append(instance.context)
                question_list.append(instance.question)

            if use_gpu:
                inputs = inputs.to('cuda')
            offset_mapping_batch = inputs.pop("offset_mapping")

            outputs = model.forward(**inputs)
            loss = outputs.loss

            epoch_loss.append(loss.item())
            spans_pred_batch_top_1 = get_top_valid_spans(
                context_list=context_list,
                question_list=question_list,
                prompt_mapper=dataloader.dataset.prompt_mapper,
                inputs=inputs,
                outputs=outputs,
                offset_mapping_batch=offset_mapping_batch,
                n_best_size=1,
                max_answer_length=100,
            )

            for idx in range(len(spans_pred_batch_top_1)):
                if not spans_pred_batch_top_1[idx]:
                    empty_span = QASpan(
                        token="",
                        label="O",
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    spans_pred_batch_top_1[idx] = [empty_span]

            gold_labels_per_sentence.append([instance.answer.label for instance in instances_batch])
            predict_labels_per_sentence.append([span.label for spans in spans_pred_batch_top_1 for span in spans])

    calculate_token_metrics(gold_labels_per_sentence, predict_labels_per_sentence, id_to_tag)

