import json
import matplotlib.pyplot as plt

from torch import no_grad, save
from torch.nn import Module
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from ner.metrics.ner import calculate_metrics as calculate_token_metrics
from ner.metrics.qa import get_top_valid_spans, calculate_metrics
from ner.qa_types import QASpan


def train_qa(
    model: Module,
    optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    id_to_tag: Dict[int, str],
    number_of_epochs: int = 3,
    gradient_clip: float = 5.0,
    print_every: int = 100,
    plot_every: int = 1000,
    use_gpu: bool=False
):
    if use_gpu:
        model = model.cuda()

    model.train()
    losses = []
    steps = 0
    for epoch in range(number_of_epochs):
        epoch_loss = []
        gold_labels_per_sentence = []
        predict_labels_per_sentence = []
        for inputs in tqdm(train_dataloader):
            optimizer.zero_grad()

            instances_batch = inputs.pop("instances")

            context_list, question_list = [], []
            for instance in instances_batch:
                context_list.append(instance.context)
                question_list.append(instance.question)

            if use_gpu:
                inputs = inputs.to('cuda')
            offset_mapping_batch = inputs.pop("offset_mapping")

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            clip_grad_norm(model.parameters(), gradient_clip)
            optimizer.step()

            steps += 1

            epoch_loss.append(loss.item())
            with no_grad():
                model.eval()
                outputs_inference = model(**inputs)
                model.train()

            spans_pred_batch_top_1 = get_top_valid_spans(
                context_list=context_list,
                question_list=question_list,
                prompt_mapper=train_dataloader.dataset.prompt_mapper,
                inputs=inputs,
                outputs=outputs_inference,
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

            if steps % print_every == 0:
                print(
                    "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
                )

            if steps % plot_every == 0:
                plt.plot(losses + epoch_loss)
                plt.show()

        save(model.state_dict(), f'qa_ner_{epoch}.pkl')
        with open(f'qa_ner_losses_{epoch}.json', 'w') as f:
            json.dump(losses, f)

        losses += epoch_loss
        plt.plot(losses)
        plt.show()

        calculate_token_metrics(gold_labels_per_sentence, predict_labels_per_sentence, id_to_tag)
        calculate_metrics(model, valid_dataloader, id_to_tag, use_gpu)

    calculate_metrics(model, test_dataloader, id_to_tag, use_gpu)

    return model, losses
