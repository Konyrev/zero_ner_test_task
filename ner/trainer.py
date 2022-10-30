import json
import time
import matplotlib.pyplot as plt

from torch import save
from torch.nn import Module
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Dict

from ner.metrics.ner import transform_target, transform_logits, calculate_metrics, calculate_span_metrics, check_metrics


def adjust_learning_rate(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(
        model: Module,
        optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        id_to_tag: Dict[int, str],
        number_of_epochs: int = 8,
        gradient_clip: float = 5.0,
        learning_rate: float = 0.0105,
        decay_rate: float = 0.05,
        transform_target: Callable = transform_target,
        transform_logits: Callable = transform_logits,
        model_input_formatter: Callable = lambda input: input,
        model_output_formatter: Callable = lambda output: output,
        calculate_metrics: Callable = calculate_metrics,
        calculate_span_metrics: Callable = calculate_span_metrics,
        check_metrics: Callable = check_metrics,
        model_name: str = 'baseline',
        use_gpu: bool = False,
        plot_every: int = 200
):
    if use_gpu:
        model = model.cuda()

    losses = {"train_losses": [], "valid_losses": [], 'test_losses': []}

    tr = time.time()
    count = 0
    loss = 0.
    for epoch in range(0, number_of_epochs):
        model.train(True)
        predict_labels, gold_labels = [], []
        for data in tqdm(train_dataloader):
            count += 1

            model.zero_grad()

            loss_, logits = model_output_formatter(model.forward(model_input_formatter(data)))
            loss += loss_.item() / len(data[0])
            loss_.backward()  # Считаем градиенты

            clip_grad_norm(model.parameters(), gradient_clip)  # Избегаем взрыв градиента
            optimizer.step()  # Шаг градиентного спуска

            gold_labels.append(transform_target(data[-1]))
            predict_labels.append(transform_logits(logits))

            if (count % plot_every) == 0:
                loss /= plot_every
                print(count, ': ', loss)
                # ОШбика
                losses['train_losses'].append(loss)
                loss = 0.0

        adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * (epoch + 1)))

        save(model.state_dict(), f'{model_name}_{epoch}.pkl')
        with open(f'losess_{model_name}_{epoch}.json', 'w') as f:
            json.dump(losses, f)

        calculate_metrics(gold_labels, predict_labels, id_to_tag)
        calculate_span_metrics(gold_labels, predict_labels, id_to_tag)
        losses['valid_losses'] += check_metrics(
            valid_dataloader,
            model,
            id_to_tag,
            model_input_formatter,
            model_output_formatter,
            transform_target,
            transform_logits
        )
        print(time.time() - tr)
        plt.plot(losses['train_losses'])
        plt.show()

    losses['test_losses'] = check_metrics(
        test_dataloader,
        model,
        model,
        id_to_tag,
        model_input_formatter,
        model_output_formatter,
        transform_target,
        transform_logits
    )

    return model, losses