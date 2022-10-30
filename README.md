# Варианты решения

Для кодировки таргета я по-умолчанию использую IO кодировку, но возможность использовать BIO кодировку также присутствует.

### NER как классификация токенов

Самый очевидный подход к решению задачи извлечения сущностей. Для решения я выбрал две архитектуры:
* CNN-BiLSTM-Softmax и GLOVE как бэйзлайн на чистом PyTorch (Ноутбук: CNN-biLSTM-Softmax.ipynb, Macro F1 Score: by tokens 0.87, by spans 0.83). Я использовал гиперпараметры из статьи https://arxiv.org/pdf/1511.08308v5.pdf; 
* BERT-Softmax (Ноутбук: BERT-HF.ipynb, Macro F1 Score: by tokens 0.91, by spans 0.91). Для реализации этого варианта я использовал фреймворк HuggingFace и предобученную модель 'bert-base-cased', файнтюнил модель на Google Colab;


### NER как Question Answering

Протестировать работу модели можно в ноутбуке: Test QANer.ipynb.

Так как для исходной задачи подразумевается отсутствие какой-либо разметки и есть только данные типа "текст - что нужно извлечь" я решил попробовать поставить задачу иным способом. 

Самым очевидным вариантом было использовать подход "NER как NMT задача" (Template-Based Named Entity Recognition Using BART, 2021: https://arxiv.org/abs/2106.01760 ), но репозиторий из статьи (https://github.com/Nealcly/templateNER) не содержит в себе предобученную модель. И я решил отказаться от этого подхода, так как обучать с нуля модель на своем ноутбуке достаточно трудоемко.

Вместо этого я решил попробовать подход из статьи QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition, 2022: https://arxiv.org/abs/2203.01543 и для этого адаптировал код из репозитория https://github.com/dayyass/QaNER/ под датасет Conll2003 и пайплайн из исходного задания. Для решения задачи QA я использую предобученную модель из HuggingFace BERTForQuestionAbswering('bert-base-uncased')

Финальный ноутбук: QANer.ipynb, Macro F1 Score: by tokens 0.90.