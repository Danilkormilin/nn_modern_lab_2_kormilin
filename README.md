# Лабораторная работа 5

## Выполнение работы на датесете [Doom-Crossing](https://www.kaggle.com/datasets/andrewmvd/doom-crossing)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/dataset-cover.png)

Для работы был выбран датасет Doom-Crossing из 400 изображений разделенный на тестовую и обучаюшую выборку в соотношении 8\2.
Была использована предобученая сеть resnet50 с измененной полносвязной частью на следующий блок:

```
  (fc): Sequential(
    (0): Linear(in_features=2048, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=2, bias=True)
    (5): Sigmoid()
  )
```

Для обучения сам resnet был заборожен, по сути обучался только классификатор
В качестве оптимизатора был выбран AdamW.
В качестве scheduler'а был выбран ReduceLROnPlateau.
Также для данной работы были выбраны следующие аугментации:
```
A.Perspective(scale=(0.05, 0.3), p=1.),
A.Affine(rotate=25, scale=(0.9, 0.9),  p=1.),
A.HorizontalFlip(p=.5)
```
это позволило немного увеличить датасет.

Удалось добится точности в 85,6%!!! Это уже неплохо в сравнении с предыдущими попытками
Далее представлены результаты обучения модели в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab5/loss_accu.png)

Далее можно увидеть confusion matrix модели
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab5/conf.png)


Метрики
Precision: 0.8581081081081081
Recall: 0.8355263157894737
Accuracy: 0.85625
