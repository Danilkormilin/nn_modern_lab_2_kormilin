# Лабораторная работа 3
## Выполнение работы на датесете [Doom-Crossing](https://www.kaggle.com/datasets/andrewmvd/doom-crossing)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/dataset-cover.png)

Для работы был выбран датасет Doom-Crossing из 1600 изображений разделенный на тестовую и обучаюшую выборку в соотношении 8\2.
Было разработано две архитектуры перцептрона на основе nn.Sequentional:

```
model_1 = nn.Sequential(
    nn.Conv2d(3, 8, (3,3), 1),
    nn.Conv2d(8, 8, (3,3), 1),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(8, 16, (3,3), 1),
    nn.Conv2d(16, 16, (3,3), 1),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(16, 32, (3,3), 1),
    nn.Conv2d(32, 32, (3,3), 1),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(32, 32, (3,3), 1),
    nn.Conv2d(32, 32, (3,3), 1),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(32, 32, (3,3), 1),
    nn.Conv2d(32, 64, (3,3), 1),
    nn.Conv2d(64, 128, (3,3), 1),
    nn.MaxPool2d((2,2), 2),
    nn.Flatten(),
    nn.Linear(128, 2),
    nn.Sigmoid()
).to(device)
```

Максимальная точность - 73.5% - меньше чем в прошлой работе, вероятно из за того что сеть содержит больше связей а датасет маленький и необходимо использовать аугментации

В качестве оптимизатора был выбран AdamW. 
Далее представлены результаты обучения первой модели в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/loss_lab3.png)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/accu_lab3.png)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/conf_lab3.png)
