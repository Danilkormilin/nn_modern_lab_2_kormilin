# Лабораторная работа 4

## Выполнение работы на датесете [Doom-Crossing](https://www.kaggle.com/datasets/andrewmvd/doom-crossing)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/dataset-cover.png)

Для работы был выбран датасет Doom-Crossing из 400 изображений разделенный на тестовую и обучаюшую выборку в соотношении 8\2.
Было разработано две архитектуры сверточных нейронных сетей с помощью nn.Sequentional:

<table>
    <thead>
        <tr>
            <th>-</th>
            <th>Model 1</th>
            <th>Model 2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Architecture</td>
            <td><code>model_1 = nn.Sequential(
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
).to(device)</code>code></td>
            <td><code>model_2 = nn.Sequential(
    nn.Conv2d(3, 8, (3,3), 1),
    nn.Conv2d(8, 8, (3,3), 1),
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(8, 16, (3,3), 1),
    nn.Conv2d(16, 16, (3,3), 1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.AvgPool2d((2,2), 2),
    nn.Conv2d(16, 32, (3,3), 1),
    nn.Conv2d(32, 32, (3,3), 1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.AvgPool2d((2,2), 2),
    nn.Flatten(),
    nn.Linear(14112, 512),
    nn.Linear(512, 2),
    nn.Sigmoid()
).to(device)</code></td>
        </tr>
        <tr>
            <td align="center">Best Accuracy</td>
            <td align="center">72,8%</td>
            <td align="center">76,8%</td>
        </tr>
        <tr>
            <td align="center">LearningRate</td>
            <td align="center">0.002 - на старте</td>
            <td align="center">0.002 - на старте</td>
        </tr>
        <tr>
            <td align="center">Number of epochs</td>
            <td align="center">100</td>
            <td align="center">100</td>
        </tr>
    </tbody>
</table>

В качестве оптимизатора был выбран AdamW.
##Примененные улучшения
В качестве scheduler'а был выбран ReduceLROnPlateau.
Также для данной работы были выбраны следующие аугментации:
'''
A.Perspective(scale=(0.05, 0.3), p=1.),
A.Affine(rotate=25, scale=(0.9, 0.9),  p=1.),
A.HorizontalFlip(p=.5)
'''
это позволило немного увеличить датасет.
Далее представлены результаты обучения первой модели в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab4/plots_1.png)

Далее представлены результаты обучения второй модели в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab4/plots_2.png)

Далее можно увидеть confusion matrix первой и второй модели
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab4/conf_1.png)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/lab4/conf_2.png)

Интересной особенностью является то, что первая модель лучше справляется с animal crossing, а вторая с doom)))

Метрики

Модель 1:
Precision: 0.703
Recall: 0.803
Accuracy: 0.732

Модель 2:
Precision: 0.784
Recall: 0.737
Accuracy: 0.767
