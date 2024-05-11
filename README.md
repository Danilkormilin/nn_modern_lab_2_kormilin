# Лабораторная работа 2
## Тестирование PyTorch
Для выпоонения работы была выбрана библиотека PyTorch. Для этого была создана модель перцептрона на основе класса nn.Sequentional со следующей архитектурой:

```
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 10),
    nn.Sigmoid()
)
```

Результатом стала точность в 96.8% 

Результат обучения в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/loss_accu.png)

## Выполнение работы на датесете [Doom-Crossing](https://www.kaggle.com/datasets/andrewmvd/doom-crossing)
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/dataset-cover.png)

Для работы был выбран датасет Doom-Crossing из 400 изображений разделенный на тестовую и обучаюшую выборку в соотношении 8\2.
Было разработано две архитектуры перцептрона на основе nn.Sequentional:

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
  nn.Linear(kImageSizes[0] * kImageSizes[1] * 3, 400),
  nn.ReLU(),
  nn.Linear(400, 150),
  nn.ReLU(),
  nn.Linear(150, 50),
  nn.ReLU(),
  nn.Linear(50, 2),
  nn.Sigmoid()
).to(device)</code>code></td>
            <td><code>model_2 = nn.Sequential(
    nn.Linear(kImageSizes[0] * kImageSizes[1] * 3, 1000),
    nn.ReLU(),
    nn.Linear(1000, 450),
    nn.ReLU(),
    nn.Linear(450, 450),
    nn.ReLU(),
    nn.Linear(450, 150),
    nn.ReLU(),
    nn.Linear(150, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 2),
    nn.Sigmoid()
).to(device)</code></td>
        </tr>
        <tr>
            <td align="center">Best Accuracy</td>
            <td align="center">70,3%</td>
            <td align="center">74,8%</td>
        </tr>
        <tr>
            <td align="center">LearningRate</td>
            <td align="center">0.0001</td>
            <td align="center">0.0001-0.00001-0.000002</td>
        </tr>
        <tr>
            <td align="center">Number of epochs</td>
            <td align="center">10</td>
            <td align="center">9 - 10 - 10</td>
        </tr>
    </tbody>
</table>

В качестве оптимизатора был выбран AdamW. 
Далее представлены результаты обучения первой модели в виде графика:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/1_nn.png)

Вторая модель обучалась несколько раз с понижением LR:
![Датасет](https://github.com/Danilkormilin/nn_modern_lab_2_kormilin/blob/main/2_nn.png)
