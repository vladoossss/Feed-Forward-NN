import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

from dataset import Circles
from nn import SimpleMLP

import os
import imageio

plt.style.use('dark_background')


def create_gif(path):
    images = []
    for file_name in sorted(os.listdir(path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(path, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(path + '/final.gif', images, duration=0.5)


class Trainer():
    def __init__(self, model, lr, criterion_num=0, optimizer_num=0):
        self.model = model

        criterions = [CrossEntropyLoss(),
                      NLLLoss()]
        self.criterion = criterions[criterion_num]

        optimizers = [Adam(self.model.parameters(), lr=lr),
                      SGD(self.model.parameters(), lr=lr)]
        self.optimizer = optimizers[optimizer_num]

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

    def fit(self, train_dataloader, n_epochs):
        all_epochs = []
        all_loss = []
        for epoch in range(n_epochs):
            print("Epoch: ", epoch)
            epoch_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                self.optimizer.zero_grad()  # зануляем градиенты
                # print("Batch ", i)

                output = self.model(x_batch)  # делаем feedforward
                loss = self.criterion(output, y_batch.type(torch.LongTensor))  # считаем ошибку модели
                loss.backward()  # шаг backpropagation
                # print(loss)
                self.optimizer.step()  # обновляем веса модели

                epoch_loss += loss.item()

            print(epoch_loss / len(train_dataloader))

            all_epochs.append(epoch)
            all_loss.append(epoch_loss / len(train_dataloader))

            # на каждой эпохе сохраняем график loss
            plt.plot(all_epochs, all_loss)
            plt.title('Loss function')
            plt.xlabel('num of epoch')
            plt.ylabel('loss')
            plt.savefig('loss_img/' + str(epoch) + '_loss.png')

        # очищаем график
        plt.clf()

        # сохраняем модель
        torch.save(model.state_dict(), 'net.pth')

        # создаем из имеющихся изображений .gif для loss
        create_gif('loss_img')

    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                _, predicted = torch.max(output_batch.data, 1)
                all_outputs = torch.cat((all_outputs, predicted), 0)

                # для каждого батча тестовых данных строим нормализованную матрицу ошибок
                cm = confusion_matrix(y_batch, predicted)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion matrix')
                plt.colorbar()

                thresh = cm.max() / 2.
                for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, k, format(cm[k, j], '.2f'), horizontalalignment="center",
                             color="white" if cm[k, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig('cm_img/' + str(i) + '_cm.png')
                plt.clf()

                # для каждого батча тестовых данных визуализируем результаты бинарной классификации
                # при удачно классификации один класс будет окрашен в бледно синий цвет(желтый+синий), а второй
                # в оранжевый(красный+желтый)
                plt.scatter(x_batch[:, 0], x_batch[:, 1], c=y_batch,
                            cmap=ListedColormap(['#FFFF00', '#FF0000']), marker='*')
                plt.scatter(x_batch[:, 0], x_batch[:, 1], c=predicted,
                            cmap=ListedColormap(['#0000FF', '#FFFF00']), marker='*',
                            alpha=0.5)
                plt.title('Predictions')
                plt.savefig('class_img/' + str(i) + '_class.png')
                plt.clf()

        # создаем из имеющихся изображений .gif для confusion matrix
        create_gif('cm_img')

        # создаем из имеющихся изображений .gif для predictions
        create_gif('class_img')

        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)
        return all_outputs

    def predict_proba_tensor(self, T):
        self.model.eval()
        with torch.no_grad():
            output = self.model(T)
        return output


if __name__ == "__main__":
    # генерируем данные
    train_dataset = Circles(n_samples=5000, shuffle=True, random_state=1, factor=0.5)
    test_dataset = Circles(n_samples=1000, shuffle=True, random_state=2, factor=0.5)
    print(test_dataset[:][1])

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # инициализируем нейросеть
    layers_list_example = [(2, 3), (3, 2)]
    model = SimpleMLP(layers_list_example)

    # обучаем модель
    trainer = Trainer(model, lr=0.01)
    print(trainer.device)

    trainer.fit(train_dataloader, n_epochs=10)

    # делаем предсказания на тестовом множестве
    pred = trainer.predict(test_dataloader)
    print(pred[:10])

    test = test_dataset[:][1][:10]
    print(test)

    # оцениваем количество верно предсказанных на тестовом множестве
    true_classified = (pred == test_dataset[:][1]).sum()  # количество верно предсказанных объектов
    test_accuracy = true_classified / len(pred)  # accuracy

    print(f"Test accuracy: {test_accuracy}")
