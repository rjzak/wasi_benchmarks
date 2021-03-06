#!/usr/bin/env python3
# Based on https://janakiev.com/blog/pytorch-iris/

import numpy as np
import tqdm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from struct import pack

EPOCHS = 100


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


if __name__ == "__main__":

    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

    model = Model(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()

    loss_list = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    torch.onnx.export(model, (X_test,), 'model.onnx')
    with open("iris_data.dat", "wb") as all_data:
        flat = X.flatten()
        all_data.write(pack("<" + "".join(["f"] * flat.shape[0]), *flat))

    with open("iris_labels.dat", "wb") as all_labels:
        all_labels.write(pack("<" + "".join(["f"] * y.shape[0]), *y))