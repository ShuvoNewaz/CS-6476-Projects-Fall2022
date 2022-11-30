import os
from typing import List, Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision.part2_baseline import Baseline
from vision.part3_pointnet import PointNet
from vision.part5_positional_encoding import PointNetPosEncoding


def train(
    model: Union[Baseline, PointNet, PointNetPosEncoding], 
    optimizer: optim.Optimizer, 
    epochs: int, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    device: str='cpu'
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains the given model using the given optimizer and datasets. Uses the
    val_loader to test model after each epoch. The validation data does not 
    contribute to the gradient descent performed by the optimizer.
    
    Args:
    -   model: The model to train
    -   optimizer: Optimizer to use when performing gradient descent
    -   epochs: Number of epochs to train the model for
    -   train_loader: The dataset to train the model with
    -   val_loader: The dataset to test the model with after each epoch
    -   device: if 'cuda' then trains on GPU. Trains on CPU by default.
    
    Output:
    -   train_acc_hist: History of training accuracy
    -   train_loss_hist: History of average training loss
    -   val_acc_hist: History of testing accuracy
    -   val_losS_hist: History of average test loss
    """
    os.makedirs('output', exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_acc_hist = [] 
    train_loss_hist = []
    val_acc_hist = []
    val_loss_hist = []
    best_val_acc = 0

    for epoch in range(epochs):
        print('Epoch %d' % (epoch+1))

        model.train()
        train_correct = 0
        train_loss = 0

        print('\nTraining...')
        for pts, labels in tqdm(train_loader):
            optimizer.zero_grad()
            pts, labels = pts.to(device), labels.to(device)
            preds, _ = model(pts)
            loss = criterion(preds, labels)
            loss.backward()
            train_correct += torch.sum(torch.argmax(preds, dim=-1) == labels)
            train_loss += loss

            optimizer.step()

        print('\nTesting...')
        with torch.no_grad():
            model.eval()
            val_correct = 0
            val_loss = 0
            for pts, labels in tqdm(val_loader):
                pts, labels = pts.to(device), labels.to(device)
                preds, _ = model(pts)
                val_loss += criterion(preds, labels)

                val_correct += torch.sum(torch.argmax(preds, dim=-1) == labels)    

        train_acc = train_correct / len(train_loader.dataset)
        train_avg_loss = train_loss / len(train_loader)
        val_acc = val_correct / len(val_loader.dataset)
        val_avg_loss = val_loss / len(val_loader)

        train_acc_hist.append(train_acc) 
        train_loss_hist.append(train_avg_loss)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_avg_loss)
        
        print('\nEpoch %d Stats:' % (epoch+1))
        print('\tTraining accuracy: %0.4f' % train_acc)
        print('\tTraining loss: %0.4f' % train_avg_loss)
        print('\tValidation accuracy: %0.4f' % val_acc)
        print('\tValidation loss: %0.4f' % val_avg_loss)

        if val_acc > best_val_acc:
            print(f'\nValidation accuracy improved from {best_val_acc} to {val_acc}')
            print(f'Saving model to {model.__class__.__name__}.pt')
            torch.save(model, os.path.join('output', f'{model.__class__.__name__}.pt'))
            best_val_acc = val_acc

        print('\n===============================================================================\n')  

    return train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist


def test(
    model: nn.Module, 
    loader: DataLoader, 
    device: str='cpu'
) -> Tuple[float, float]:
    """
    Tests the given model on the given data. If a save_path is specified,
    we load the model state dictionary from the file.

    Args:
    -   model: Model to test
    -   loader: Dataset to test on
    -   device: If 'cuda' then runs on GPU. Runs on CPU by default 

    Output:
    -   accuracy: Testing accuracy
    -   avg_loss: Average testing loss
    """
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0

    with torch.no_grad():
        for pts, labels in tqdm(loader):
            pts, labels = pts.to(device), labels.to(device)
            preds, _ = model(pts)
            loss += criterion(preds, labels)
            correct += torch.sum(torch.argmax(preds, dim=-1) == labels)

    accuracy = correct / len(loader.dataset)
    avg_loss = loss / len(loader)
        
    print('Test accuracy: %0.4f' % accuracy)
    print('Test loss: %0.4f' % avg_loss)

    return accuracy, avg_loss
