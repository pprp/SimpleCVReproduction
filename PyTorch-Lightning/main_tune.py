import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl

class NeuralNet(pl.LightningModule):
    def __init__(self, learning_rate=0.001, batch_size=3):
        super().__init__()
        # Boring model
        self.layer_1 = nn.Linear(30, 16)
        self.layer_2 = nn.Linear(16, 1)
        
        # 
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Log hyperparameters
        self.save_hyperparameters()

        # Built-in API for metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # Simple forward
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x)

        return x.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        
        # Feed the model and catch the prediction
        y_pred = self.forward(x)

        # Calculates loss for the current batch
        loss = F.binary_cross_entropy(y_pred, y)
        
        # Calculates accuracy for current batch
        train_acc_batch = self.train_accuracy(y_pred, y)

        # Save metrics for current batch
        self.log('train_acc_batch', train_acc_batch)
        self.log('train_loss_batch', loss)
        
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}
    
    def training_epoch_end(self, outputs):
    
        accuracy = self.train_accuracy.compute()

        # Save the metric
        self.log('Train_acc_epoch', accuracy)

    def validation_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        
        # Feed the model and catch the prediction (no need to set the model as "evaluation" mode)
        y_pred = self.forward(x)
        
        # Calculate loss for the current batch
        loss = F.binary_cross_entropy(y_pred, y)

        # Calculates accuracy for the current batch
        val_acc_batch = self.val_accuracy(y_pred, y)
        
        # Save metrics for current batch
        self.log('val_acc_batch', val_acc_batch)
        self.log('val_loss_batch', loss)

        return {'loss' : loss, 'y_pred' : y_pred, 'target' : y}

    def validation_epoch_end(self, outputs):

        accuracy = self.val_accuracy.compute()

        # Save the metric
        self.log('Val_acc_epoch', accuracy)


    def test_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        
        # Feed the model and catch the prediction (no need to set the model as "evaluation" mode)
        y_pred = self.forward(x)
        
        # Calculate loss for the current batch
        loss = F.binary_cross_entropy(y_pred, y)

        # Calculates accuracy for the current batch
        test_acc_batch = self.test_accuracy(y_pred, y)
        
        # Save metrics for current batch
        self.log('test_acc_batch', test_acc_batch)
        self.log('test_loss_batch', loss)

        # return {'loss' : loss, 'y_pred' : y_pred, 'target' : y}
        return test_acc_batch

    def prepare_data(self):
        self.x, self.y = load_breast_cancer(return_X_y=True)

    def setup(self, stage=None):

        x_train, x_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.3)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
            self.y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
            self.x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
            self.y_val = torch.from_numpy(y_val).type(torch.FloatTensor)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
            self.y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

    def train_dataloader(self):
        self.train_dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        self.val_dataset = torch.utils.data.TensorDataset(self.x_val, self.y_val)
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        self.test_dataset = torch.utils.data.TensorDataset(self.x_test, self.y_test)
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__ == "__main__":

    # Instantiate model
    model = NeuralNet()

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=100, 
                        check_val_every_n_epoch=10, 
                        precision=32,
                        weights_summary=None,
                        progress_bar_refresh_rate=1, 
                        auto_scale_batch_size='binsearch')
    
    # It is implemented the built-in function for finding the
    # optimal learning rate. Source: https://arxiv.org/pdf/1506.01186.pdf
    lr_finder = trainer.tuner.lr_find(model, 
                            min_lr=0.0005, 
                            max_lr=0.005,
                             mode='linear')
    
    # Plots the optimal learning rate
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # The suggested optimal learning rate is now taken as the default learning rate
    model.learning_rate = lr_finder.suggestion()

    # Once the optimal learning rate is found, let's find the largest optimal batch size
    trainer.tune(model)

    # # Once everything is done, let's train the model
    trainer.fit(model)

    # # Testing the model
    trainer.test()