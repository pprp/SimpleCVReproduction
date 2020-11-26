# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

# For handling dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# The super PyTorch Lightning
import pytorch_lightning as pl

# A callback (we are gonna talk about it later)
from pytorch_lightning.callbacks import EarlyStopping

class NeuralNet(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        # Boring model
        self.layer_1 = nn.Linear(30, 16)
        self.layer_2 = nn.Linear(16, 1)

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()

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
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like: 
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]

        # Option 1
        # We can unfold the out['y_pred'] and out['y_true']
        # and calculate the accuracy for each batch, then just take the mean
        # accuracy = []
        # for out in outputs:
        #     accuracy.append(self.train_accuracy(out['y_pred'], out['y_true']))
        # accuracy = torch.mean(torch.stack(accuracy))
        # print(f"Train Accuracy: {accuracy}")

        # Option 2
        # We can directly implement the method ".compute()" from the accuracy function
        accuracy = self.train_accuracy.compute()
        # print(f"Train Accuracy: {accuracy}")

        # Save the metric
        self.log('Train_acc_epoch', accuracy, prog_bar=True)

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
        self.log('val_acc_batch', val_acc_batch, prog_bar=False)
        self.log('val_loss_batch', loss, prog_bar=False)

        return {'loss' : loss, 'y_pred' : y_pred, 'target' : y}

    def validation_epoch_end(self, outputs):
        
        # Option 1
        # accuracy = []
        # for out in outputs:
        #     accuracy.append(self.val_accuracy(out['y_pred'], out['y_true']))
        # accuracy = torch.mean(torch.stack(accuracy))
        # print(f"Validation Accuracy: {accuracy}")

        # Option 2
        # We can directly implement the method ".compute()" from the accuracy function
        accuracy = self.val_accuracy.compute()

        # Save the metric
        self.log('val_acc_epoch', accuracy, prog_bar=True)



if __name__ == "__main__":
    # Gets and split data
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    # From numpy to torch tensors
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

    # From numpy to torch tensors
    x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.FloatTensor)
    
    # Implements Dataset and DataLoader
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5)

    # Implements Dataset and DataLoader
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=5)

    # Init Neural Net model
    nn = NeuralNet(learning_rate=0.001)

    early_stopping = EarlyStopping('val_acc_epoch')
    
    # Init Trainer
    trainer = pl.Trainer(max_epochs=10, callbacks=[early_stopping])
    
    # Train
    trainer.fit(nn, train_dataloader, val_dataloader)