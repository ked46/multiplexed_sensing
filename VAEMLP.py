import torch
import torch.nn as nn

''' Define classes and functions for MLP portion of training and testing including the combined VAE_MLP model'''
class MLP(nn.Module):
    def __init__(self,latent_dim,hidden_size,output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size), #adding the second hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.layers(x)

class CombinedModel(nn.Module):
    def __init__(self, vae, mlp):
        super(CombinedModel, self).__init__()
        self.vae = vae
        self.mlp = mlp

    def forward(self, x,latent_channel,seq_length):
        # Pass the input dataset through the VAE to obtain latent variables
        _,latent_variables, _ = self.vae(x,latent_channel,seq_length)

        # Pass the latent variables through the MLP to predict the parameters
        input_predictions = self.mlp(latent_variables)

        return input_predictions

def combo_train(model, train_loader, optimizer, criterion, device,latent_channel,seq_length):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        #predict input concentrations
        input_predictions = model(inputs,latent_channel,seq_length)

        loss = criterion(input_predictions, targets)
        #compute gradients
        loss.backward()
        #updates models parameters using the optimizer
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, test_loader, criterion, device,latent_channel,seq_length):
    model.eval()
    val_loss = 0.0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        #predict input concentrations
        input_predictions = model(inputs,latent_channel,seq_length)

        loss = criterion(input_predictions, targets)

        val_loss += loss.item()

    return val_loss/len(test_loader)
