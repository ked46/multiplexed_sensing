import torch
import torch.nn as nn
import torch.nn.functional as F

'''Define classes for VAE using 1D CNN'''
# define CNN encoder class which maps input data to a latent space distribution
class CNNEncoder(nn.Module):
    #constructs the cnn encoder
    def __init__(self, latent_dim,latent_channel,seq_length): #input latent-dim is the size of the latent space (number of latent dimensions to use)
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim

        # define CNN layers consisting of 4 convolutional layers with ReLU activation functions
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 36, kernel_size=3, stride=1, padding=1), # layer 1 takes 1D input and outputs 128 dimensions (or channels)
            nn.ReLU(), #piecewise linear fxn that outputs the input directly if it's positive otherwise outputs zero
            nn.Conv1d(36, 36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(36,36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(36, latent_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        #define the fully connected linear layers
        self.fc_mean = nn.Linear(latent_channel * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(latent_channel * seq_length, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim,latent_channel,seq_length):
        #constructs the cnn decoder
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel=latent_channel
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(latent_channel, 36, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(36, 36, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(36, 36, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(36, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
        )

    def forward(self, z,latent_channel,seq_length):
        x = self.fc(z)
        x = x.view(x.size(0), latent_channel, seq_length)
        x = self.decoder(x)
        return F.relu(x)

class VAE(nn.Module):
    def __init__(self, latent_dim,latent_channel,seq_length):
        #constructs the VAE
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel=latent_channel
        self.encoder = CNNEncoder(latent_dim,latent_channel,seq_length)
        self.decoder = CNNDecoder(latent_dim,latent_channel,seq_length)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x,latent_channel,seq_length,use_mean=False):
        mean, logvar = self.encoder(x)
        if use_mean:
            z=mean #deterministic mean for evaluating pre-trained models
        else:
            z = self.reparameterize(mean, logvar) #stochastic output for training
        reconstruction = self.decoder(z,latent_channel,seq_length)
        return reconstruction, mean, logvar


'''Define training and evaluation functions'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#train VAE
def train(model, dataloader, optimizer, criterion, alpha,device,latent_channel,seq_length):
    model.train()
    running_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        reconstruction, mean, logvar = model(data,latent_channel,seq_length)
        recon_loss = criterion(reconstruction, data) #reconstruction loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp()) #regularization term(KL-divergence)
        #calculate total combined loss
        loss = recon_loss + alpha * kl_loss
        #compute gradients
        loss.backward()
        #updates models parameters using the optimizer
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

#evaluate VAE
def test(model, dataloader, criterion,device,latent_channel,seq_length):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            reconstruction, _, _ = model(data,latent_channel,seq_length)
            loss = criterion(reconstruction, data)

            running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

#extract latent variables (mean vectors) for a given dataset using the trained VAE model
def get_latent_variables(model, dataloader,device,latent_channel,seq_length):
    model.eval()
    all_latent_vars = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, mean, _ = model(data,latent_channel,seq_length)
            all_latent_vars.append(mean.detach().cpu())
    return torch.cat(all_latent_vars)

# implements a warmup schedule to start from a small learn rate.
def warmup_scheduler(epoch,warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 1.0