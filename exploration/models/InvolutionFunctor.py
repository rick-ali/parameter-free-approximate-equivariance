import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from models.logging_utils import log_reconstructed_and_decodedlatent


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, int(latent_dim/2), kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(int(latent_dim/2), latent_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(latent_dim, int(latent_dim*2), kernel_size=7),  # 1x1
            nn.ReLU()
        )
        self.fc = nn.Linear(int(latent_dim*2), latent_dim)  # Flatten to latent_dim

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)
        return self.fc(x)  # Compress to latent_dim
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, int(latent_dim*2))  # Expand to match encoder output
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(latent_dim*2), latent_dim, kernel_size=7),  # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, int(latent_dim/2), kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(int(latent_dim/2), 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, z):
        z = self.fc(z)  # Expand latent vector
        z = z.view(z.size(0), int(self.latent_dim*2), 1, 1)  # Reshape to match encoder output
        return self.decoder(z)  # Decode to original size


class OrthogonalInvolutionMatrix(nn.Module):
    def __init__(self, size):
        super(OrthogonalInvolutionMatrix, self).__init__()

        self.Q_layer = parametrizations.orthogonal(nn.Linear(size, size, bias=False))
        
        # Initialize D as a vector for the diagonal entries
        self.D = nn.Parameter(torch.randn(size))

    def forward(self):
        Q = self.Q_layer.weight
        # Project D to +/-1
        D_proj = torch.diag(torch.sign(self.D))

        # Construct W = Q D Q^T
        W = Q @ D_proj @ Q.T
        return W

class InvolutionFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01):
        super(InvolutionFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

        self.W_layer = OrthogonalInvolutionMatrix(latent_dim)


        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

    def forward(self, x1, x2):
        encoded = self.encoder(x1)
        reconstructed = self.decoder(encoded)

        W = self.W_layer()
        transformed = F.linear(encoded, W)

        return reconstructed, transformed

    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2) = batch
        reconstructed, transformed_latent = self.forward(x1, x2)

        loss_reconstruction = self.criterion(reconstructed, x1)
        #loss_transformation = self.criterion(transformed_latent, self.encoder(x2))
        loss_transformation = self.criterion(self.decoder(transformed_latent), x2)

        W = self.W_layer()
        loss_W_algebra = torch.dist(W @ W, torch.eye(self.latent_dim, device='cuda'))
        Q = self.W_layer.Q_layer.weight
        loss_Q_orthogonal = torch.dist(Q @ Q.T, torch.eye(self.latent_dim, device='cuda'))
      
        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebra': loss_W_algebra,
                f'{stage}_loss_Q_orthogonal': loss_Q_orthogonal
            }
        
        loss = loss_reconstruction + self.lambda_t*loss_transformation
        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss
        return losses

    def training_step(self, batch, batch_idx):
        (x1, y1), (x2, y2) = batch
        losses = self.shared_step(batch, batch_idx, 'train')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        # Visualise what went into the model before the training step
        if batch_idx % 50 == 0:
            n_display = 10
            comparison = torch.cat([x1[:n_display], x2[:n_display]])
            grid = torchvision.utils.make_grid(comparison, nrow=n_display)
            self.logger.experiment.add_image(f"Input x1 and x2", grid, self.global_step)

        return losses

    def validation_step(self, batch, batch_idx):
        (x1, y1), (x2, y2) = batch
        reconstructed, transformed_latent = self.forward(x1, x2)
        decoded_transformed_latent = self.decoder(transformed_latent)
        losses = self.shared_step(batch, batch_idx, 'validation')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        log_reconstructed_and_decodedlatent(self, x1,x2,decoded_transformed_latent,reconstructed,'validation',n_display=8)

        return losses

    def test_step(self, batch, batch_idx):
        (x1, y1), (x2, y2) = batch
        reconstructed, transformed_latent = self.forward(x1, x2)
        decoded_transformed_latent = self.decoder(transformed_latent)
        losses = self.shared_step(batch, batch_idx, 'test')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        log_reconstructed_and_decodedlatent(self,x1,x2,decoded_transformed_latent,reconstructed,stage='test',n_display=16)

        return losses

    def on_test_epoch_end(self):
        Q = self.W_layer.Q_layer.weight
        loss_Q_orthogonal = torch.dist(Q @ Q.T, torch.eye(self.latent_dim, device='cuda'))
        W = self.W_layer()
        loss_W_algebra = torch.dist(W @ W, torch.eye(self.latent_dim, device='cuda'))

        self.log_dict(
            {
                'final_loss_Q_orthogonal' : loss_Q_orthogonal,
                'final_loss_W_algebra' : loss_W_algebra
            }
        )

        D = torch.diag(torch.sign(self.W_layer.D))
        diagonal_elements = torch.diagonal(D)
        num_ones = torch.sum(diagonal_elements == 1).item()
        num_neg_ones = torch.sum(diagonal_elements == -1).item()

        self.log_dict({'#+1s':num_ones, '#-1s':num_neg_ones})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def log_reconstructed_and_decodedlatent(self, x1, x2, decoded_transformed_latent, reconstructed, stage, n_display=8):
        comparison = torch.cat([x2[:n_display], decoded_transformed_latent[:n_display]])
        grid = torchvision.utils.make_grid(comparison, nrow=n_display)
        self.logger.experiment.add_image(f"{stage}: x2 vs. decoded latent", grid, self.global_step)

        comparison = torch.cat([x1[:n_display], reconstructed[:n_display]])
        grid = torchvision.utils.make_grid(comparison, nrow=n_display)
        self.logger.experiment.add_image(f"{stage}: x1 vs. reconstructed", grid, self.global_step)


    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)
