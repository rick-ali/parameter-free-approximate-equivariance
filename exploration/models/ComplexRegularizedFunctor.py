import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu
from torch import sigmoid

def matrix_square(layer, dtype=torch.complex64):
    W_r = layer.fc_r.weight
    W_i = layer.fc_i.weight
    result_real = (W_r @ W_r - W_i @ W_i).type(dtype)
    result_im = 1j*(W_r @ W_i + W_i @ W_r).type(dtype)
    return result_real + result_im

def complex_sigmoid(inp):
    return sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(
        torch.complex64
    )

class ComplexRelu(nn.Module):
    def __init__(self):
        super(ComplexRelu, self).__init__()
    
    def forward(self, x):
        return complex_relu(x)
    
class ComplexSigmoid(nn.Module):
    def __init__(self):
        super(ComplexSigmoid, self).__init__()

    def forward(self, x):
        return complex_sigmoid(x)


class ComplexEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ComplexEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ComplexConv2d(1, int(latent_dim/2), kernel_size=3, stride=2, padding=1),  # 14x14
            ComplexRelu(),
            ComplexConv2d(int(latent_dim/2), latent_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            ComplexRelu(),
            ComplexConv2d(latent_dim, int(latent_dim*2), kernel_size=7),  # 1x1
            ComplexRelu()
        )
        self.fc = ComplexLinear(int(latent_dim*2), latent_dim)  # Flatten to latent_dim

    def forward(self, x):
        x = x.type(torch.complex64)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)
        return self.fc(x)  # Compress to latent_dim
    

class ComplexDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ComplexDecoder, self).__init__()
        self.fc = ComplexLinear(latent_dim, int(latent_dim*2))  # Expand to match encoder output
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            ComplexConvTranspose2d(int(latent_dim*2), latent_dim, kernel_size=7),  # 7x7
            ComplexRelu(),
            ComplexConvTranspose2d(latent_dim, int(latent_dim/2), kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            ComplexRelu(),
            ComplexConvTranspose2d(int(latent_dim/2), 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            ComplexSigmoid()  # Normalize to [0, 1]
        )

    def forward(self, z):
        z = self.fc(z)  # Expand latent vector
        z = z.view(z.size(0), int(self.latent_dim*2), 1, 1)  # Reshape to match encoder output
        z = self.decoder(z)  # Decode to original size
        return z.real


class ComplexRegularizedFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01, lambda_W=0.01):
        super(ComplexRegularizedFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.encoder = ComplexEncoder(latent_dim=latent_dim)
        self.decoder = ComplexDecoder(latent_dim=latent_dim)

        #self.W = torch.nn.Parameter(torch.eye(latent_dim, latent_dim), requires_grad=True)
        self.W = ComplexLinear(latent_dim, latent_dim)

        self.criterion = nn.MSELoss()

    def forward(self, x1, x2):
        encoded = self.encoder(x1)
        reconstructed = self.decoder(encoded)

        transformed = self.W(encoded)

        return reconstructed, transformed

    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2) = batch
        reconstructed, transformed_latent = self.forward(x1, x2)
        loss_reconstruction = self.criterion(reconstructed, x1)
        loss_transformation = self.criterion(self.decoder(transformed_latent), x2)
        loss_W_algebra = torch.dist(matrix_square(self.W), torch.eye(self.latent_dim, device='cuda'))
        
        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebra': loss_W_algebra
            }
        
        loss = loss_reconstruction + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra
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

        self.log_reconstructed_and_decodedlatent(x1,x2,decoded_transformed_latent,reconstructed,'validation',n_display=8)

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

        self.log_reconstructed_and_decodedlatent(x1,x2,decoded_transformed_latent,reconstructed,stage='test',n_display=16)

        return losses

    def on_test_epoch_end(self):
        loss_W_algebra = torch.dist(matrix_square(self.W), torch.eye(self.latent_dim, device='cuda'))

        self.log_dict(
            {
                'final_loss_W_algebra' : loss_W_algebra
            }
        )

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
