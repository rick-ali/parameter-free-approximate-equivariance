import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu
from torch import sigmoid
import os
from models.logging_utils import log_reconstructed_and_decodedlatent

class UnitaryComplexMatrix(nn.Module):
    def __init__(self, size, perturbation=0.3):
        super(UnitaryComplexMatrix, self).__init__()
        
        # Initialize the complex matrix as a unitary matrix
        # Step 1: Generate a random complex matrix
        real_part = torch.randn(size, size, device='cuda')
        imag_part = torch.randn(size, size, device='cuda')
        A = real_part + 1j * imag_part
        
        # Step 2: Apply QR decomposition to get a unitary matrix
        Q, R = torch.linalg.qr(A)
        
        # Ensure unitarity by adjusting phase
        Q = Q * torch.abs(torch.det(Q))**(-1/size)
        # Add some noise to Q 
        Q += torch.randn_like(Q, device='cuda') * perturbation
        
        # Step 3: Register real and imaginary parts of the unitary matrix as learnable parameters
        self.real = nn.Parameter(Q.real)  # Real part of the unitary matrix
        self.imag = nn.Parameter(Q.imag)  # Imaginary part of the unitary matrix
    
    def forward(self):
        # Combine real and imaginary parts to form the complex matrix W
        W = self.real + 1j * self.imag
        return W


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


class ComplexExponentRegularizedFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01, lambda_W=0.001, W_exponent_algebra=2, save_W=True, run_id=None):
        super(ComplexExponentRegularizedFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.W_exponent_algebra = W_exponent_algebra

        self.save_W = save_W
        self.run_id = run_id

        self.encoder = ComplexEncoder(latent_dim=latent_dim)
        self.decoder = ComplexDecoder(latent_dim=latent_dim)

        self.W = UnitaryComplexMatrix(latent_dim, perturbation=0.3)

        if self.save_W:
            self.save_dir = f'complex_rotations/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            save_path = f'{self.save_dir}/initialW_{self.run_id}.pt'
            torch.save(self.W(), save_path)

        self.criterion = nn.MSELoss()


    def forward(self, x1, x2, transformation_type, covariate):
        encoded = self.encoder(x1)
        reconstructed = self.decoder(encoded)
        
        with torch.no_grad():
            W_powers_list = [None] * (covariate.max()+1)
        
        for c in covariate.unique():
            W_powers_list[c] = torch.linalg.matrix_power(self.W(), c)

        W_angle_powers = torch.stack([W_powers_list[c] for c in covariate])
        transformed = torch.bmm(W_angle_powers, encoded.unsqueeze(2)).squeeze(2)

        return reconstructed, transformed


    def loss_W_algebra(self):
        """
        Regularization term for the algebraic properties of W.
        """
        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W(), self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(self.W(), self.W_exponent_algebra-1), torch.linalg.pinv(self.W()))
        return modularity_penalty + modularity_penalty_2


    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2), transf_type, covariate = batch
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)

        loss_reconstruction = self.criterion(reconstructed, x1)
        loss_transformation = self.criterion(self.decoder(transformed_latent), x2)

        loss_W_algebra = self.loss_W_algebra()

        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W(), self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        norm_W = torch.linalg.norm(self.W(), ord='fro')

        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebra': loss_W_algebra,
                f'{stage}_modularity_penalty': modularity_penalty,
                f'{stage}_norm_W': norm_W
            }
        
        loss = loss_reconstruction + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra
        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss
        return losses


    def training_step(self, batch, batch_idx):
        (x1, y1), (x2, y2), _, _ = batch
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
        (x1, y1), (x2, y2), transf_type, covariate = batch
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
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
        (x1, y1), (x2, y2), transf_type, covariate = batch
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
        decoded_transformed_latent = self.decoder(transformed_latent)
        losses = self.shared_step(batch, batch_idx, 'test')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        log_reconstructed_and_decodedlatent(self,x1,x2,decoded_transformed_latent,reconstructed,stage='test',n_display=16)

        if self.save_W:
            save_path = f'{self.save_dir}/losses_{self.run_id}.pkl'
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(losses, f)
        return losses


    def on_test_epoch_end(self):
        loss_W_algebra = self.loss_W_algebra()
        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W(), self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))

        self.log_dict({'final_loss_W_algebra' : loss_W_algebra})
        self.log_dict({'final_modularity_loss' : modularity_penalty})

        if self.save_W:
            save_path = f'{self.save_dir}/{self.run_id}.pt'
            torch.save(self.W().detach().cpu(), save_path)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)
