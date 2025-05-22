import os
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


class FixedMultiWFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01, lambda_W=0.001, lambda_comm=0.001, change_of_coords=False, save_W=True, run_id=None):
        super(FixedMultiWFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.lambda_comm = lambda_comm
        self.change_of_coords = change_of_coords
        self.save_W = save_W
        self.run_id = run_id

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

        self.W_exponent_algebra = 3
        print("W_exponent_algebra = ", self.W_exponent_algebra)

        # rho_2 is the 1D non trivial representation
        rho_2_reflection = torch.tensor([[-1]]) # rho_2(reflection)
        rho_2_rotation = torch.tensor([[1]])    # rho_2(rotation)

        # Rho_3 is the 2D non trivial representation
        rho_3_rotation = torch.tensor([[-0.5, -torch.sqrt(torch.tensor(3.0)) / 2],
                                [torch.sqrt(torch.tensor(3.0)) / 2, -0.5]])  # rho_3(rotation)
        rho_3_reflection = torch.tensor([[1, 0], [0, -1]])  # rho_3(reflection)
        
        self.W_rotation = torch.block_diag(torch.eye(latent_dim-3), rho_2_rotation, rho_3_rotation).to('cuda')
        self.W_reflection = torch.block_diag(torch.eye(latent_dim-3), rho_2_reflection, rho_3_reflection).to('cuda')
        
        if self.change_of_coords:
            #! It has to be the same basis!
            # self.P = nn.Parameter(torch.randn(latent_dim, latent_dim)).to('cuda')
            # Orthogonal P
            self.P = parametrizations.orthogonal(nn.Linear(latent_dim, latent_dim, bias=False)).to('cuda')

        

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

        if self.save_W:
            self.save_dir = f'fixed_D3/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}_lambdacomm={self.lambda_comm}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            save_path = f'{self.save_dir}/initialW_rotation_{self.run_id}.pt'
            torch.save(self.get_W_rotation(), save_path)
            save_path = f'{self.save_dir}/initialW_reflection_{self.run_id}.pt'
            torch.save(self.get_W_reflection(), save_path)

    def change_coordinates(self, W):
        P = self.P.weight
        P_inv = self.P.weight.T
        return P_inv @ W @ P

    def get_W_rotation(self):
        if self.change_of_coords:
            return self.change_coordinates(self.W_rotation)
        else:
            return self.W_rotation
    
    def get_W_reflection(self):
        if self.change_of_coords:
            return self.change_coordinates(self.W_reflection)
        else:
            return self.W_reflection

    def forward(self, x1, x2, transformation_type, covariate):
        encoded = self.encoder(x1)
        reconstructed = self.decoder(encoded)
        
        mask_rotation   = transformation_type == 0
        covariate_rotation = covariate[mask_rotation]
        mask_reflection = transformation_type == 1

        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()

        # Apply Rotations
        with torch.no_grad():
            W_rotation_powers_list = [None] * (covariate_rotation.max()+1)
        
        for c in covariate_rotation.unique():
            W_rotation_powers_list[c] = torch.linalg.matrix_power(W_rotation, c)

        W_rotation_powers = torch.stack([W_rotation_powers_list[c] for c in covariate_rotation])
        transformed_rotation = torch.bmm(W_rotation_powers, encoded[mask_rotation].unsqueeze(2)).squeeze(2)

        # Apply Reflections
        transformed_reflection = F.linear(encoded[mask_reflection], W_reflection)

        # Combine the transformations
        transformed = torch.zeros(x1.size(0), self.latent_dim, device=x1.device, dtype=reconstructed.dtype)
        transformed[mask_rotation]   = transformed_rotation
        transformed[mask_reflection] = transformed_reflection

        return reconstructed, transformed


    def loss_W_algebra(self):
        """
        Regularization term for the algebraic properties of the Ws.
        """
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()
        modularity_penalty_1 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        #modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), torch.linalg.pinv(W_rotation))
        modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_rotation.T)
        modularity_penalty_3 = torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda'))
        #modularity_penalty_4 = torch.dist(W_reflection, torch.linalg.pinv(W_reflection))
        modularity_penalty_4 = torch.dist(W_reflection, W_reflection.T)

        
        return (modularity_penalty_1 + modularity_penalty_2+ modularity_penalty_3 + modularity_penalty_4) / 2.0


    def loss_commutativity(self):
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()
        #commutation_penalty_1 = torch.dist(W_reflection @ W_rotation @ W_reflection, torch.linalg.pinv(W_rotation))
        commutation_penalty_2 = torch.dist(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda'))
        commutation_penalty_1 = torch.dist(W_reflection @ W_rotation @ W_reflection, W_rotation.T)
        return commutation_penalty_1 + commutation_penalty_2


    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2), transf_type, covariate = batch
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()

        loss_reconstruction_1 = self.criterion(reconstructed, x1)
        loss_reconstruction_2 = self.criterion(self.decoder(self.encoder(x2)), x2)
        loss_reconstruction = (0.5 * loss_reconstruction_1 + 0.5 * loss_reconstruction_2)
        #loss_reconstruction = nn.functional.nll_loss(reconstructed, x1)
        loss_transformation_1 = self.criterion(transformed_latent, self.encoder(x2))
        loss_transformation_2 = self.criterion(self.decoder(transformed_latent), x2)

        # T = 2 #is the epoch where interpolant=1
        # k = 5 #is the steepness of the sigmoid. Higher k means the transition is more abrupt.
        # interpolant = 1 / (1 + torch.exp(torch.tensor(-k * (self.current_epoch - T))))
        loss_transformation = (0.95 * loss_transformation_2 + 0.05 * loss_transformation_1)


      
        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_commutativity': torch.dist(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_rotation_modularity': torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_reflection_modularity': torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_transformation_fidelity': loss_transformation_2,
                f'{stage}_loss_latent_transformation': loss_transformation_1
            }
        
        

        loss = loss_reconstruction + self.lambda_t*loss_transformation 
        #loss = loss_reconstruction + lambda_t*loss_transformation + self.lambda_W*loss_W_algebra + self.lambda_comm*loss_commutativity
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

        log_reconstructed_and_decodedlatent(self, x1, x2, decoded_transformed_latent,reconstructed, 'validation', n_display=8)

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
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()
        self.log_dict(
            {
            'final_loss_W_rotation_modularity': torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
            'final_loss_W_reflection_modularity': torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda')),
            'final_loss_commutativity' : self.loss_commutativity()
            }
            )
        
        if self.save_W:
            save_path = f'{self.save_dir}/W_rotation_{self.run_id}.pt'
            torch.save(self.get_W_rotation().detach().cpu(), save_path)
            save_path = f'{self.save_dir}/W_reflection_{self.run_id}.pt'
            torch.save(self.get_W_reflection().detach().cpu(), save_path)


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
