import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from models.Encoders import MNISTDecoder, MNISTEncoder
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


class MLPEncoder(nn.Module):
    """
    MLP encoder for MNIST images.
    Input:  batch of images shaped (B, 1, 28, 28)
    Output: latent vectors shaped (B, latent_dim)
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z = self.fc4(x)   # (B, latent_dim), no activation
        return z


class MLPDecoder(nn.Module):
    """
    MLP decoder for MNIST latent vectors.
    Input:  latent vectors shaped (B, latent_dim)
    Output: reconstructed images shaped (B, 1, 28, 28)
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 28*28)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.unflatten(x)
        return x

class MultiWFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01, lambda_W=0.001, lambda_comm=0.001, save_W=True, run_id=None):
        super(MultiWFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.lambda_comm = lambda_comm
        self.save_W = save_W
        self.run_id = run_id
        self.losses = {}

        #self.encoder = Encoder(latent_dim=latent_dim)
        #self.decoder = Decoder(latent_dim=latent_dim)

        # self.encoder = MNISTEncoder(latent_dim=latent_dim)
        # self.decoder = MNISTDecoder(latent_dim=latent_dim)
        self.encoder = MLPEncoder(latent_dim=latent_dim)
        self.decoder = MLPDecoder(latent_dim=latent_dim)

        self.W_exponent_algebra = 3
        print("W_exponent_algebra = ", self.W_exponent_algebra)

        # self.W_rotation = parametrizations.orthogonal(nn.Linear(latent_dim, latent_dim, bias=False))
        # self.W_reflection = parametrizations.orthogonal(nn.Linear(latent_dim, latent_dim, bias=False))
        self.W_rotation = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.W_reflection = nn.Parameter(torch.randn(latent_dim, latent_dim))

        # W_rotation = torch.empty(latent_dim, latent_dim, device='cuda')  # Replace n with desired matrix size
        # nn.init.orthogonal_(W_rotation)
        # # add a perturbation to W
        # W_rotation += torch.randn_like(W_rotation, device='cuda') * 0.3
        # self.W_rotation = nn.Parameter(W_rotation)

        # W_reflection = torch.empty(latent_dim, latent_dim, device='cuda')  # Replace n with desired matrix size
        # nn.init.orthogonal_(W_reflection)
        # # add a perturbation to W
        # W_reflection += torch.randn_like(W_reflection, device='cuda') * 0.3
        # self.W_reflection = nn.Parameter(W_reflection)

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()
        self.algebra_criterion = nn.MSELoss()

        if self.save_W:
            self.save_dir = f'autoencoder_exp/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}_lambdacomm={self.lambda_comm}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            os.makedirs(f'{self.save_dir}/W_history_{self.run_id}', exist_ok=True)
            save_path = f'{self.save_dir}/W_history_{self.run_id}/W_rotation_initial.pt'
            torch.save(self.get_W_rotation(), save_path)
            save_path = f'{self.save_dir}/W_history_{self.run_id}/W_reflection_initial.pt'
            torch.save(self.get_W_reflection(), save_path)

    def get_W_rotation(self):
        return self.W_rotation
    
    def get_W_reflection(self):
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
        modularity_penalty_1 = self.algebra_criterion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        #modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), torch.linalg.pinv(W_rotation))
        #modularity_penalty_2 = self.algebra_criterion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_rotation.T)
        modularity_penalty_3 = self.algebra_criterion(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda'))
        #modularity_penalty_4 = torch.dist(W_reflection, torch.linalg.pinv(W_reflection))
        #modularity_penalty_4 = self.algebra_criterion(W_reflection, W_reflection.T)

        return (modularity_penalty_1 + modularity_penalty_3) / 2.0
        #return (modularity_penalty_1 + modularity_penalty_2+ modularity_penalty_3 + modularity_penalty_4) / 2.0


    def loss_commutativity(self):
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()
        #commutation_penalty_1 = torch.dist(W_reflection @ W_rotation @ W_reflection, torch.linalg.pinv(W_rotation))
        commutation_penalty_2 = self.algebra_criterion(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda'))
        #commutation_penalty_1 = self.algebra_criterion(W_reflection @ W_rotation @ W_reflection, W_rotation.T)
        return commutation_penalty_2#commutation_penalty_1 + commutation_penalty_2


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
        loss_transformation = (0.99 * loss_transformation_2 + 0.01 * loss_transformation_1)

        loss_W_algebra = self.loss_W_algebra()
        loss_commutativity = self.loss_commutativity()
      
        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebras': loss_W_algebra,
                f'{stage}_loss_commutativity': self.algebra_criterion(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_rotation_modularity': self.algebra_criterion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_reflection_modularity': self.algebra_criterion(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_transformation_fidelity': loss_transformation_2,
                f'{stage}_loss_latent_transformation': loss_transformation_1,
                f'{stage}_loss_orthogonality_W_rotation': self.algebra_criterion(torch.linalg.pinv(W_rotation), W_rotation.T),
                #f'{stage}_lambda_t': lambda_t,
            }
        
        

        loss = loss_reconstruction + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra + self.lambda_comm*loss_commutativity
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
        if False:
            if self.global_step < 2000 and self.global_step % 100 == 0:
                # Create "W_history" folder inside self.save_dir
                os.makedirs(f'{self.save_dir}/W_history_{self.run_id}', exist_ok=True)
                # Save the W matrices along with the current step
                save_path = f'{self.save_dir}/W_history_{self.run_id}/W_rotation_{self.global_step}.pt'
                torch.save(self.get_W_rotation().detach().cpu(), save_path)
                save_path = f'{self.save_dir}/W_history_{self.run_id}/W_reflection_{self.global_step}.pt'
                torch.save(self.get_W_reflection().detach().cpu(), save_path)
                
                self.losses[self.global_step] = losses     
                with open(f'{self.save_dir}/W_history_{self.run_id}/losses_{self.run_id}.pkl', 'wb') as f:
                    import pickle
                    pickle.dump(self.losses, f)
            elif self.global_step > 2000 and self.global_step % 2000 == 0:
                # Save the W matrices along with the current step
                save_path = f'{self.save_dir}/W_history_{self.run_id}/W_rotation_{self.global_step}.pt'
                torch.save(self.get_W_rotation().detach().cpu(), save_path)
                save_path = f'{self.save_dir}/W_history_{self.run_id}/W_reflection_{self.global_step}.pt'
                torch.save(self.get_W_reflection().detach().cpu(), save_path)
                
                self.losses[self.global_step] = losses     
                with open(f'{self.save_dir}/W_history_{self.run_id}/losses_{self.run_id}.pkl', 'wb') as f:
                    import pickle
                    pickle.dump(self.losses, f)
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
            'final_loss_W_rotation_modularity': self.algebra_criterion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
            'final_loss_W_reflection_modularity': self.algebra_criterion(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda')),
            'final_loss_commutativity' : self.loss_commutativity()
            }
            )
        from utils.char_tables import D3_CharTable

        table = D3_CharTable()
        dimensions = table.calculate_irreducible_reps_dimensions([torch.eye(self.latent_dim).to('cpu'), self.get_W_reflection().to('cpu'), self.get_W_rotation().to('cpu')])
        for i, dim in enumerate(dimensions):
            self.log(f'final_dim_{i}', dim)
        if self.save_W:
            save_path = f'{self.save_dir}/W_history_{self.run_id}/W_rotation_final.pt'
            torch.save(self.get_W_rotation().detach().cpu(), save_path)
            save_path = f'{self.save_dir}/W_history_{self.run_id}/W_reflection_final.pt'
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