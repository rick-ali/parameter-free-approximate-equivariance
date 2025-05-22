import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from models.logging_utils import log_reconstructed_and_decodedlatent
from models.Encoders import MNISTDecoder, MNISTEncoder, CIFAR10Decoder, CIFAR10Encoder
import random
from models.initialise_W_utils import initialise_W_random, initialise_W_orthogonal, initialise_W_random_roots_of_unity


class MultiWFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.0003, lambda_t=0.01, lambda_W=0.001, lambda_comm=0.001, save_W=True, run_id=None):
        super(MultiWFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.lambda_comm = lambda_comm
        self.save_W = save_W
        self.run_id = run_id

        # self.encoder = CIFAR10Encoder(latent_dim=latent_dim)
        # self.decoder = CIFAR10Decoder(latent_dim=latent_dim)
        self.encoder = MNISTEncoder(latent_dim=latent_dim)
        self.decoder = MNISTDecoder(latent_dim=latent_dim)

        self.W_exponent_algebra = 3
        print("W_exponent_algebra = ", self.W_exponent_algebra)

        self.W_rotation = nn.Parameter(initialise_W_random(latent_dim))
        self.W_reflection = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.01))

        # Loss function (MSE for reconstruction)
        self.reconstruction_criterion = nn.MSELoss()
        self.criterion = nn.MSELoss()

        if self.save_W:
            self.save_dir = f'autoencoder_exp/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}_lambdacomm={self.lambda_comm}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            save_path = f'{self.save_dir}/initialW_rotation_{self.run_id}.pt'
            torch.save(self.get_W_rotation(), save_path)
            save_path = f'{self.save_dir}/initialW_reflection_{self.run_id}.pt'
            torch.save(self.get_W_reflection(), save_path)

    def get_W_rotation(self):
        return self.W_rotation
    
    def get_W_reflection(self):
        return self.W_reflection
    
    def get_inv_W_rotation(self):
        return torch.linalg.solve(self.W_rotation, torch.eye(self.latent_dim, device='cuda'))
    
    def get_inv_W_reflection(self):
        return self.W_reflection.T

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
        W_rotation_inv = self.get_inv_W_rotation()
        W_reflection_inv = self.get_inv_W_reflection()

        modularity_penalty_1 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_rotation_inv)

        modularity_penalty_3 = torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty_4 = torch.dist(W_reflection, W_reflection_inv)

        
        return (modularity_penalty_1 + modularity_penalty_2 + modularity_penalty_3 + modularity_penalty_4) / 2.0


    def loss_commutativity(self):
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()
        W_rotation_inv = self.get_inv_W_rotation()

        commutation_penalty_2 = torch.dist(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda'))
        commutation_penalty_1 = torch.dist(W_reflection @ W_rotation @ W_reflection, W_rotation_inv)
        return commutation_penalty_1 + commutation_penalty_2


    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2), transf_type, covariate = batch
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
        W_rotation = self.get_W_rotation()
        W_reflection = self.get_W_reflection()

        loss_reconstruction_1 = self.reconstruction_criterion(reconstructed, x1)
        loss_reconstruction_2 = self.reconstruction_criterion(self.decoder(self.encoder(x2)), x2)
        loss_reconstruction = (0.5 * loss_reconstruction_1 + 0.5 * loss_reconstruction_2)

        loss_transformation_1 = self.criterion(transformed_latent, self.encoder(x2))
        loss_transformation_2 = self.criterion(self.decoder(transformed_latent), x2)
        loss_transformation = (0.95 * loss_transformation_2 + 0.05 * loss_transformation_1)

        loss_W_algebra = self.loss_W_algebra()
        loss_commutativity = self.loss_commutativity()
        # loss_W_rotation_orthogonality = torch.dist(W_rotation @ W_rotation.T, torch.eye(self.latent_dim, device='cuda'))
        # loss_W_reflection_orthogonality = torch.dist(W_reflection @ W_reflection.T, torch.eye(self.latent_dim, device='cuda'))
      
        losses = {
                f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebras': loss_W_algebra,
                f'{stage}_loss_commutativity': torch.dist(W_reflection @ W_rotation @ W_reflection @ W_rotation, torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_rotation_modularity': torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_loss_W_reflection_modularity': torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda')),
                f'{stage}_transformation_fidelity': loss_transformation_2,
                f'{stage}_loss_latent_transformation': loss_transformation_1,
                # f'{stage}_loss_orthogonality_W_rotation': loss_W_rotation_orthogonality,
                # f'{stage}_loss_orthogonality_W_reflection': loss_W_reflection_orthogonality, #!
                f'{stage}_lambda_W': self.lambda_W,
            }
        
        

        loss = loss_reconstruction + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra + self.lambda_comm*loss_commutativity
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
            grid = torchvision.utils.make_grid(comparison, nrow=n_display, normalize=True)
            self.logger.experiment.add_image(f"Input x1 and x2", grid, self.global_step)

        return losses


    # def on_train_epoch_end(self):
    #     if self.current_epoch % 4 == 0 and self.current_epoch > 2:  # Every 2 epochs, add small noise
    #         print("Adding noise!")
    #         with torch.no_grad():
    #             BASE = 0.1
    #             W_rotation = self.get_W_rotation()
    #             W_reflection = self.get_W_reflection()
    #             loss_W_rotation_orthogonality = torch.dist(W_rotation @ W_rotation.T, torch.eye(self.latent_dim, device='cuda'))
    #             loss_W_reflection_orthogonality = torch.dist(W_reflection @ W_reflection.T, torch.eye(self.latent_dim, device='cuda'))
    #             loss_W_rotation_mod = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
    #             loss_W_reflection_mod = torch.dist(torch.linalg.matrix_power(W_reflection, 2), torch.eye(self.latent_dim, device='cuda'))
            
    #             max_rotation = min(max(round(loss_W_rotation_orthogonality.item()), round(loss_W_rotation_mod.item())), 3)
    #             max_reflection = min(max(round(loss_W_reflection_orthogonality.item()), round(loss_W_reflection_mod.item())), 3)


    #             self.W_rotation += torch.randn_like(self.W_rotation) * BASE**(3-max_rotation)
    #             self.W_reflection += torch.randn_like(self.W_reflection) * BASE**(3-max_reflection)
    #             print(f"Added noise to W_rotation and W_reflection: {BASE**(3-max_rotation)} and {BASE**(3-max_reflection)}")


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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss_reconstruction"}


    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)
