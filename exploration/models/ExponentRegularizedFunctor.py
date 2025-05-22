import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from models.logging_utils import log_reconstructed_and_decodedlatent
from models.Encoders import MNISTEncoder

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


class ExponentRegularizedFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, learning_rate=0.003, lambda_t=0.01, lambda_W=0.001, W_exponent_algebra=2, save_W=True, run_id=None):
        super(ExponentRegularizedFunctor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.W_exponent_algebra = W_exponent_algebra

        self.losses = {}

        self.save_W = save_W
        self.run_id = run_id

        #self.encoder = Encoder(latent_dim=latent_dim)
        self.encoder = MNISTEncoder(latent_dim=latent_dim)
        #self.decoder = Decoder(latent_dim=latent_dim)
        self.classifier = nn.Linear(latent_dim, 10)  # Assuming 10 classes for classification

        # W = torch.empty(latent_dim, latent_dim, device='cuda')  # Replace n with desired matrix size
        # nn.init.orthogonal_(W)
        # # add a perturbation to W
        # W += torch.randn_like(W, device='cuda') * 0.3
        # self.W = nn.Parameter(W)
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim, device='cuda')*0.5)
        W = self.W
        print(torch.dist(torch.linalg.matrix_power(W, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')))
        print(torch.linalg.norm(W, ord='fro'))

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

        if self.save_W:
            #self.save_dir = f'rotations/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}'
            self.save_dir = f'classifier_exp/Winit=random_latentdim={self.latent_dim}_lambdaW={self.lambda_W}_lambdaT={self.lambda_t}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            save_path = f'{self.save_dir}/initialW_{self.run_id}.pt'
            torch.save(self.W, save_path)


    def forward(self, x1, x2, transformation_type, covariate):
        encoded = self.encoder(x1)
        #reconstructed = self.decoder(encoded)
        numeral_class = self.classifier(encoded)
        
        with torch.no_grad():
            W_powers_list = [None] * (covariate.max()+1)
        
        for c in covariate.unique():
            W_powers_list[c] = torch.linalg.matrix_power(self.W, c)

        W_angle_powers = torch.stack([W_powers_list[c] for c in covariate])
        transformed = torch.bmm(W_angle_powers, encoded.unsqueeze(2)).squeeze(2)

        return numeral_class, transformed
        #return reconstructed, transformed


    def loss_W_algebra(self):
        """
        Regularization term for the algebraic properties of W.
        """
        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(self.W, self.W_exponent_algebra-1), torch.linalg.pinv(self.W))
        return modularity_penalty + modularity_penalty_2


    def shared_step(self, batch, batch_idx, stage):
        """Common step for training, validation, and testing."""
        (x1, y1), (x2, y2), transf_type, covariate = batch
        y1 = y1[:, 0]
        y2 = y2[:, 0]
        reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)

        #loss_reconstruction = self.criterion(reconstructed, x1)
        #loss_transformation = self.criterion(self.decoder(transformed_latent), x2)
        loss_classification = 0.5*nn.functional.cross_entropy(reconstructed, y1) + 0.5*nn.functional.cross_entropy(reconstructed, y2)
        loss_transformation = nn.functional.mse_loss(transformed_latent, self.encoder(x2))

        loss_W_algebra = self.loss_W_algebra()

        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        norm_W = torch.linalg.norm(self.W, ord='fro')

        losses = {
                #f'{stage}_loss_reconstruction': loss_reconstruction,
                f'{stage}_loss_classification': loss_classification,
                f'{stage}_loss_transformation': loss_transformation,
                f'{stage}_loss_W_algebra': loss_W_algebra,
                f'{stage}_modularity_penalty': modularity_penalty,
                f'{stage}_norm_W': norm_W
            }
        
        #loss = loss_reconstruction + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra
        loss = loss_classification + self.lambda_t*loss_transformation + self.lambda_W*loss_W_algebra
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
        if self.global_step % 100 == 0:
            # Create "W_history" folder inside self.save_dir
            os.makedirs(f'{self.save_dir}/W_history_{self.run_id}', exist_ok=True)
            # Save the W matrices along with the current step
            save_path = f'{self.save_dir}/W_history_{self.run_id}/W_{self.global_step}.pt'
            torch.save(self.W.detach().cpu(), save_path)
            
            self.losses[self.global_step] = losses     
            with open(f'{self.save_dir}/W_history_{self.run_id}/losses_{self.run_id}.pkl', 'wb') as f:
                import pickle
                pickle.dump(self.losses, f)
        return losses


    def validation_step(self, batch, batch_idx):
        (x1, y1), (x2, y2), transf_type, covariate = batch
        #reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
        #decoded_transformed_latent = self.decoder(transformed_latent)
        losses = self.shared_step(batch, batch_idx, 'validation')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        #log_reconstructed_and_decodedlatent(self, x1,x2,decoded_transformed_latent,reconstructed,'validation',n_display=8)

        return losses


    def test_step(self, batch, batch_idx):
        (x1, y1), (x2, y2), transf_type, covariate = batch
        #reconstructed, transformed_latent = self.forward(x1, x2, transf_type, covariate)
        #decoded_transformed_latent = self.decoder(transformed_latent)
        losses = self.shared_step(batch, batch_idx, 'test')
        self.log_dict(losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        #log_reconstructed_and_decodedlatent(self,x1,x2,decoded_transformed_latent,reconstructed,stage='test',n_display=16)

        if self.save_W:
            save_path = f'{self.save_dir}/losses_{self.run_id}.pkl'
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(losses, f)
        return losses


    def on_test_epoch_end(self):
        loss_W_algebra = self.loss_W_algebra()
        modularity_penalty =  torch.dist(torch.linalg.matrix_power(self.W, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))

        self.log_dict({'final_loss_W_algebra' : loss_W_algebra})
        self.log_dict({'final_modularity_loss' : modularity_penalty})

        if self.save_W:
            save_path = f'{self.save_dir}/{self.run_id}.pt'
            torch.save(self.W.detach().cpu(), save_path)


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
