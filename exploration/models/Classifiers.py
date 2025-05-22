import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import random
from models.Encoders import MNISTEncoder, CIFAR10Encoder
from models.initialise_W_utils import initialise_W_random, initialise_W_orthogonal, initialise_W_random_roots_of_unity, initialise_W_real_Cn_irreps
import torchvision
from utils.char_tables import Cn_CharTable

class LinearClassifier(pl.LightningModule):
    def __init__(self, latent_dim=32, num_classes=2, autoencoder=None, freeze_latent=True):
        super(LinearClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_dim, num_classes)
        )

        self.autoencoder = autoencoder
        if autoencoder is not None and freeze_latent:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        

        self.classification_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    
    def forward(self, x):
        return self.layer(x)
    
    def shared_step(self, batch, batch_idx, stage):
        (x1, y1), (x2, y2) = batch
        # In the train.py, I would need to set classifier.autoencoder = autoencoder, after that's being trained
        latent = self.autoencoder.encoder(x1)  # Get latent representation
        x1_classification = self(latent)
        loss = nn.functional.cross_entropy(x1_classification, y1)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if stage == 'test':
            test_accuracy = self.classification_metric(x1_classification, y1)
            self.log(f'test_accuracy', test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss
     

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'validation')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class EncoderClassifierFunctor(pl.LightningModule):
    def __init__(self, latent_dim=32, num_classes=10, x2_angle=None, lambda_W=0.5, lambda_t=0.5, save_W=True, run_id=None):
        super(EncoderClassifierFunctor, self).__init__()
        self.save_hyperparameters()

        assert 360 % x2_angle == 0
        self.W_exponent_algebra = int(360 // x2_angle)
        print("NUM CLASSES = ", num_classes)
        print("W_exponent_algebra = ", self.W_exponent_algebra)
        self.initial_lambda_W = lambda_W
        self.lambda_W = lambda_W
        self.lambda_t = lambda_t
        self.latent_dim = latent_dim
        self.save_W = save_W
        self.run_id = run_id

        self.algebra_critertion = nn.MSELoss()

        self.W_rotation = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.3))
        #self.W_rotation = nn.Parameter(initialise_W_real_Cn_irreps([12, 2, 2, 2], self.W_exponent_algebra, True)).to('cuda')
        #self.W_rotation = nn.Parameter(initialise_W_random(latent_dim)).to('cuda')

        self.encoder = CIFAR10Encoder(latent_dim=latent_dim)
        
        self.numeral_classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes, bias=False),
            nn.Softmax(dim=1)
        )
        self.rotation_classifier = nn.Sequential(
            nn.Linear(latent_dim, self.W_exponent_algebra, bias=False),
            nn.Softmax(dim=1)
        )

        self.entangled_classifier = nn.Linear(latent_dim, num_classes + self.W_exponent_algebra, bias=False)

        self.numeral_classification_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.rotation_classification_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.W_exponent_algebra)

        

        if self.save_W:
            self.save_dir = f'new_C4_CIFAR_classification_disentangled/latentdim={self.latent_dim}_lambda_t={self.lambda_t}_lambdaW={self.lambda_W}'
            os.makedirs(self.save_dir, exist_ok=True)
            import yaml
            with open(f'{self.save_dir}/hyperparameters.yaml', 'w') as f:
                yaml.dump(dict(self.hparams), f, default_flow_style=False)
            save_path = f'{self.save_dir}/initialW_rotation_{self.run_id}.pt'
            torch.save(self.get_W_rotation(), save_path)

    
    def forward(self, x1, x2, transformation_type, covariate):
        encoded_x1 = self.encoder(x1)
        encoded_x2 = self.encoder(x2)
        x1_numeral_prediction, x1_rotation_prediction = self.get_disentangled_classification(encoded_x1)
        #x1_numeral_prediction, x1_rotation_prediction = self.get_entangled_classification(encoded_x1)

        
        with torch.no_grad():
            W_powers_list = [None] * (covariate.max()+1)
        
        for c in covariate.unique():
            W_powers_list[c] = torch.linalg.matrix_power(self.W_rotation, c)

        W_rotation_powers = torch.stack([W_powers_list[c] for c in covariate])
        transformed = torch.bmm(W_rotation_powers, encoded_x1.unsqueeze(2)).squeeze(2)
        #transformed = encoded_x1

        #W_rotation_powers_inverse = torch.einsum('ijk->ikj', W_rotation_powers)
        #inverse_transformed = torch.bmm(W_rotation_powers_inverse, encoded_x2.unsqueeze(2)).squeeze(2)

        #return x1_numeral_prediction, x1_rotation_prediction, transformed, inverse_transformed, encoded_x1, encoded_x2
        return x1_numeral_prediction, x1_rotation_prediction, transformed, encoded_x1, encoded_x2
    
    def get_disentangled_classification(self, latent_x):
        x_numeral = nn.functional.softmax(self.numeral_classifier(latent_x), dim=1)
        x_rotation = nn.functional.softmax(self.rotation_classifier(latent_x), dim=1)

        return x_numeral, x_rotation

    def get_entangled_classification(self, latent_x):
        entangled = self.entangled_classifier(latent_x)
        x_numeral = nn.functional.softmax(entangled[:, :10], dim=1)
        x_rotation = nn.functional.softmax(entangled[:, 10:], dim=1)

        return x_numeral, x_rotation


    def get_W_rotation(self):
        return self.W_rotation


    def loss_W_algebra(self):
        """
        Regularization term for the algebraic properties of W.
        """
        W_rotation = self.get_W_rotation()
        #i = random.randint(0, self.W_exponent_algebra-1)
        #modularity_penalty =  torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.linalg.matrix_power(W_rotation, i).T)
        # i = random.randint(0, 2)
        # if i == 0:
        #     modularity_penalty =  torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        # elif i == 1:
        #     modularity_penalty = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_rotation.T)
        # elif i == 2:
        #     modularity_penalty = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-2), W_rotation.T @ W_rotation.T)
        # return modularity_penalty
        # modularity_penalty_1 =  torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        # modularity_penalty_2 = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_rotation.T)
        # return modularity_penalty_1 + modularity_penalty_2
        W_inverse = torch.linalg.solve(W_rotation, torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty =  self.algebra_critertion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
        modularity_penalty_2 = self.algebra_critertion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-1), W_inverse)
        
        # i = random.randint(0, self.W_exponent_algebra)
        # modularity_penalty_2 =  torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra-i), torch.linalg.matrix_power(W_inverse, i))
        return modularity_penalty + modularity_penalty_2
       


    def loss_W_orthogonality(self):
        W_rotation = self.get_W_rotation()
        i = random.randint(0, 1)
        if i == 0:
            return self.algebra_critertion(W_rotation.T @ W_rotation, torch.eye(self.latent_dim, device='cuda'))
        else:
            return self.algebra_critertion(W_rotation @ W_rotation.T, torch.eye(self.latent_dim, device='cuda'))


    def lambda_scheduler(self, max_value, k, T):
        return max_value / (1 + torch.exp(torch.tensor(-k * (self.current_epoch - T/2))))


    def shared_step(self, batch, batch_idx, stage):
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        x1_numeral = y1[:,0]
        x1_rotation = y1[:,1]
        x2_numeral = y2[:,0]
        x2_rotation = y2[:,1]

        W_rotation = self.get_W_rotation()

        #x1_numeral_prediction, x1_rotation_prediction, transformed_latent, inverse_transformed_latent, encoded_x1, encoded_x2 = self.forward(x1, x2, transformation_type, covariate)
        x1_numeral_prediction, x1_rotation_prediction, transformed_latent, encoded_x1, encoded_x2 = self.forward(x1, x2, transformation_type, covariate)
        x2_numeral_prediction, x2_rotation_prediction = self.get_disentangled_classification(encoded_x2)
        transformed_latent_numeral_prediction, transformed_latent_rotation_prediction = self.get_disentangled_classification(transformed_latent)
        # x2_numeral_prediction, x2_rotation_prediction = self.get_entangled_classification(encoded_x2)
        # transformed_latent_numeral_prediction, transformed_latent_rotation_prediction = self.get_entangled_classification(transformed_latent)

   
        # Prediction loss
        loss_numeral_x1 = nn.functional.cross_entropy(x1_numeral_prediction, x1_numeral)
        loss_rotation_x1 = nn.functional.cross_entropy(x1_rotation_prediction, x1_rotation)
        loss_numeral_x2 = nn.functional.cross_entropy(x2_numeral_prediction, x2_numeral)
        loss_rotation_x2 = nn.functional.cross_entropy(x2_rotation_prediction, x2_rotation)
        #loss_prediction = 0.5*(loss_numeral_x1 + 0.5*loss_rotation_x1) + 0.5*(loss_numeral_x2 + 0.5*loss_rotation_x2)
        loss_prediction = .5*loss_numeral_x1 + .5*loss_numeral_x2

        # Transformation loss
        loss_transformation_latent = nn.functional.mse_loss(transformed_latent, encoded_x2)
        loss_transformation_class = nn.functional.cross_entropy(transformed_latent_numeral_prediction, x2_numeral)
        loss_transformation_rotation = nn.functional.cross_entropy(transformed_latent_rotation_prediction, x2_rotation)
        #loss_transformation = 0.8*loss_transformation_latent + 0.2*(loss_transformation_class + loss_transformation_rotation)
        loss_transformation = 0.5*loss_transformation_latent + 0.5*loss_transformation_class

        # Algebra loss
        loss_W_algebra = self.loss_W_algebra()
        loss_algebra = loss_W_algebra
            

        losses = {
            f'{stage}_x1_numeral_classification_accuracy': self.numeral_classification_metric(x1_numeral_prediction, x1_numeral),
            f'{stage}_x1_rotation_classification_accuracy': self.rotation_classification_metric(x1_rotation_prediction, x1_rotation),
            f'{stage}_transformed_x2_numeral_classification_accuracy': self.numeral_classification_metric(transformed_latent_numeral_prediction, x2_numeral),
            f'{stage}_transformed_x2_rotation_classification_accuracy': self.rotation_classification_metric(transformed_latent_rotation_prediction, x2_rotation),
            f'{stage}_x2_numeral_classification_accuracy': self.numeral_classification_metric(x2_numeral_prediction, x2_numeral),
            f'{stage}_x2_rotation_classification_accuracy': self.rotation_classification_metric(x2_rotation_prediction, x2_rotation),
            f'{stage}_loss_numeral_x1': loss_numeral_x1,
            f'{stage}_loss_rotation_x1': loss_rotation_x1,
            f'{stage}_loss_numeral_x2': loss_numeral_x2,
            f'{stage}_loss_rotation_x2': loss_rotation_x2,
            f'{stage}_loss_transformation': loss_transformation,
            f'{stage}_loss_transformation_latent': loss_transformation_latent,
            f'{stage}_loss_transformation_class': loss_transformation_class,
            #f'{stage}_loss_inverse_transformation_latent': loss_inverse_transformation_latent,
            f'{stage}_W_modularity': self.algebra_critertion(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda')),
            f'{stage}_W_orthogonality': self.algebra_critertion(W_rotation.T @ W_rotation, torch.eye(self.latent_dim, device='cuda')),
            f'{stage}_loss_W_algebra': loss_W_algebra,
            f'{stage}_lambda_W': self.lambda_W,
            }
        

        loss = loss_prediction + self.lambda_t * loss_transformation + self.lambda_W * loss_algebra
        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss

        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return losses
     

    def training_step(self, batch, batch_idx):
        (x1, y1), (x2, y2), _, _ = batch

        if batch_idx % 50 == 0:
            n_display = 10
            comparison = torch.cat([x1[:n_display], x2[:n_display]])
            grid = torchvision.utils.make_grid(comparison, nrow=n_display, normalize=True)
            self.logger.experiment.add_image(f"Input x1 and x2", grid, self.global_step)

        return self.shared_step(batch, batch_idx, 'train')
    
    # After train epoch ends, we want to update the lambda_W
    def on_train_epoch_end(self):
        with torch.no_grad():
            W_rotation = self.get_W_rotation()
            modularity_loss = torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))
            if (self.current_epoch > 1) and (modularity_loss > 0.5):
                self.lambda_W = self.lambda_W * 1.1
            elif modularity_loss < 0.2:
                self.lambda_W = max(self.lambda_W * 0.9, self.initial_lambda_W)
    

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'validation')
    
    def test_step(self, batch, batch_idx):
        losses = self.shared_step(batch, batch_idx, 'test')

        return losses
    

    def on_test_epoch_end(self):
        W_rotation = self.get_W_rotation()

        self.log_dict({'final_loss_W_rotation_modularity': torch.dist(torch.linalg.matrix_power(W_rotation, self.W_exponent_algebra), torch.eye(self.latent_dim, device='cuda'))})
        chartable = Cn_CharTable(self.W_exponent_algebra)
        dimensions = chartable.calculate_irreducible_reps_dimensions(W_rotation.cpu())
        for i, dimension in enumerate(dimensions):
            self.log_dict({f'{i}-th root of unities': dimension})

        if self.save_W:
            save_path = f'{self.save_dir}/W_rotation_{self.run_id}.pt'
            torch.save(W_rotation.detach().cpu(), save_path)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)