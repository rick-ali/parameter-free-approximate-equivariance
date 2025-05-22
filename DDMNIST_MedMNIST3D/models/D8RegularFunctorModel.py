from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from medmnist import Evaluator
from models.BaseModels import ResNet18
from utils.initialise_W_utils import initialise_W_orthogonal, initialise_W_random, initialise_W_real_Cn_irreps
from utils.representations import D8RegularRepresentation

class D8RegularFunctor(pl.LightningModule):
    def __init__(self, model_flag, n_channels, n_classes, task, data_flag, size, run,
                 lr=0.001, gamma=0.1, milestones=None, output_root=None,
                 latent_dim=512, lambda_c=1.0, lambda_t=0.5, device='cuda'):
        super().__init__()
        # Save all hyperparameters including new ones for evaluation
        self.save_hyperparameters()
        self.lr = lr
        self.gamma = gamma
        self.milestones = milestones
        self.used_device = device

        # Task specifics
        self.task = task
        self.criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
        self.output_root = output_root

        # Get Evaluators
        self.train_evaluator = Evaluator(data_flag, 'train', size=size)
        self.val_evaluator = Evaluator(data_flag, 'val', size=size)
        self.test_evaluator = Evaluator(data_flag, 'test', size=size)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Select base model
        if model_flag == 'resnet18':
            self.model = ResNet18(n_channels, n_classes, get_latent=True)
        elif model_flag == 'resnet50':
            self.model = resnet50(pretrained=False, num_classes=n_classes)
        else:
            raise NotImplementedError

        # Initialise functor parameters
        self.lambda_c = lambda_c
        self.lambda_t = lambda_t
        self.latent_dim = latent_dim
        assert latent_dim % 8 == 0, "Latent dimension must be divisible by 8"
    
        self.W = D8RegularRepresentation(device=device)


    def forward(self, x):
        outputs, latent = self.model(x)
        return outputs, latent

    def get_W(self, g):
        return torch.kron(torch.eye(int(self.latent_dim/16), device=self.used_device), self.W(g))
        
    def get_transformed_latent(self, latent, transformation_type, covariate):
        transformed = torch.zeros_like(latent, device=self.used_device)
        for c in covariate.unique():
            transformed[covariate == c] = F.linear(latent[covariate == c], self.get_W(c.item()))

        return transformed

    def get_transformation_loss(self, transformed_latent, latent2):
        transformation_loss = nn.functional.mse_loss(transformed_latent, latent2)
        return transformation_loss

    def get_natural_loss(self, outputs, y):
        if self.task == 'multi-label, binary-class':
            targets_proc = y.float()
        else:
            targets_proc = torch.squeeze(y, 1).long()
        loss = self.criterion(outputs, targets_proc)
        return loss

    def calculate_loss(self, batch, batch_idx, stage):
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        labels1 = y1
        labels2 = y1
        
        outputs1, latent1 = self(x1)
        outputs2, latent2 = self(x2)
        

        ########### natural loss ###########
        natural_loss1 = self.get_natural_loss(outputs1, labels1)
        natural_loss2 = self.get_natural_loss(outputs2, labels2)
        natural_loss = 0.5*natural_loss1 + 0.5*natural_loss2


        ########### transformation loss ###########
        if self.lambda_t > 0:
            transformed_latent = self.get_transformed_latent(latent1, transformation_type, covariate)
            transformation_loss = self.get_transformation_loss(transformed_latent, latent2)
        else:
            transformation_loss = 0



        ########### logging ###########
        losses = {
            f'{stage}_natural_loss': natural_loss,
            f'{stage}_transformation_loss': transformation_loss,
        }
        loss = self.lambda_c * natural_loss + self.lambda_t * transformation_loss
        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss
            unreg_loss = natural_loss + transformation_loss
            losses[f'{stage}_unreg_loss'] = unreg_loss
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        

        ########### evaluation ###########
        if stage == 'val':
            self.validation_step_outputs.append(outputs1)
        elif stage == 'test':
            self.test_step_outputs.append(outputs1)
        
        return losses


    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'val')
        return loss

    def on_validation_epoch_end(self):
        result = self.standard_evaluation('val', self.validation_step_outputs, self.val_evaluator)
        self.validation_step_outputs.clear()
        self.log_dict(result)
        
        return result

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'test')
        return loss

    def on_test_epoch_end(self):
        result = self.standard_evaluation('test', self.test_step_outputs, self.test_evaluator)
        self.test_step_outputs.clear()
        self.log_dict(result)
        return result
    
    @torch.no_grad()
    def standard_evaluation(self, stage: str, outputs: List[torch.Tensor], evaluator: Evaluator):
        # Skip sanity check
        if self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return {f'{stage}_auc': 0, f'{stage}_acc': 0}
        
        logits = torch.cat(outputs, dim=0)
        if self.task == 'multi-label, binary-class':
            y_score = torch.nn.functional.sigmoid(logits)
        else:
            y_score = torch.nn.functional.softmax(logits, dim=1)
        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, None, self.hparams.run)
        
        return {f'{stage}_auc': auc, f'{stage}_acc': acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
    
    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)