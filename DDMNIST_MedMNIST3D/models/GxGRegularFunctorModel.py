from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from models.BaseModels import DDMNISTCNN, CNN3DResnet
from utils.representations import C4xC4RegularRepresentation, D4xD4RegularRepresentation, D1xD1RegularRepresentation

class GxGRegularFunctor(pl.LightningModule):
    def __init__(self, lr=0.001, gamma=0.1, milestones=None, output_root=None, data_flag=None,
                 group='C4xC4', lambda_c=1.0, lambda_t=0.5, equivariant_layer_id=9, device='cuda'):
        super().__init__()
        # Save all hyperparameters including new ones for evaluation
        self.save_hyperparameters()
        self.lr = lr
        self.gamma = gamma
        self.milestones = milestones
        self.used_device = device
        self._W_full_cache = {}

        # Task specifics
        self.criterion = nn.CrossEntropyLoss()
        self.output_root = output_root


        # Select base model
        if lambda_t > 0:
            get_latent = True
        else:
            get_latent = False
        
        self.model = DDMNISTCNN(n_classes=100, n_channels=1, mnist_type='double', get_latent=get_latent)
        #self.latent_dim = self.model.dims[equivariant_layer_id]

        # Initialise functor parameters
        self.lambda_c = lambda_c
        self.lambda_t = lambda_t
        self.equivariant_layer_id = equivariant_layer_id
        
        self.group = group
        if group == 'C4xC4':
            self.W = C4xC4RegularRepresentation(device=device)
        elif group == 'D4xD4':
            self.W = D4xD4RegularRepresentation(device=device)
        elif group == 'D1xD1':
            self.W = D1xD1RegularRepresentation(device=device)
        else:
            raise ValueError(f"Group {group} not supported. Use 'C4xC4' or 'D4xD4'.")


    def forward(self, x):
        if self.lambda_t > 0:
            outputs, latent = self.model(x, self.equivariant_layer_id)
            return outputs, latent
        else:
            outputs = self.model(x, self.equivariant_layer_id)
            return outputs

    def _build_W_full(self, g: int, d: int) -> torch.Tensor:
        """
        Build a d x d transformation for group element g:
        block-kron of I_{d//B} and W(g) plus identity padding for remainder.
        """
        base = self.W(g)                # B x B base block
        B = base.size(-1)
        n_blocks = d // B
        rem = d - n_blocks * B

        # build entirely on GPU
        eye_nb = torch.eye(n_blocks, device=self.used_device)
        W_block = torch.kron(eye_nb, base)  # (n_blocks*B) x (n_blocks*B)

        if rem > 0:
            I_rem = torch.eye(rem, device=self.used_device)
            W_full = torch.block_diag(W_block, I_rem)
        else:
            W_full = W_block

        return W_full

    def get_W(self, g: int, d = None) -> torch.Tensor:
        """
        Return a cached full-d x d transform for group element g.
        If d is None, assert error.
        """
        assert isinstance(d, int) and d > 0, "Must specify latent dimension 'd'"
        key = (g, d)
        if key not in self._W_full_cache:
            self._W_full_cache[key] = self._build_W_full(g, d)
        return self._W_full_cache[key]
        
    def get_transformed_latent(self, latents, transformation_type, covariate):
        transformed = [torch.zeros_like(l, device=self.used_device)
                       for l in latents]
        # for each unique group element
        for g in covariate.unique().cpu().tolist():
            mask = covariate == g
            for idx, l in enumerate(latents):
                d_i = l.size(-1)
                W_g = self.get_W(int(g), d=d_i)
                # apply linear transform on masked entries
                transformed[idx][mask] = F.linear(l[mask], W_g)
        return transformed

    def get_transformation_loss(
        self,
        transformed_latents: List[torch.Tensor],
        latents2: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Sum MSE loss over corresponding latents.
        """
        losses = [F.mse_loss(t, l2)
                  for t, l2 in zip(transformed_latents, latents2)]
        return sum(losses)

    def get_natural_loss(self, outputs, y):
        loss = self.criterion(outputs, y)
        return loss
    
    def get_accuracy(self, outputs, y):
        pred_class = torch.argmax(outputs, dim=1)
        return ((pred_class == y).sum() / pred_class.shape[0]).item()

    def calculate_loss(self, batch, batch_idx, stage):
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        labels1 = y1
        labels2 = y2
        
        if self.lambda_t > 0:
            outputs1, latent1 = self(x1)
            outputs2, latent2 = self(x2)
        else:
            outputs1 = self(x1)
        

        ########### natural loss ###########
        if self.lambda_t > 0:
            natural_loss1 = self.get_natural_loss(outputs1, labels1)
            natural_loss2 = self.get_natural_loss(outputs2, labels2)
            natural_loss = 1*natural_loss1 + 0.*natural_loss2
        else:
            natural_loss = self.get_natural_loss(outputs1, labels1)
            

        ########### transformation loss ###########
        if self.lambda_t > 0:
            transformed_latent = self.get_transformed_latent(latent1, transformation_type, covariate)
            transformation_loss = self.get_transformation_loss(transformed_latent, latent2)
        else:
            transformation_loss = 0


        ########### logging ###########
        losses = {
            f'{stage}_natural_loss': natural_loss,
            f'{stage}_transformation_loss': transformation_loss
        }
        losses[f'{stage}_acc'] = self.get_accuracy(outputs1, labels1)

        loss = self.lambda_c * natural_loss + self.lambda_t * transformation_loss

        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss
            unreg_loss = natural_loss + transformation_loss
            losses[f'{stage}_unreg_loss'] = unreg_loss
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        return losses


    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'val')
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            loss = self.calculate_loss(batch, batch_idx, 'test')
        elif dataloader_idx == 1:
            loss = self.calculate_loss(batch, batch_idx, 'aug_test')
        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer]#, [scheduler]
    

    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)