from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from medmnist import Evaluator
from models.BaseModels import ResNet18

class MedMNISTModel(pl.LightningModule):
    def __init__(self, model_flag, n_channels, n_classes, task, data_flag, size, run,
                 lr=0.001, gamma=0.1, milestones=None, output_root=None):
        super().__init__()
        # Save all hyperparameters including new ones for evaluation
        self.save_hyperparameters()
        
        if model_flag == 'resnet18':
            #self.model = resnet18(pretrained=False, num_classes=n_classes)
            self.model = ResNet18(n_channels, n_classes)
        elif model_flag == 'resnet50':
            self.model = resnet50(pretrained=False, num_classes=n_classes)
        else:
            raise NotImplementedError

        self.task = task
        self.criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
        self.output_root = output_root

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.train_evaluator = Evaluator(data_flag, 'train', size=size)
        self.val_evaluator = Evaluator(data_flag, 'val', size=size)
        self.test_evaluator = Evaluator(data_flag, 'test', size=size)

    def forward(self, x):
        return self.model(x)
    
    def calculate_loss(self, batch, batch_idx, stage):
        inputs, targets = batch
        outputs = self(inputs)
        if self.task == 'multi-label, binary-class':
            targets_proc = targets.float()
        else:
            targets_proc = torch.squeeze(targets, 1).long()
        loss = self.criterion(outputs, targets_proc)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        if stage == 'val':
            self.validation_step_outputs.append(outputs)
        elif stage == 'test':
            self.test_step_outputs.append(outputs)

        return loss

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
        auc, acc = evaluator.evaluate(y_score, self.output_root, self.hparams.run)
        
        return {f'{stage}_auc': auc, f'{stage}_acc': acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]