from mlp import mlp
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryROC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class Deferral_Model(pl.LightningModule):
    def __init__(self, in_channels, num_classes = 3, alpha = 1.9):
        super().__init__()

        self.save_hyperparameters()
        self.mlp = mlp(in_channels, num_classes)
        self.alpha = alpha
        self.roc = BinaryROC().to("cpu")

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        loss = self.deferral_loss(out, y, self.alpha)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def evaluate(self, stage, batch, batch_idx):
        x, y = batch
        out = self(x)

        # loss
        loss = self.deferral_loss(out, y, self.alpha)
        self.log(f"{stage}/loss", loss, on_epoch=True)
        
        # pAUC
        self.roc.update(out[:,1].cpu(), y.cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.evaluate("test", batch, batch_idx)

    def on_validation_epoch_end(self):
        # log pAUC over whole dataset
        pAUC = self.compute_pAUC()
        self.log("val/pauc", pAUC, on_epoch=True)
    
    def on_test_epoch_end(self):
        # log pAUC over whole dataset
        pAUC = self.compute_pAUC()
        self.log("test/pauc", pAUC, on_epoch=True)

        fpr, tpr, _ = self.roc.compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                        estimator_name='Liu et al. Deferral Model')
        display.plot()
        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1e-4)
        return optimizer
    
    def deferral_loss(self, out, target, alpha = 1.9, eps_cst= 1e-8):
        loss = eps_cst
        batch_size = out.size(0)
        defer_class = 2

        for i in range(batch_size):
        
            l = - torch.log(
                torch.exp(out[i, target[i]]) / torch.sum(torch.exp(out[i]))
            ) - alpha * torch.log(
                torch.exp(out[i, defer_class]) / torch.sum(torch.exp(out[i]))
            )
            loss += l

        return loss / batch_size
    
    def compute_pAUC(self, specificity_range=(0.9, 1.0)):
        fpr, tpr, _ = self.roc.compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        specificity = 1 - fpr

        mask = (specificity >= specificity_range[0]) & (specificity <= specificity_range[1])
        selected_fpr = fpr[mask]
        selected_tpr = tpr[mask]

        if len(selected_fpr) > 1:
            pAUC = metrics.auc(selected_fpr, selected_tpr)
            # normalise to [0,1] by max possible AUc in this range
            pAUC = pAUC / (specificity_range[1] - specificity_range[0])
            pAUC = torch.from_numpy(np.array(pAUC))
        else:
            pAUC = torch.tensor(0.0)

        return pAUC