from ResNet50_dropout import ResNet_50_mcdropout
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.AIROGS_dataloader import AIROGS
import numpy as np
from sklearn import metrics
from torchmetrics.classification import BinaryROC


class McDropoutModel(pl.LightningModule):
    def __init__(self, image_channels, num_classes, dropout_rate=0.2):
        super().__init__()

        self.save_hyperparameters()
        self.model = ResNet_50_mcdropout(image_channels, num_classes, dropout_rate)
        self.roc = BinaryROC().to("cpu")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        loss = F.cross_entropy(out, y)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def evaluate(self, stage, batch, batch_idx):
        x, y = batch
        out = self(x)

        loss = F.cross_entropy(out, y)
        
        self.log(f"{stage}/loss", loss, on_epoch=True)
        
        samples = self.sample(x, N=10)
        mean_sample = torch.mean(samples, dim=0)

        # pAUC
        self.roc.update(mean_sample[:,1].cpu(), y.cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.evaluate("test", batch, batch_idx)

    def on_validation_epoch_end(self):
        # log pAUC over whole dataset
        pAUC = self.compute_pAUC()
        self.log("val/pauc", pAUC, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def test_dataloader(self):
        path = "/AIROGS.h5"
        test_dataset = AIROGS(file_path=path, t="test", transform=None)
        
        return DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)

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

    def sample(self, x, N):
        samples = []

        for _ in range(N):
            samples.append(self(x).unsqueeze(0))
        
        samples = torch.cat(samples, dim=0) # N x batch_size x 2
        
        return samples