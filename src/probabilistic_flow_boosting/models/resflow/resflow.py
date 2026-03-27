from typing import Any, Callable, Iterable, List, Union, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

# from probabilistic_flow_boosting.models.flow import ContinuousNormalizingFlow
# from probabilistic_flow_boosting.models.node import DenseODSTBlock
import zuko
import pdb
from probabilistic_flow_boosting.models.resflow.augmentations import embed_data_mask_mlp, embed_data_mask_mlp_cont
from probabilistic_flow_boosting.models.resflow.model import ResNetModel
from probabilistic_flow_boosting.models.node.activations import sparsemax, sparsemoid


class ResFlowDataModule(pl.LightningDataModule):
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.DataFrame] = None,
            split_size: float = 0.8,
            batch_size: int = 1024
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.split_size = split_size

        self.X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train

        if X_test is not None:
            self.X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
            self.y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test

        if self.split_size is not None:
            num_training_examples = int(self.split_size * self.X_train.shape[0])
            self.x_tr, self.x_val = self.X_train[:num_training_examples], self.X_train[num_training_examples:]
            self.y_tr, self.y_val = self.y_train[:num_training_examples], self.y_train[num_training_examples:]
        else:
            self.x_tr = self.X_train
            self.y_tr = self.y_train

        self.feature_scaler = Pipeline(
            [
                ("quantile", QuantileTransformer(output_distribution="normal")),
                ("standarize", StandardScaler()),
            ]
        )
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.x_tr: np.ndarray = self.feature_scaler.fit_transform(self.x_tr)
            self.y_tr: np.ndarray = self.target_scaler.fit_transform(self.y_tr)
            if self.split_size is not None:
                self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
                self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "validate":
            self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
            self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "test":
            self.x_tr: np.ndarray = self.feature_scaler.fit_transform(self.x_tr)
            self.y_tr: np.ndarray = self.target_scaler.fit_transform(self.y_tr)
            self.X_test = self.feature_scaler.transform(self.X_test)
            self.y_test = self.target_scaler.transform(self.y_test)

    def _to_dataloader(self, X, y):
        X: torch.Tensor = torch.as_tensor(X, dtype=torch.float32)
        y: torch.Tensor = torch.as_tensor(y, dtype=torch.float32)
        return DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._to_dataloader(X=self.x_tr, y=self.y_tr)

    def val_dataloader(self):
        return self._to_dataloader(X=self.x_val, y=self.y_val)

    def test_dataloader(self):
        return self._to_dataloader(X=self.X_test, y=self.y_test)

    def predict_dataloader(self):
        return self._to_dataloader(X=self.X_test, y=self.y_test)


class ResFlow(pl.LightningModule):
    def __init__(
            self,
            input_dim: int,
            output_dim: int=1,
            hidden_dim: int=250,
            depth: int=3,
            continous_embedding: str="MLP",
            hidden_dropout: float=0.0,
            residual_dropout: float=0.0,
            d_hidden_factor: float=1.0,
            flow_hidden_dims: Iterable[int] = 216,
            flow_num_blocks: int = 3,
            flow_layers: int=3,
            device: str = "cuda",
            dim: int=256,
            random_state: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        ####
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.depth = depth
        self.continous_embedding=continous_embedding
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout
        self.d_hidden_factor =d_hidden_factor
        ####
        self.flow_hidden_dims = hidden_dim
        self.flow_num_blocks = flow_num_blocks
        self.flow_layers = flow_layers
        self.random_state = random_state
        self._best_epoch = None
        # self.device=device


        # resnet model parameters
        cat_dims = []
        num_idx = [i for i in range(input_dim)]
        dim = self.dim 

        self.base_model = ResNetModel(
            categories=tuple(cat_dims), #  no categorical data now.
            num_continuous=len(num_idx),
            dim=dim,
            hidden_dim = self.hidden_dim,
            dim_out=1,
            depth=self.depth,  # 6
            mlp_hidden_mults=(4, 2),
            cont_embeddings=self.continous_embedding,
            final_mlp_style="sep",
            hidden_dropout=self.hidden_dropout,
            residual_dropout=self.residual_dropout,
            d_hidden_factor=self.d_hidden_factor,
            y_dim=1 # not used. Headless resnet model
        )

        self.flow_model = zuko.flows.NSF(1, self.flow_hidden_dims, transforms=self.flow_num_blocks, hidden_features=[self.hidden_dim]*self.flow_layers)



    @torch.enable_grad()
    def forward(self, y, x_cont_enc):
        """Calculate the log probability of the model (batch). Method used only for training and validation."""
        c = self.base_model(x_cont_enc) 
        log_prob = self.flow_model(c).log_prob(y) 
        log_prob += np.log(np.abs(np.prod(self.trainer.datamodule.target_scaler.scale_)))
        return log_prob

    @torch.enable_grad()
    def _log_prob(self, y, x_cont_enc):
        """Calculate the log probability of the model (batch). Method used only for testing."""
        grad_x = x_cont_enc.clone().requires_grad_()
        c = self.base_model(grad_x)
        log_prob = self.flow_model(c).log_prob(y)
        log_prob += np.log(np.abs(np.prod(
            self.trainer.datamodule.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return log_prob

    def training_step(self, batch, batch_idx):
        X, y = batch
        x_cont = X
        x_cont = x_cont
        con_mask = torch.ones_like(x_cont).long()

        x_cont_enc = embed_data_mask_mlp_cont(x_cont, con_mask, self.base_model) 
        
        logpx = self(y, x_cont_enc)
        loss = -logpx.mean()
        self.log("train_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        x_cont = X
        x_cont = x_cont
        con_mask = torch.ones_like(x_cont).long()

        x_cont_enc = embed_data_mask_mlp_cont(x_cont, con_mask, self.base_model) 

        logpx = self(y, x_cont_enc)
        loss = -logpx.mean()
        self.log("val_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        x_cont = X
        x_cont = x_cont
        con_mask = torch.ones_like(x_cont).long()

        x_cont_enc = embed_data_mask_mlp_cont(x_cont, con_mask, self.base_model) 
        
        logpx = self._log_prob(y, x_cont_enc)
        loss = -logpx.mean()
        self.log("test_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _embed_features(self, X: torch.Tensor) -> torch.Tensor:
        """Embed raw input features using the base model's continuous embeddings.

        Args:
            X: Raw input features of shape (batch_size, input_dim).

        Returns:
            Embedded features of shape (batch_size, input_dim, dim).
        """
        con_mask = torch.ones_like(X).long()
        return embed_data_mask_mlp_cont(X, con_mask, self.base_model)

    @torch.enable_grad()
    def _get_distribution(self, x_cont_enc: torch.Tensor):
        """Get the conditioned normalizing flow distribution for embedded features.

        Args:
            x_cont_enc: Embedded input features of shape (batch_size, input_dim, dim).

        Returns:
            A ``zuko.distributions.NormalizingFlow`` conditioned on the given features.
        """
        grad_x = x_cont_enc.clone().requires_grad_()
        c = self.base_model(grad_x)
        return self.flow_model(c)

    @torch.enable_grad()
    def cdf(self, y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute CDF values P(Y <= y | X=x) for each (y, x) pair.

        Uses the forward (normalizing) pass of the flow to map observations into
        the standard-Normal base space, then evaluates the Normal CDF there.

        Args:
            y: Target values of shape (batch_size, 1) in the model's scaled space.
            X: Raw input features of shape (batch_size, input_dim).

        Returns:
            CDF values of shape (batch_size, 1), each in [0, 1].
        """
        dist = self._get_distribution(self._embed_features(X))
        z = dist.transform.inv(y)
        return dist.base.base_dist.cdf(z)

    @torch.enable_grad()
    def quantile(self, q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute quantile values y such that P(Y <= y | X=x) = q.

        Uses the inverse (generative) pass of the flow: maps quantile levels
        through the base Normal inverse CDF, then through the generative
        transform back to the data space.

        Args:
            q: Quantile levels of shape (batch_size, 1), with values in (0, 1).
            X: Raw input features of shape (batch_size, input_dim).

        Returns:
            Quantile values of shape (batch_size, 1) in the model's scaled space.
        """
        dist = self._get_distribution(self._embed_features(X))
        z = dist.base.base_dist.icdf(q)
        return dist.transform(z)

    @torch.enable_grad()
    def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        grad_x = X.clone().requires_grad_()
        c = self.base_model(grad_x)
        x = self.flow_model(c).sample((num_samples,))[:,:,0].transpose(1,0)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, num_samples: int = 1000) -> Any:
        X, y = batch
        con_mask = torch.ones_like(X).long()
        x_cont_enc = embed_data_mask_mlp_cont(X, con_mask, self.base_model) 
        samples = self._sample(x_cont_enc, num_samples)

        # Inverse target transformation
        samples_size = samples.shape
        samples: np.ndarray = samples.detach().cpu().numpy()
        samples: np.ndarray = self.trainer.datamodule.target_scaler.inverse_transform(samples)
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = optim.RAdam(self.parameters(), lr=0.003)
        return optimizer

    def save(self, filename: str):
        torch.save(self, f"{filename}-resflow.pt")

    @classmethod
    def load(cls, filename: str, map_location=None):
        return torch.load(f"{filename}-resflow.pt", map_location=map_location)
