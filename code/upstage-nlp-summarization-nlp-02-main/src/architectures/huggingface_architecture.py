from typing import Dict, Any
import os

import numpy as np
import pandas as pd

import torch
from torch import optim, nn
from torchmetrics import MetricCollection
from torchmetrics.text.rouge import ROUGEScore

from einops import repeat

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from transformers import AutoTokenizer


class HuggingFaceArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pretrained_model_name: str,
        strategy: str,
        lr: float,
        t_max: int,
        eta_min: float,
        interval: str,
        options: Dict[str, Any],
        target_max_length: int,
        target_min_length: int,
        per_device_save_path: str,
        target_column_name: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.pretrained_model_name = pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        self.strategy = strategy
        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.interval = interval
        self.options = options
        self.target_max_length = target_max_length
        self.target_min_length = target_min_length
        self.per_device_save_path = per_device_save_path
        self.target_column_name = target_column_name

        metrics = MetricCollection(
            [
                ROUGEScore(),
            ]
        )
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Invalid model mode: {mode}")
        output = self.model(encoded)
        return output

    def step(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        encoded = batch["encoded"]
        if (
            "bart" not in self.pretrained_model_name
            and "t5" not in self.pretrained_model_name
        ):
            encoded["labels"] = encoded["input_ids"]
        label = encoded["labels"]
        index = batch["index"]
        output = self(
            encoded=encoded,
            mode=mode,
        )
        logit = output.logits
        pred = torch.argmax(
            logit,
            dim=-1,
        )
        loss = output.loss
        return {
            "loss": loss,
            "logit": logit,
            "pred": pred,
            "label": label,
            "index": index,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(
                self.parameters(),
                lr=self.lr,
            )
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.lr,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="train",
        )
        loss = output["loss"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        pred = output["pred"]
        label = output["label"]
        encoded = batch["encoded"]
        generation = self.model.generate(
            encoded=encoded,
        )
        decoded_generation = self.data_encoder.batch_decode(
            sequences=generation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_label = self.data_encoder.batch_decode(
            sequences=label,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        metrics = self.test_metrics(
            decoded_generation,
            decoded_label,
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        return {
            "loss": loss,
            "pred": pred,
            "generation": generation,
            "decoded_generation": decoded_generation,
            "label": label,
            "decoded_label": decoded_label,
        }

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        logit = output["logit"]
        if len(logit.shape) < 3:
            logit = logit.unsqueeze(0)
        encoded = batch["encoded"]
        index = batch["index"]
        index_expanded = repeat(
            index,
            "batch_size -> batch_size data_max_length 1",
            data_max_length=logit.size(dim=1),
        )
        index_list = index.tolist()
        logit_with_index = (
            torch.cat(
                (
                    logit,
                    index_expanded,
                ),
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        device_num = self.device.index if self.device.index is not None else 0
        if not os.path.exists(f"{self.per_device_save_path}/logits"):
            os.makedirs(
                f"{self.per_device_save_path}/logits",
                exist_ok=True,
            )
        logit_file = f"{self.per_device_save_path}/logits/device_num={device_num}-batch_idx={batch_idx}.npy"
        if not os.path.exists(logit_file):
            np.save(
                logit_file,
                logit_with_index,
            )
        else:
            raise FileExistsError(f"{logit_file} already exists")

        generation = self.model.generate(
            encoded=encoded,
            options=self.options,
            target_max_length=self.target_max_length,
            target_min_length=self.target_min_length,
        )
        if (
            "bart" not in self.pretrained_model_name
            and "t5" not in self.pretrained_model_name
        ):
            input_length = len(encoded["input_ids"][0])
            generation = generation[:, input_length:]
        decoded_generation = self.data_encoder.batch_decode(
            sequences=generation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        cleaned_generation = list(
            map(
                lambda sentence: sentence.replace("\n", " ").replace("\r", " "),
                decoded_generation,
            )
        )
        output = {index_list[i]: cleaned_generation[i] for i in range(len(index_list))}
        if not os.path.exists(f"{self.per_device_save_path}/generations"):
            os.makedirs(
                f"{self.per_device_save_path}/generations",
                exist_ok=True,
            )
        generation_file = f"{self.per_device_save_path}/generations/device_num={device_num}-batch_idx={batch_idx}.csv"
        df = pd.DataFrame(
            {
                "index": output.keys(),
                self.target_column_name: output.values(),
            }
        )
        if not os.path.exists(generation_file):
            df.to_csv(
                generation_file,
                mode="w",
                header=True,
                index=False,
            )
        else:
            raise FileExistsError(f"{generation_file} already exists")

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        self.test_metrics.reset()
