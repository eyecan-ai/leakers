from leakers.datasets.factory import AlphabetDatasetFactory
from leakers.nn.modules.base import LeakerModule
from leakers.nn.modules.factory import LeakerModuleFactory, RandomizersFactory
from leakers.trainers.utils import ImageLogger, PipelineUtils, TensorUtils
import torch
import pytorch_lightning as pl


class LeakersTrainingModule(pl.LightningModule, ImageLogger):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # Coder Model
        self.model: LeakerModule = LeakerModuleFactory.create(self.hparams.coder)

        # Randomizer Model
        self.randomizer = RandomizersFactory.create(self.hparams.randomizer)

        # Reference dataset to compute word-to-index conversion
        self.proto_dataset = AlphabetDatasetFactory.create(self.hparams.dataset)

        # # Losses
        self.code_loss = eval(self.hparams.losses["code_loss"])()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.weight_code = self.hparams.losses["code_loss_weight"]
        self.weight_rot = self.hparams.losses["rot_loss_weight"]

        # Rotations
        self.randomize_rotation = self.hparams.rotations["randomize"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # Extract data
        code = batch["x"]
        B, _ = code.shape

        # Compute input angles classes
        if self.randomize_rotation:
            angles = torch.randint(0, 4, [B]).to(code.device)
        else:
            angles = (
                torch.Tensor([self.global_step % 4]).repeat(B).to(code.device).long()
            )

        # Generate Leakers
        imgs = self.model.generate(code, angles)

        # Randomize Leakers
        imgs = self.randomizer(imgs)

        # Compute leakers codes
        out = self.model.encode(imgs)

        # Predicted codes
        code_pred = out["code"]
        rot_logit = out["rot_logits"]

        # Losses
        loss_code = self.code_loss(code, code_pred)
        loss_rot = self.rot_loss(rot_logit, angles)
        loss = self.weight_code * loss_code + self.weight_rot * loss_rot

        self.log("train/loss", loss.item())
        self.log("train/loss_rot", loss_rot.item())
        self.log("train/loss_code", loss_code.item())

        return {
            "loss": loss,
            "loss_code": loss_code.detach(),
            "loss_rot": loss_rot.detach(),
        }

    def validation_step(self, batch, batch_idx):

        # Extract data
        code = batch["x"]
        y = batch["y"]
        B, _ = code.shape

        # Initalize running counters
        code_corrects = 0.0
        code_total = 0.0
        rot_corrects = 0.0
        rot_total = 0.0

        for rot in [0, 1, 2, 3]:
            angles = torch.Tensor([rot]).repeat(B).to(code.device).long()

            # Generate and Transform Leakers
            imgs = self.model.generate(code, angles)
            imgs_t = self.randomizer(imgs)

            # Log images
            if batch_idx == 0 and rot == 0:
                # self.debug_display(imgs, imgs_t)
                self.log_image(
                    "val/leakers",
                    TensorUtils.display_tensors([imgs, imgs_t], max_batch_size=8),
                )

            # Compute leakers codes
            out = self.model.encode(imgs_t)

            # Extract word index from codes
            code_pred = out["code"].detach().cpu().numpy()
            code_idx_pred = self.proto_dataset.words_to_indices(code_pred)
            code_idx = y.cpu().numpy()

            # Extract rotation classes
            rot_target = angles
            rot_pred = out["rot_classes"]

            # Computer running corrects
            code_corrects += (code_idx == code_idx_pred).sum().item()
            code_total += B
            rot_corrects += (rot_target == rot_pred).sum().item()
            rot_total += B

        return {
            "code_corrects": torch.Tensor([code_corrects]),
            "code_total": torch.Tensor([code_total]),
            "rot_corrects": torch.Tensor([rot_corrects]),
            "rot_total": torch.Tensor([rot_total]),
        }

    def validation_epoch_end(self, outputs) -> None:

        # Merge Results provided as List of Dicts
        code_total = PipelineUtils.sum_outputs(outputs, key="code_total")
        code_corrects = PipelineUtils.sum_outputs(outputs, key="code_corrects")
        rot_total = PipelineUtils.sum_outputs(outputs, key="rot_total")
        rot_corrects = PipelineUtils.sum_outputs(outputs, key="rot_corrects")

        # Compute accuracy
        code_accuracy = code_corrects / code_total
        rot_accuracy = rot_corrects / rot_total
        self.log("val/accuracy/code", code_accuracy)
        self.log("val/accuracy/rot", rot_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer["lr"])
