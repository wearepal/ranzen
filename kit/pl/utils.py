from __future__ import annotations
import sys
from typing import Any, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar, convert_inf, reset
from torch import Tensor
from tqdm import tqdm

__all__ = ["IterationBasedProgBar"]


class IterationBasedProgBar(ProgressBar):
    """Iteration-based PL progress bar.

    Training in Pytorch-lightning is epoch-centric - the default progress bar reflects this ethos.
    However in many cases iteration-based training is desirable or even required (when using non-sequential
    sampling, for instance). This progress bar is designed to be used with iteration-based training (which can
    be enabled by using, for instance,  an infBatchSampler), which means removing 'Epoch' from the display,
    excluding the validation iterations from the length of the main progress bar, and displaying progress with
    respect to max_steps instead of a combination of epochs and batches.

    Example:
        >>> datamodule = MyDataModule()
        >>> model = MyModel()
        >>> trainer = pl.Trainer(max_steps=1000, val_check_interval=150, callbacks=[IterationBasedProgBar()])
        >>> trainer.fit(model=model, dm=dm)
    """

    def init_train_tqdm(self) -> tqdm:
        """Initialise the tqdm bar for training."""
        return tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_batch_idx = trainer.batch_idx  # type: ignore
        self.main_progress_bar = self.init_train_tqdm()
        reset(self.main_progress_bar, trainer.max_steps)  # type: ignore

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ...

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[Any],
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._train_batch_idx += 1
        if self._should_update(self.train_batch_idx, convert_inf(trainer.max_steps)):  # type: ignore
            self._update_bar(self.main_progress_bar)  # type: ignore
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

    def init_validation_tqdm(self) -> tqdm:
        return tqdm(
            desc="Validating",
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._val_batch_idx = 0
        if trainer.sanity_checking:
            reset(self.val_progress_bar, sum(trainer.num_sanity_val_batches))  # type: ignore
        else:
            self.val_progress_bar = self.init_validation_tqdm()
            # fill up remaining
            self._update_bar(self.val_progress_bar)  # type: ignore
            reset(self.val_progress_bar, int(sum(trainer.num_val_batches)))  # type: ignore

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[Any],
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._update_bar(self.val_progress_bar)  # type: ignore

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.val_progress_bar.close()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._test_batch_idx = 0
        self.test_progress_bar = self.init_test_tqdm()
        reset(self.test_progress_bar)
        self.test_progress_bar.total = convert_inf(sum(trainer.num_test_batches))

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[Any],
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._test_batch_idx += 1
        if self._should_update(self.test_batch_idx, sum(trainer.num_test_batches)):
            self._update_bar(self.test_progress_bar)
