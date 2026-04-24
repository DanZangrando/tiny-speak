"""PyTorch Lightning module for TinyEyes training."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.tiny_eyes import TinyEyes


class TinyEyesLightning(pl.LightningModule):
    """LightningModule that encapsulates TinyEyes training & evaluation."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 512,
        topk: Iterable[int] | None = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("class_names",))

        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Instanciar nuevo modelo custom
        self.model = TinyEyes(
            num_classes=len(self.class_names), 
            hidden_dim=hidden_dim
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.topk = tuple(sorted(set(topk or (1,))))

        # Buffers for UI-friendly analysis after validation/testing.
        self._validation_buffer: List[Dict] = []
        self._test_buffer: List[Dict] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # TinyEyes retorna (logits, embeddings)
        logits, _ = self.model(images)
        return logits

    # ------------------------------------------------------------------
    # Training & evaluation steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        logits = self.forward(images)
        
        # Loss principal
        ce_loss = F.cross_entropy(logits, labels)
        
        # Loss simple y directo - sin complicaciones que puedan romper el aprendizaje
        loss = ce_loss

        metrics = self._compute_topk(logits, labels)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        for k, acc in metrics.items():
            self.log(
                f"{stage}_top{k}",
                acc,
                prog_bar=(k == 1),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            # Alias genérico para la interfaz
            if k == 1:
                self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        if stage in {"val", "test"}:
            buffer = self._validation_buffer if stage == "val" else self._test_buffer
            buffer.append(self._pack_predictions(batch, logits))

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:  # noqa: D401
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # noqa: D401
        # ✅ OPTIMIZADOR SIMPLE: Evitar problemas con param_groups por ahora
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _compute_topk(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[int, float]:
        max_k = min(max(self.topk), logits.size(1))
        _, pred = logits.topk(max_k, dim=1)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        metrics: Dict[int, float] = {}
        batch_size = labels.size(0)
        for k in self.topk:
            topk_correct = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            metrics[k] = (topk_correct.mul_(100.0 / batch_size)).item()
        return metrics

    def _pack_predictions(self, batch: Dict, logits: torch.Tensor) -> Dict:
        probs = torch.softmax(logits.detach().cpu(), dim=-1)
        topk = min(5, probs.size(1))
        top_vals, top_idx = torch.topk(probs, topk, dim=-1)
        return {
            "letters": list(batch["letter"]),
            "true_labels": batch["label"].detach().cpu().tolist(),
            "probabilities": probs.tolist(),
            "top_indices": top_idx.tolist(),
            "top_scores": top_vals.tolist(),
        }

    # Buffers exposed for Streamlit analysis ----------------------------------
    @property
    def validation_predictions(self) -> List[Dict]:
        return self._validation_buffer

    @property
    def test_predictions(self) -> List[Dict]:
        return self._test_buffer

    def on_validation_epoch_start(self) -> None:  # noqa: D401
        self._validation_buffer.clear()

    def on_test_epoch_start(self) -> None:  # noqa: D401
        self._test_buffer.clear()
