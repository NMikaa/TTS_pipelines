"""
Pocket TTS Trainer - SOTA training pipeline.

Fine-tunes Pocket TTS FlowLM on pre-computed latents using LSD loss.
Based on the CALM paper architecture.

Training flow:
    1. Load pre-computed latents [B, S, 32] (in Mimi quantizer space)
    2. Normalize to flow space: (latents - emb_mean) / emb_std
    3. Teacher forcing: input = [BOS, x_0, ..., x_{S-2}], target = [x_0, ..., x_{S-1}]
    4. Project through input_linear: [B, S, 32] -> [B, S, 1024]
    5. Prepend text embeddings: [B, T+S, 1024]
    6. Transformer with causal masking -> [B, S, 1024] conditioning
    7. Flow matching + LSD loss on conditioning vs target latents
"""

import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pocket_tts_training.config import TrainingConfig
from pocket_tts_training.lsd_loss import LSDLoss
from pocket_tts_training.dataset import create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (save originals first)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def _set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PocketTTSTrainer:
    """Trainer for fine-tuning Pocket TTS with LSD loss."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        _set_seed(config.seed)

        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize components in order
        self._init_models()
        self._init_data()
        self._init_optimizer()
        self._init_loss()
        self._init_amp()
        self._init_ema()
        self._init_logging()

        self.global_step = 0
        self.epoch = 0

    def _init_models(self):
        """Load pretrained model and configure for training."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "pocket-tts"))
        from pocket_tts.models.tts_model import TTSModel
        from pocket_tts.modules.stateful_module import init_states
        self._init_states = init_states

        logger.info("Loading pretrained Pocket TTS model...")
        self.tts_model = TTSModel.load_model(config=self.config.pretrained_model_variant)

        # Move FlowLM to GPU
        self.flow_lm = self.tts_model.flow_lm.to(self.device)
        self.flow_net = self.flow_lm.flow_net
        self.transformer = self.flow_lm.transformer
        self.text_conditioner = self.flow_lm.conditioner

        # Load normalization stats from pre-computed latents
        stats_path = Path(self.config.latents_dir) / "normalization_stats.pt"
        if stats_path.exists():
            stats = torch.load(stats_path, map_location=self.device, weights_only=True)
            self.flow_lm.emb_mean.data.copy_(stats["emb_mean"].to(self.device))
            self.flow_lm.emb_std.data.copy_(stats["emb_std"].to(self.device))
            logger.info("Loaded normalization stats from pre-computed latents")

        # Freeze: only train FlowLM components
        # Mimi encoder/decoder stays on CPU and is not used during training
        for param in self.flow_lm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.flow_lm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.flow_lm.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    def _get_param_groups(self):
        """Separate parameters for weight decay (no decay on biases, norms, embeddings)."""
        decay_params = []
        no_decay_params = []

        for name, param in self.flow_lm.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name or "emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _init_data(self):
        """Initialize data loaders."""
        logger.info("Creating data loaders...")
        self.train_loader = create_dataloader(
            latents_dir=self.config.latents_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            max_frames=int(self.config.max_audio_seconds * 12.5),
            use_bucketing=True,
        )
        steps_per_epoch = len(self.train_loader)
        if self.config.max_steps > 0:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {self.total_steps}")

    def _init_optimizer(self):
        """Initialize optimizer and LR scheduler."""
        param_groups = self._get_param_groups()
        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        min_lr_ratio = self.config.min_lr / self.config.learning_rate
        self.scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=min_lr_ratio,
        )

    def _init_loss(self):
        """Initialize LSD loss."""
        self.lsd_loss = LSDLoss(fm_ratio=self.config.fm_ratio).to(self.device)

    def _init_amp(self):
        """Initialize automatic mixed precision."""
        self.scaler = GradScaler("cuda") if self.config.mixed_precision else None
        self.amp_dtype = self.config.amp_dtype

    def _init_ema(self):
        """Initialize EMA of model weights."""
        self.ema = EMAModel(self.flow_lm, decay=self.config.ema_decay)

    def _init_logging(self):
        """Initialize tensorboard logging."""
        log_dir = Path(self.config.output_dir) / "runs" / self.config.experiment_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def _tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        """Tokenize text strings using the model's tokenizer.

        Returns: [B, max_len] token tensor on device.
        """
        all_tokens = []
        max_len = 0
        for text in texts:
            tok_result = self.text_conditioner.tokenizer(text)
            tokens = tok_result.tokens[0]  # [L]
            all_tokens.append(tokens)
            max_len = max(max_len, tokens.shape[0])

        padded = torch.zeros(len(texts), max_len, dtype=torch.long, device=self.device)
        for i, tokens in enumerate(all_tokens):
            padded[i, : tokens.shape[0]] = tokens.to(self.device)

        return padded

    def _compute_conditioning(
        self,
        latents_flow: torch.Tensor,
        text_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run backbone transformer to get conditioning vectors.

        Teacher forcing: input is [BOS, x_0, ..., x_{S-2}], predicts [x_0, ..., x_{S-1}].

        Args:
            latents_flow: [B, S, 32] latents in flow space (normalized)
            text_tokens: [B, T] text token IDs
            mask: [B, S] boolean mask for valid latent positions

        Returns:
            conditioning: [B, S, d_model] transformer outputs
            target_mask: [B, S] mask for positions with valid targets
        """
        B, S, D = latents_flow.shape

        # Teacher forcing: shift input right by 1, prepend BOS
        bos = self.flow_lm.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, D)
        shifted_input = torch.cat([bos, latents_flow[:, :-1, :]], dim=1)  # [B, S, D]

        # Project to transformer dimension
        transformer_input = self.flow_lm.input_linear(shifted_input)  # [B, S, d_model]

        # Get text embeddings
        from pocket_tts.conditioners.text import TokenizedText
        text_emb = self.text_conditioner(TokenizedText(text_tokens))  # [B, T, d_model]
        T = text_emb.shape[1]

        # Concatenate: [text_emb | transformer_input]
        combined = torch.cat([text_emb, transformer_input], dim=1)  # [B, T+S, d_model]

        # Initialize model state for transformer (StatefulModule requires it)
        model_state = self._init_states(
            self.flow_lm, batch_size=B, sequence_length=T + S
        )

        # Forward through transformer with causal masking
        transformer_out = self.transformer(combined, model_state)

        # Apply output norm
        transformer_out = self.flow_lm.out_norm(transformer_out)

        # Get outputs for audio positions only (remove text prefix)
        conditioning = transformer_out[:, T:, :]  # [B, S, d_model]

        return conditioning, mask

    def train_step(self, batch: dict) -> dict:
        """Single training step with gradient accumulation support.

        Returns: metrics dict
        """
        latents = batch["latents"].to(self.device)  # [B, S, 32]
        mask = batch["mask"].to(self.device)         # [B, S]
        texts = batch["texts"]

        B, S, D = latents.shape

        # Normalize latents to flow space
        latents_flow = (latents - self.flow_lm.emb_mean) / (self.flow_lm.emb_std + 1e-8)

        # Tokenize text
        text_tokens = self._tokenize_texts(texts)

        with autocast("cuda", enabled=self.config.mixed_precision, dtype=self.amp_dtype):
            # Get conditioning from backbone transformer
            conditioning, target_mask = self._compute_conditioning(
                latents_flow, text_tokens, mask
            )

            # Flatten for loss: [B*S, dim] with masking
            target_flat = latents_flow.reshape(B * S, D)  # [B*S, 32]
            cond_flat = conditioning.reshape(B * S, -1)   # [B*S, d_model]
            mask_flat = target_mask.reshape(B * S).float() # [B*S]

            # Head batch multiplier: reuse conditioning for multiple noise samples
            hbm = self.config.head_batch_multiplier
            if hbm > 1:
                target_flat = target_flat.repeat(hbm, 1)
                cond_flat = cond_flat.repeat(hbm, 1)
                mask_flat = mask_flat.repeat(hbm)

            # Compute LSD loss
            loss, metrics = self.lsd_loss(
                flow_net=self.flow_net,
                x_data=target_flat,
                conditioning=cond_flat,
                mask=mask_flat,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return metrics

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.flow_lm.parameters(), self.config.gradient_clip
        )

        # Skip step if gradients are NaN/Inf to avoid corrupting weights
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if not math.isfinite(grad_norm_val):
            logger.warning(f"Non-finite grad_norm ({grad_norm_val:.4f}), skipping optimizer step")
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.update()
            return grad_norm_val

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        # EMA update
        if self.global_step >= self.config.ema_start_step:
            self.ema.update(self.flow_lm)

        return grad_norm_val

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.flow_lm.train()
        epoch_metrics = {}
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            # Gradient accumulation
            do_step = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
            if do_step:
                grad_norm = self._optimizer_step()
                self.global_step += 1
                metrics["grad_norm"] = grad_norm
                metrics["lr"] = self.scheduler.get_last_lr()[0]

                # Log to tensorboard
                if self.global_step % self.config.log_every == 0:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

            # Update progress bar
            if batch_idx % self.config.log_every == 0:
                avg_loss = sum(epoch_metrics.get("total_loss", [0])) / max(
                    len(epoch_metrics.get("total_loss", [1])), 1
                )
                postfix = {"loss": f"{avg_loss:.4f}", "step": self.global_step}
                if "lr" in metrics:
                    postfix["lr"] = f"{metrics['lr']:.2e}"
                pbar.set_postfix(postfix)

            # Check max steps
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Config: {self.config}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")

        start_epoch = self.epoch + 1 if self.epoch > 0 else 0
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch(epoch)

            logger.info(f"Epoch {epoch} completed:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")

            if self.config.save_every_epoch:
                self.save_checkpoint(f"epoch_{epoch}")

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        logger.info("Training completed!")
        self.save_checkpoint("final")
        self.writer.close()

    def save_checkpoint(self, name: str):
        """Save full training state."""
        path = Path(self.config.output_dir) / f"{name}.pt"
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "flow_lm_state_dict": self.flow_lm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "config": self.config,
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.flow_lm.load_state_dict(checkpoint["flow_lm_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(f"Resumed from {path} (step {self.global_step}, epoch {self.epoch})")
