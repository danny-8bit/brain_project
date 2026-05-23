import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from tqdm import tqdm

from .data.dataset import load_datalist
from .data.transforms import get_train_transforms, get_val_transforms
from .models.builder import build_model
from .losses import DiceBCELoss, DeepSupervisionLoss
from .metrics import BratsMetrics
from .utils import EMA, set_seed, get_logger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{cfg.get('gpu', 0)}" if torch.cuda.is_available() else "cpu"
        )

        set_seed(cfg.get("seed", 2024))

        self.work_dir = Path(cfg["work_dir"]) / cfg["model"] / f"fold_{cfg['fold']}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(self.work_dir / "train.log")
        self.writer = SummaryWriter(self.work_dir / "tb")

        self.logger.info(f"Config: {cfg}")

        self._setup_data()
        self._setup_model()

    # -------- setup --------
    def _setup_data(self):
        cfg = self.cfg
        train_files = load_datalist(cfg["folds_json"], cfg["fold"], "train")
        val_files = load_datalist(cfg["folds_json"], cfg["fold"], "val")

        self.logger.info(f"Train cases: {len(train_files)}, Val cases: {len(val_files)}")

        train_tf = get_train_transforms(cfg["roi_size"])
        val_tf = get_val_transforms()

        train_ds = CacheDataset(
            data=train_files,
            transform=train_tf,
            cache_rate=cfg.get("cache_rate", 0.0),
            num_workers=cfg["num_workers"],
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_tf,
            cache_rate=cfg.get("cache_rate", 0.0),
            num_workers=cfg["num_workers"],
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg["num_workers"] > 0,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
        )

    def _setup_model(self):
        cfg = self.cfg
        self.model = build_model(
            cfg["model"],
            in_channels=4,
            out_channels=3,
            roi_size=cfg["roi_size"],
            **{k: v for k, v in cfg.items() if k in (
                "init_filters", "dropout", "feature_size", "use_checkpoint"
            )},
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.logger.info(f"Model: {cfg['model']}, params: {n_params:.2f}M")

        # EMA
        self.ema = EMA(self.model, decay=0.999) if cfg.get("ema", True) else None

        # Loss
        base = DiceBCELoss(
            lambda_dice=cfg.get("lambda_dice", 1.0),
            lambda_bce=cfg.get("lambda_bce", 1.0),
            channel_weights=cfg.get("loss_channel_weights", [1.0, 1.0, 1.0]),
        )
        self.loss_fn = (
            DeepSupervisionLoss(base)
            if cfg.get("deep_supervision", True)
            else base
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        # Cosine schedule with warmup
        from torch.optim.lr_scheduler import LambdaLR
        warmup = cfg.get("warmup_epochs", 5)
        total = cfg["max_epochs"]

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            import math
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # AMP
        self.scaler = GradScaler(enabled=cfg.get("amp", True))

        # Metrics & postprocessing for validation
        self.metrics = BratsMetrics()
        self.post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])

        self.best_dice = 0.0
        self.best_epoch = -1
        self.start_epoch = 0

        if self.cfg.get("resume"):
            self._load_ckpt(self.cfg["resume"])

    # -------- train --------
    def train_one_epoch(self, epoch):
        self.model.train()
        loss_sum = 0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=100)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.get("amp", True)):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.scaler.scale(loss).backward()

            if self.cfg.get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return loss_sum / max(1, n)

    @torch.no_grad()
    @torch.no_grad()
    def validate(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        self.metrics.reset()

        for batch in tqdm(self.val_loader, desc="Val", ncols=100):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            def predictor(x):
                out = model(x)
                return out[0] if isinstance(out, (list, tuple)) else out

            with autocast(enabled=self.cfg.get("amp", True)):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=self.cfg["roi_size"],
                    sw_batch_size=self.cfg["sw_batch_size"],
                    predictor=predictor,
                    overlap=0.5,
                    mode="gaussian",
                )

            # === 关键修复：剥掉 MetaTensor 包装，转成普通 Tensor ===
            if hasattr(logits, "as_tensor"):
                logits = logits.as_tensor()
            if hasattr(labels, "as_tensor"):
                labels = labels.as_tensor()

            # sigmoid + 0.5 阈值 -> binary mask (region-based)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            # 不用 decollate_batch，直接按 batch 维度遍历
            # batch 形状: (B, 3, D, H, W)
            for p, t in zip(preds, labels):
                # 喂给 metrics: 期望 (3, D, H, W) 的 numpy/tensor
                self.metrics(
                    y_pred=[p.cpu().numpy()],
                    y=[t.cpu().numpy()],
                )

        return self.metrics.aggregate()

    def fit(self):
        cfg = self.cfg
        for epoch in range(self.start_epoch + 1, cfg["max_epochs"] + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr", lr, epoch)

            self.logger.info(
                f"Epoch {epoch}/{cfg['max_epochs']} | "
                f"loss={train_loss:.4f} | lr={lr:.2e} | "
                f"time={time.time()-t0:.1f}s"
            )

            if epoch % cfg["val_interval"] == 0 or epoch == cfg["max_epochs"]:
                model_to_eval = self.ema.ema_model if self.ema else self.model
                r = self.validate(model_to_eval)
                self.logger.info(
                    f"[Val ep={epoch}] mean={r['dice_mean']:.4f} | "
                    f"TC={r['dice_tc']:.4f} WT={r['dice_wt']:.4f} ET={r['dice_et']:.4f}"
                )
                for k, v in r.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                if r["dice_mean"] > self.best_dice:
                    self.best_dice = r["dice_mean"]
                    self.best_epoch = epoch
                    self._save_ckpt("best.pt", epoch, r)
                    self.logger.info(f"** Best dice={self.best_dice:.4f} at epoch {epoch} **")

            self._save_ckpt("last.pt", epoch, {})

        self.logger.info(
            f"Done. Best dice={self.best_dice:.4f} at epoch {self.best_epoch}"
        )
        self.writer.close()

    # -------- ckpt --------
    def _save_ckpt(self, name, epoch, metrics):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "metrics": metrics,
            "cfg": self.cfg,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, self.work_dir / name)

    def _load_ckpt(self, path):
        s = torch.load(path, map_location=self.device)
        self.model.load_state_dict(s["model"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.best_dice = s.get("best_dice", 0)
        self.start_epoch = s.get("epoch", 0)
        if self.ema is not None and "ema" in s:
            self.ema.load_state_dict(s["ema"])
        self.logger.info(f"Loaded checkpoint from {path}, start_epoch={self.start_epoch}")