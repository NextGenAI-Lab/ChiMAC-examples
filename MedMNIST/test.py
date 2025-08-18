import argparse
import threading
import time
from inspect import signature
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import wandb
import yaml
from chimac import ChiMAC, Logger, seed_all
from chimac.utils import (
    op_brightness,
    op_contrast,
    op_flip,
    op_hue,
    op_rotate,
    op_saturation,
    op_scale,
    op_translate,
)
from medmnist import INFO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# --------------------------
# System monitoring thread
# --------------------------
gpu_stats: List[Tuple[float | None, float | None]] = []
cpu_stats: List[Tuple[float, float]] = []

try:
    import pynvml
except ImportError:
    pynvml = None


def monitor_system(interval: float):
    """Background thread to sample CPU/GPU usage."""

    if pynvml:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        have_nvml = True
    else:
        handle = None
        have_nvml = False

    while True:
        cpu_util = psutil.cpu_percent(interval=None)
        mem_util = psutil.virtual_memory().percent
        if have_nvml and handle and pynvml:
            try:
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                gpu_mem_util = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / 1e6
            except Exception:
                gpu_util, gpu_mem_util = None, None
        else:
            gpu_util, gpu_mem_util = None, None

        cpu_stats.append((cpu_util, mem_util))
        gpu_stats.append((gpu_util, gpu_mem_util))
        time.sleep(interval)


# --------------------------
# Helpers
# --------------------------
def make_log_filename(cfg: dict) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    aug_tag = "_aug" if "augmentation" in cfg else ""
    return f"{stamp}_{cfg['dataset']['name']}-{cfg['training']['model']}-{cfg['dataset']['img_size']}{aug_tag}.log"


def build_model(model_name: str, num_classes: int, device: torch.device, grayscale: bool = False) -> nn.Module:
    if not hasattr(models, model_name):
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    model_fn = getattr(models, model_name)

    # initialize model
    if "num_classes" in signature(model_fn).parameters:
        model = model_fn(num_classes=num_classes)
    else:
        model = model_fn()

        # replace classifier / fc head
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            else:
                try:
                    last_idx = len(model.classifier) - 1
                    in_feat = model.classifier[last_idx].in_features
                    model.classifier[last_idx] = nn.Linear(in_feat, num_classes)
                except Exception:
                    pass

    # if grayscale, adjust first conv
    if grayscale:
        first_layer_name = None
        # common naming in torchvision models
        for name in ["conv1", "features.0"]:
            if hasattr(model, name):
                first_layer_name = name
                break

        if first_layer_name is None:
            raise RuntimeError("Couldn't find first conv layer to modify for grayscale input")

        first_conv = getattr(model, first_layer_name)
        if isinstance(first_conv, nn.Conv2d) and first_conv.in_channels == 3:
            # create new conv with 1 channel
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size, # type: ignore
                stride=first_conv.stride, # type: ignore
                padding=first_conv.padding, # type: ignore
                bias=first_conv.bias is not None,
            )
            # optional: average pretrained RGB weights for grayscale
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)
            setattr(model, first_layer_name, new_conv)

    return model.to(device)


def choose_optimizer(optimizer_name: str, parameters, lr: float):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name in ("sgd", "nesterov"):
        return optim.SGD(
            parameters, lr=lr, momentum=0.9, nesterov=(optimizer_name == "nesterov")
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int], y_score: np.ndarray, average="macro"
):
    """Returns dict of common metrics; y_score is (N, num_classes) probs."""
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    results["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["precision_micro"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    results["recall_micro"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # multiclass AUC (try; may fail for single-class scenarios)
    try:
        # if shape mismatch, raise
        if y_score is not None and y_score.shape[1] > 1:
            y_true_onehot = np.eye(y_score.shape[1])[y_true]
            results["roc_auc_macro"] = roc_auc_score(
                y_true_onehot, y_score, average="macro", multi_class="ovr"
            )
            results["roc_auc_micro"] = roc_auc_score(
                y_true_onehot, y_score, average="micro", multi_class="ovr"
            )
        else:
            results["roc_auc_macro"] = None
            results["roc_auc_micro"] = None
    except Exception:
        results["roc_auc_macro"] = None
        results["roc_auc_micro"] = None

    return results


# --------------------------
# Train / Evaluate
# --------------------------
def evaluate_model(
    model, dataloader, device, criterion=None
) -> Tuple[float, List[int], List[int], np.ndarray]:
    model.eval()
    loss_sum = 0.0
    y_true = []
    y_pred = []
    y_score_list = []
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            if criterion is not None:
                loss = criterion(outputs, targets)
                loss_sum += float(loss.item()) * preds.shape[0]

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score_list.append(probs)
            total += preds.shape[0]

    avg_loss = loss_sum / total if (criterion is not None and total > 0) else 0.0
    y_score = np.vstack(y_score_list) if y_score_list else np.zeros((len(y_true), 1))
    return avg_loss, y_true, y_pred, y_score


def train_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    criterion,
    optimizer,
    cfg,
    logger,
):
    train_metrics = None
    val_metrics = None
    best_val_loss = float("inf")
    patience = cfg["training"].get("patience", 5)
    patience_counter = 0
    epochs = cfg["training"]["epochs"]
    log_interval = cfg["logging"].get("log_interval", 10) if cfg.get("logging") else 10
    use_wandb = bool(cfg.get("logging"))

    # optionally set scheduler
    scheduler = None
    if cfg["training"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg["training"].get("scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"].get("step_size", 5),
            gamma=cfg["training"].get("gamma", 0.1),
        )

    # training epochs
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        all_targets = []
        all_preds = []
        all_scores = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            batch_count += inputs.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_scores.append(probs)

            if batch_idx % log_interval == 0:
                avg_batch_loss = running_loss / batch_count
                train_acc = (
                    accuracy_score(all_targets, all_preds) if all_targets else 0.0
                )
                logger.info(
                    f"[Epoch {epoch}] Batch {batch_idx} train_loss={avg_batch_loss:.4f} train_acc={train_acc:.4f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/batch_loss": avg_batch_loss,
                            "train/batch_acc": train_acc,
                            "epoch": epoch,
                        }
                    )

        # epoch-level train metrics
        train_loss = running_loss / batch_count if batch_count > 0 else 0.0
        train_scores = (
            np.vstack(all_scores) if all_scores else np.zeros((len(all_preds), 1))
        )
        train_metrics = compute_classification_metrics(
            all_targets, all_preds, train_scores
        )

        # validation
        val_loss, val_y_true, val_y_pred, val_scores = evaluate_model(
            model, val_loader, device, criterion
        )
        val_metrics = compute_classification_metrics(val_y_true, val_y_pred, val_scores)

        # Log epoch metrics
        logger.info(
            f"Epoch {epoch} summary: train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_metrics["accuracy"],
                    "train/f1_macro": train_metrics["f1_macro"],
                    "val/loss": val_loss,
                    "val/acc": val_metrics["accuracy"],
                    "val/f1_macro": val_metrics["f1_macro"],
                    "val/roc_auc_macro": val_metrics.get("roc_auc_macro"),
                }
            )

        # early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # ckpt_path = Path("best_model.pth")
            # torch.save(model.state_dict(), ckpt_path)
            # logger.info(f"Saved best checkpoint to {ckpt_path}")
        else:
            patience_counter += 1
            logger.info(f"Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        if scheduler is not None:
            scheduler.step()

    # After training, evaluate best model on test set
    if Path("best_model.pth").exists():
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        logger.info("Loaded best_model.pth for final evaluation.")

    test_loss, test_y_true, test_y_pred, test_scores = evaluate_model(
        model, test_loader, device, criterion
    )
    test_metrics = compute_classification_metrics(test_y_true, test_y_pred, test_scores)

    # classification report string
    cls_report = classification_report(test_y_true, test_y_pred, zero_division=0)

    # log final metrics
    logger.info(
        f"Final Test - loss: {test_loss:.4f} acc: {test_metrics['accuracy']:.4f} f1_macro: {test_metrics['f1_macro']:.4f} roc_auc_macro: {test_metrics.get('roc_auc_macro')}"
    )
    logger.info("Classification Report:\n" + str(cls_report))

    if use_wandb:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/acc": test_metrics["accuracy"],
                "test/f1_macro": test_metrics["f1_macro"],
                "test/roc_auc_macro": test_metrics.get("roc_auc_macro"),
                "classification_report": cls_report,
            }
        )

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_report": cls_report,
    }


# --------------------------
# Main
# --------------------------
def main(cfg: dict):
    # prepare logs dir
    Path("logs").mkdir(exist_ok=True)

    log_filename = make_log_filename(cfg)
    logger = Logger(log_dir="logs", filename=log_filename)
    logger.catch_all_exceptions()

    logger.info("Starting benchmark")
    logger.info(f"Config: {cfg}")

    # seed
    seed_all(
        (
            int(cfg["augmentation"]["seed"])
            if "augmentation" in cfg and "seed" in cfg["augmentation"]
            else 42
        ),
        logger=logger,
        deterministic=False,
    )

    # device
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # WandB
    use_wandb = bool(cfg.get("logging"))
    if use_wandb:
        wandb.login()
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"].get("wandb_entity"),
            name=log_filename,
            config=cfg,
        )
        logger.info("WandB initialized")

    # dataset info
    info = INFO[cfg["dataset"]["name"]]
    DataClass = getattr(
        __import__("medmnist", fromlist=[info["python_class"]]), info["python_class"]
    )
    num_classes = len(info["label"])
    logger.info(f"Dataset class: {info['python_class']} num_classes={num_classes}")

    # transforms & augmentation
    if "augmentation" in cfg:
        aug_ops = [globals()[op]() for op in cfg["augmentation"]["ops"]]
        chimac = ChiMAC(
            aug_ops,
            k=cfg["augmentation"]["k"],
            alpha=cfg["augmentation"]["alpha"],
            seed=cfg["augmentation"]["seed"],
        )
        transform_train = T.Compose(
            [T.Lambda(lambda img: chimac.augment(img)), T.ToTensor()]
        )
        logger.info(f"Using ChiMAC with ops={cfg['augmentation']['ops']}")
    else:
        transform_train = T.Compose([T.ToTensor()])
        logger.info("No augmentation; using ToTensor for train.")

    transform_eval = T.Compose([T.ToTensor()])

    # load datasets (medmnist supports size kwarg in many dataset classes)
    size_kw = cfg["dataset"]["img_size"]
    train_dataset = DataClass(
        split="train", transform=transform_train, download=True, size=size_kw
    )
    val_dataset = DataClass(
        split="val", transform=transform_eval, download=True, size=size_kw
    )
    test_dataset = DataClass(
        split="test", transform=transform_eval, download=True, size=size_kw
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # are images grayscale
    grayscale = cfg["dataset"].get("grayscale", False)

    # model, optimizer, loss
    model = build_model(
        cfg["training"]["model"], num_classes=num_classes, device=device, grayscale=grayscale
    )
    logger.info(
        f"Built model {cfg['training']['model']} with {sum(p.numel() for p in model.parameters())} params"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(
        cfg["training"].get("optimizer", "adam"),
        model.parameters(),
        cfg["training"]["lr"],
    )

    # start system monitor thread
    monitor_thread = threading.Thread(
        target=monitor_system,
        args=(cfg["system_monitor"]["interval_sec"],),
        daemon=True,
    )
    monitor_thread.start()
    logger.info("Started system monitor thread")

    # train + evaluate
    results = train_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        criterion,
        optimizer,
        cfg,
        logger,
    )

    # summarize system stats and log
    if cpu_stats:
        cpu_avg = float(np.mean([c for c, m in cpu_stats]))
        mem_avg = float(np.mean([m for c, m in cpu_stats]))
    else:
        cpu_avg = mem_avg = None

    # GPU averages ignoring None entries
    gpu_utils = [g for g, m in gpu_stats if g is not None] if gpu_stats else []
    gpu_mems = [m for g, m in gpu_stats if m is not None] if gpu_stats else []
    gpu_util_avg = float(np.mean(gpu_utils)) if gpu_utils else None
    gpu_mem_avg = float(np.mean(gpu_mems)) if gpu_mems else None

    summary = {
        "cpu_avg_util": cpu_avg,
        "cpu_avg_mem": mem_avg,
        "gpu_avg_util": gpu_util_avg,
        "gpu_avg_mem_mb": gpu_mem_avg,
    }
    logger.info(f"System summary: {summary}")

    if use_wandb:
        wandb.log(summary)
        wandb.finish()

    logger.info("Benchmark finished")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
