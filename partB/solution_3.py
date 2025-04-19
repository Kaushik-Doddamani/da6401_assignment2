import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Ensure project_root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common_utils import set_seeds, get_configs
from data.data_loader import get_train_val_data_loaders, get_test_data_loader
from utils.model_utils import train_one_epoch, validate_one_epoch, evaluate_resnet_on_test

os.environ["CUDA_VISIBLE_DEVICES"] = "3"   # select your GPU


# ------------------------------------------------------------------
# 2) MODEL BUILDING (freeze/unfreeze)
# ----------------------------------------------------------------
def build_finetune_resnet50(num_classes, freeze_until_layer=3):
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze from layer4 upwards
    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    for ln in layer_names[freeze_until_layer:]:
        for p in getattr(model, ln).parameters():
            p.requires_grad = True
    # replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------------------------------------------
# 4) MAIN TRAIN LOOP
# ------------------------------------------------------------------
def main(static_config):
    model_config   = static_config["model_config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    set_seeds(model_config["seed"])
    train_dl, val_dl, class_names = get_train_val_data_loaders(
        data_root = static_config["data_root"],
        img_size  = model_config["resize_dim"],
        batch_size= model_config["batch_size"],
        val_ratio = model_config["val_ratio"],
        seed      = model_config["seed"]
    )
    num_classes = len(class_names)
    print(f"Loaded data: {num_classes} classes")

    # W&B initialization
    wandb.init(
        project   = static_config["wandb_project"],
        name      = static_config["wandb_run_name"],
        config    = {**model_config, "num_classes": num_classes}
    )
    wandb.watch_called = False
    # watch model once it's on device:
    model = build_finetune_resnet50(num_classes,
                                    freeze_until_layer=model_config["freeze_until_layer"])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Model loaded: {model.__class__.__name__} ({num_classes} classes)")
    wandb.watch(model, log="all", log_freq=100)


    # --- optimizer with discriminative LRs for head & layer4 only ---
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    head_params   = list(base_model.fc.parameters())
    layer4_params = list(base_model.layer4.parameters())

    optimizer = optim.AdamW([
        {"params": head_params,   "lr": 1e-3},
        {"params": layer4_params, "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    patience      = model_config["patience"]
    best_val_acc  = -float("inf")
    patience_cnt  = 0
    os.makedirs(static_config["output_dir"], exist_ok=True)
    ckpt_path     = os.path.join(static_config["output_dir"], "partB_Q3_best_resnet50.pth")

    UNFREEZE_EPOCH = 4
    epochs         = model_config["epochs"]

    # Training loop
    for epoch in range(1, epochs+1):
        # staged unfreeze of layer3
        if epoch == UNFREEZE_EPOCH:
            print(f">> Unfreezing layer3 at epoch {epoch}")
            layer3 = base_model.layer3
            for p in layer3.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(layer3.parameters()),
                "lr": 1e-4
            })

        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss,   val_acc   = validate_one_epoch(model, val_dl,   criterion, device)
        scheduler.step()

        print(f"Epoch: {epoch:02d}/{epochs}]  Train_Loss: {train_loss:.3f} Train_Accuracy: {train_acc:.3f} | "
              f"Val_Loss: {val_loss:.3f} Val_Accuracy: {val_acc:.3f}")
        
        #  log to W&B
        wandb.log({
            "PartB_Q3_epoch":         epoch,
            "PartB_Q3_train_loss":    train_loss,
            "PartB_Q3_train_acc":     train_acc,
            "PartB_Q3_val_loss":      val_loss,
            "PartB_Q3_val_acc":       val_acc,
            "PartB_Q3_lr_head":       optimizer.param_groups[0]["lr"],
            "PartB_Q3_lr_layer4":     optimizer.param_groups[1]["lr"],
        })

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> New best val‑acc {best_val_acc:.3f} (checkpoint saved)")
        else:
            patience_cnt += 1
            print(f"  >> No improvement ({patience_cnt}/{patience})")
            if patience_cnt >= patience:
                print("Early stopping.")
                break

    # load best & final test eval
    print(f"Training done. Best val‑acc = {best_val_acc:.3f}")

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(ckpt_path))

    print("Getting test data loader...")
    test_loader = get_test_data_loader(static_config["data_root"], model_config["resize_dim"], model_config["batch_size"])

    print("Evaluating on test set...")
    y_true, y_pred, test_acc = evaluate_resnet_on_test(model, test_loader, device)
    print(f"Test accuracy = {test_acc*100:.2f}%")
    
    # log final metrics and plot
    wandb.log({"PartB_Q3_test_acc": test_acc})
    cm = wandb.plot.confusion_matrix(
        probs    = None,
        y_true   = y_true,
        preds    = y_pred,
        class_names = class_names
    )
    wandb.log({"PartB_Q3_confusion_matrix": cm})

if __name__ == "__main__":
    config = get_configs(project_root, "configs.yaml")["part_b_configs"]["solution_3_configs"]
    main(config)
