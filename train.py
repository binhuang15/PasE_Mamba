import os
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from data_paths import resolve_train_npy_path, resolve_validation_npy_path
from datagenerator import MyDatasetLoader
from augmentation_strategies import get_train_transform_2D
from vision_mamba import PasEMamba
from mamba_sys import SS2D
from edl_utils import edl_nll_kl_loss, dice_ce_loss, kl_consistency
import random
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


def set_anisotropic_fusion(model: nn.Module, enabled: bool = True):
    """Toggle EASF (Evidence-driven Anisotropic Scan Fusion) driven by DPE (Decoupled Prediction-Evidence) maps on all SS2D blocks."""
    num_blocks = 0
    for m in model.modules():
        if isinstance(m, SS2D):
            m.use_anisotropic_fusion = enabled
            num_blocks += 1
    print(f"[AnisoFusion] enabled={enabled}, ss2d_blocks={num_blocks}")


def DiceLoss(labels, outputs):
    dice = 2 * torch.sum(labels * outputs) / (torch.sum(labels) + torch.sum(outputs))
    return 1 - dice

def seg_loss(labels, outputs):
    ce_loss = F.cross_entropy(
        outputs,
        labels[:,0,:,:].long(),
        weight=None,
        ignore_index=0,
        reduction="mean",
        label_smoothing=0.0,
    )
    labels_one_hot = F.one_hot(labels[:,0,:,:].long(), num_classes=3).float()

    labels_one_hot = labels_one_hot.permute([0, 3, 1, 2])
    outputs = F.softmax(outputs, dim=1)
    dice_loss = DiceLoss(labels_one_hot[:, 1:2, :, :], outputs[:, 1:2, :, :]) * 0.5 + DiceLoss(labels_one_hot[:, 2:3, :, :], outputs[:, 2:3, :, :]) * 0.5

    loss = dice_loss + ce_loss
    return loss

def dsc_calc(labels, outputs):
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.round(outputs)
    labels_one_hot = F.one_hot(labels[:,0,:,:].long(), num_classes=3).float()
    labels_one_hot = labels_one_hot.permute([0, 3, 1, 2])
    dsc_list = []
    for i in range(3):
        dsc_i = 2 * (torch.sum(outputs[:, i, :, :] * labels_one_hot[:, i, :, :])) / (torch.sum(outputs[:, i, :, :]) + torch.sum(labels_one_hot[:, i, :, :]))
        dsc_list.append(dsc_i.item())

    return dsc_list


def train_process(model, optimizer, datasetloader, device, refine_with_prob: bool = False):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    dsc_list = [], [], []
    dsc_second_list = [], [], []
    counter = 0
    for j, (imageData, maskData, classLabel, patient_name) in enumerate(datasetloader):
        inputs = imageData.to(device)
        masks = maskData.to(device)
        labels = classLabel.to(device)

        inputs = inputs.float()

        predict_mask = model(inputs.clone(), refine_with_prob=refine_with_prob)

        loss_seg = dice_ce_loss(predict_mask["logits"], masks[:,0,:,:].long())
        loss_edl, _ = edl_nll_kl_loss(predict_mask["alpha"], masks[:,0,:,:].long(), kl_weight=1e-3)
        loss_c = kl_consistency(predict_mask["prob_edl"], predict_mask["prob_seg"], symmetric=True)

        loss = loss_seg + 0.2 * loss_edl + 0.05 * loss_c
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        dsc_all = dsc_calc(masks, predict_mask["logits"])

        dsc_list[0].append(dsc_all[0])
        dsc_list[1].append(dsc_all[1])
        dsc_list[2].append(dsc_all[2])

        predict_mask = model(inputs.clone(), refine_with_prob=refine_with_prob)

        loss_seg = dice_ce_loss(predict_mask["logits"], masks[:,0,:,:].long())
        loss_edl, _ = edl_nll_kl_loss(predict_mask["alpha"], masks[:,0,:,:].long(), kl_weight=1e-3)
        loss_c = kl_consistency(predict_mask["prob_edl"], predict_mask["prob_seg"], symmetric=True)

        loss = loss_seg + 0.2 * loss_edl + 0.05 * loss_c
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        dsc_all = dsc_calc(masks, predict_mask["logits"])

        dsc_second_list[0].append(dsc_all[0])
        dsc_second_list[1].append(dsc_all[1])
        dsc_second_list[2].append(dsc_all[2])

        ## init runtime attention
        model.backbone._runtime_attn = None
        if hasattr(model.backbone, "_fusion_u"):
            model.backbone._fusion_u = None

        total_loss += loss.item() * inputs.size(0)

        counter += 1

    avg_loss = total_loss / (counter * labels.size(0))

    return model, avg_loss, dsc_list, dsc_second_list

def val_process(model, datasetloader, device, refine_with_prob: bool = False):
    with torch.no_grad():
        model.eval()

        total_loss = 0
        dsc_list = [], [], []
        dsc_second_list = [], [], []
        counter = 0
        for j, (imageData, maskData, classLabel, patient_name, _, _, _) in enumerate(datasetloader):
            inputs = imageData.to(device)
            masks = maskData.to(device)
            labels = classLabel.to(device)

            inputs = inputs.float()

            predict_mask = model(inputs.clone(), refine_with_prob=refine_with_prob)

            dsc_all = dsc_calc(masks, predict_mask["logits"])

            dsc_list[0].append(dsc_all[0])
            dsc_list[1].append(dsc_all[1])
            dsc_list[2].append(dsc_all[2])

            predict_mask = model(inputs.clone(), refine_with_prob=refine_with_prob)

            dsc_all = dsc_calc(masks, predict_mask["logits"])

            dsc_second_list[0].append(dsc_all[0])
            dsc_second_list[1].append(dsc_all[1])
            dsc_second_list[2].append(dsc_all[2])

            ## init runtime attention
            model.backbone._runtime_attn = None
            if hasattr(model.backbone, "_fusion_u"):
                model.backbone._fusion_u = None

            loss = dice_ce_loss(predict_mask["logits"], masks[:,0,:,:].long())

            total_loss += loss.item() * inputs.size(0)

            counter += 1

        avg_loss = total_loss / (counter * labels.size(0))

    return model, avg_loss, dsc_list, dsc_second_list



class PasE_Mamba_Config:
    def __init__(self):
        self.MODEL = {
            "VSSM": {
                "PATCH_SIZE": 4,
                "IN_CHANS": 1,
                "EMBED_DIM": 64,
                "DEPTHS": [2, 2, 2, 2],
            },
            "SWIN": {"MLP_RATIO": 4.0, "PATCH_NORM": True},
            "DROP_RATE": 0.0,
            "DROP_PATH_RATE": 0.2,
        }
        self.TRAIN = {"USE_CHECKPOINT": True}


def train(
    *,
    data_path: str,
    pwd_path: str,
    model_save_path: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    min_epoch_for_best: int | None = None,
) -> None:
    """Training entry: ``data_path``, ``pwd_path``, ``model_save_path`` must be supplied explicitly."""
    repo = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    USE_ANISOTROPIC_FUSION = True
    USE_REFINE_WITH_PROB = True

    dp = os.path.abspath(data_path if os.path.isabs(data_path) else os.path.join(repo, data_path))
    pw = os.path.abspath(pwd_path if os.path.isabs(pwd_path) else os.path.join(repo, pwd_path))
    ms = os.path.abspath(model_save_path if os.path.isabs(model_save_path) else os.path.join(repo, model_save_path))
    epoch = epochs if epochs is not None else 200
    batchsize = batch_size if batch_size is not None else 16
    lr_use = lr if lr is not None else 1e-3
    min_best = min_epoch_for_best if min_epoch_for_best is not None else 6

    patchsize = (224, 224)
    num_class = 3
    config = PasE_Mamba_Config()
    os.makedirs(ms, exist_ok=True)

    model_save = ms
    model_vision_mamba = PasEMamba(num_classes=num_class, config=config)

    model = model_vision_mamba.to(device)
    set_anisotropic_fusion(model, enabled=USE_ANISOTROPIC_FUSION)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] total={total_params:,}, trainable={trainable_params:,}")

    train_path = resolve_train_npy_path(dp)
    val_path = resolve_validation_npy_path(dp)

    img_trans = get_train_transform_2D(patchsize)
    dataset_train = MyDatasetLoader(train_path, pwd=pw, mode="train", transform=img_trans["val"], device=device)
    dataset_val = MyDatasetLoader(val_path, pwd=pw, mode="test", transform=img_trans["val"], device=device)
    trainloader = DataLoader(dataset=dataset_train, batch_size=batchsize, shuffle=True, drop_last=False)
    valloader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(params=model.parameters(), lr=lr_use)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, factor=0.9, min_lr=1e-6)

    history = []
    best_avg_valid_loss = 9999.0
    best_avg_valid_dsc = 0.0
    best_epoch = 0
    best_dsc_placenta = best_dsc_myometrium = 0.0
    best_dsc2_placenta = best_dsc2_myometrium = 0.0

    for epoch_i in range(epoch):
        epoch_start = time.time()
        model, avg_train_loss, train_dsc, train_dsc_second = train_process(
            model, optimizer, trainloader, device, refine_with_prob=USE_REFINE_WITH_PROB
        )
        model, avg_valid_loss, test_dsc, test_dsc_second = val_process(
            model, valloader, device, refine_with_prob=USE_REFINE_WITH_PROB
        )
        scheduler.step(avg_valid_loss)
        epoch_end = time.time()

        train_dsc_placenta = np.mean(train_dsc[1])
        train_dsc_myometrium = np.mean(train_dsc[2])
        test_dsc_placenta = np.mean(test_dsc[1])
        test_dsc_myometrium = np.mean(test_dsc[2])
        train_dsc2_placenta = np.mean(train_dsc_second[1])
        train_dsc2_myometrium = np.mean(train_dsc_second[2])
        test_dsc2_placenta = np.mean(test_dsc_second[1])
        test_dsc2_myometrium = np.mean(test_dsc_second[2])

        avg_valid_dsc = test_dsc2_placenta + test_dsc2_myometrium

        history.append(
            [
                avg_train_loss,
                avg_valid_loss,
                train_dsc_placenta,
                train_dsc_myometrium,
                test_dsc_placenta,
                test_dsc_myometrium,
                train_dsc2_placenta,
                train_dsc2_myometrium,
                test_dsc2_placenta,
                test_dsc2_myometrium,
            ]
        )

        ep1 = epoch_i + 1
        if ep1 >= min_best and best_avg_valid_dsc <= avg_valid_dsc:
            best_avg_valid_loss = avg_valid_loss
            best_avg_valid_dsc = avg_valid_dsc
            best_dsc_placenta = test_dsc_placenta
            best_dsc_myometrium = test_dsc_myometrium
            best_dsc2_placenta = test_dsc2_placenta
            best_dsc2_myometrium = test_dsc2_myometrium
            best_epoch = ep1
            torch.save(model, os.path.join(model_save, "Best_model.pt"))
            print(
                "Best Epoch: {:03d}, Validation: Loss: {:.4f}, Placenta DSC: {:.4f}, Myometrium DSC: {:.4f}".format(
                    best_epoch, best_avg_valid_loss, best_dsc_placenta, best_dsc_myometrium
                )
            )
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Placenta DSC: {:.4f}, Myometrium DSC: {:.4f}, "
            "Placenta DSC2: {:.4f}, Myometrium DSC2: {:.4f} \n\t\t\tValidation: Loss: {:.4f}, "
            "Placenta DSC: {:.4f}, Myometrium DSC: {:.4f}, Placenta DSC2: {:.4f}, Myometrium DSC2: {:.4f}, "
            "Time: {:.4f}s, \n\t\t\tlearning rate: {:.7f}".format(
                ep1,
                avg_train_loss,
                train_dsc_placenta,
                train_dsc_myometrium,
                train_dsc2_placenta,
                train_dsc2_myometrium,
                avg_valid_loss,
                test_dsc_placenta,
                test_dsc_myometrium,
                test_dsc2_placenta,
                test_dsc2_myometrium,
                epoch_end - epoch_start,
                optimizer.param_groups[0]["lr"],
            )
        )

    torch.save(history, os.path.join(model_save, "history.pt"))
    history_arr = np.array(history)
    fig1 = plt.figure()
    plt.plot(history_arr[:, 0])
    plt.plot(history_arr[:, 1])
    plt.legend(["Tr Loss", "Val Loss"])
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    fig1.savefig(os.path.join(model_save, "loss_curve.png"))
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(history_arr[:, 2])
    plt.plot(history_arr[:, 4])
    plt.legend(["Tr DSC", "Val DSC=" + str(np.round(best_dsc_placenta, decimals=4))])
    plt.xlabel("Epoch Number")
    plt.ylabel("DSC")
    fig2.savefig(os.path.join(model_save, "DSC_placenta_curve.png"))
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(history_arr[:, 3])
    plt.plot(history_arr[:, 5])
    plt.legend(["Tr DSC", "Val DSC=" + str(np.round(best_dsc_myometrium, decimals=4))])
    plt.xlabel("Epoch Number")
    plt.ylabel("DSC")
    fig3.savefig(os.path.join(model_save, "DSC_myometrium_curve.png"))
    plt.close(fig3)

    fig4 = plt.figure()
    plt.plot(history_arr[:, 6])
    plt.plot(history_arr[:, 8])
    plt.legend(["Tr DSC2", "Val DSC2=" + str(np.round(best_dsc2_placenta, decimals=4))])
    plt.xlabel("Epoch Number")
    plt.ylabel("DSC")
    fig4.savefig(os.path.join(model_save, "DSC2_placenta_curve.png"))
    plt.close(fig4)

    fig5 = plt.figure()
    plt.plot(history_arr[:, 7])
    plt.plot(history_arr[:, 9])
    plt.legend(["Tr DSC2", "Val DSC2=" + str(np.round(best_dsc2_myometrium, decimals=4))])
    plt.xlabel("Epoch Number")
    plt.ylabel("DSC")
    fig5.savefig(os.path.join(model_save, "DSC2_myometrium_curve.png"))
    plt.close(fig5)

    print(f"Training done. Best epoch={best_epoch}, weights: {os.path.join(model_save, 'Best_model.pt')}")


def main() -> None:
    p = argparse.ArgumentParser(description="PasE-Mamba (PasEMamba) training")
    p.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--data-path",
        required=True,
        help="Directory with train.npy / validation.npy (repo-relative or absolute)",
    )
    p.add_argument(
        "--pwd-path",
        required=True,
        help="Processed image root (subfolders e.g. 1-Normal, 2-PAS)",
    )
    p.add_argument(
        "--model-save",
        required=True,
        help="Output directory for Best_model.pt and training curves (e.g. TrainedCheckpoints)",
    )
    p.add_argument("--min-best-epoch", type=int, default=None, help="First epoch (1-based) eligible to save Best_model")
    args = p.parse_args()
    train(
        data_path=args.data_path,
        pwd_path=args.pwd_path,
        model_save_path=args.model_save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_epoch_for_best=args.min_best_epoch,
    )


if __name__ == "__main__":
    main()
