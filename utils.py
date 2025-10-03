import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torch import nn
from typing import Tuple, Callable
from pathlib import Path
import open_clip


DPATH_VALID = Path("/data/ai/ref-data/image/ImageNet/imagenet1k/Data/CLS-LOC/val")

def init_dataloader(
    dpath_valid: Path, 
    img_pp:      Callable, 
    batch_size:  int, 
    n_workers:   int,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a class-labeled directory.

    Args:
        dpath_valid --- Directory path to the validation data
        img_pp -------- Image preprocessor
        batch_size ---- Number of samples per mini-batch
        n_workers ----- Number of worker processes used for background data loading

    Returns:
        Data loader
    """
    dataset    = ImageFolder(dpath_valid, transform=img_pp)
    dataloader = DataLoader(
        dataset, 
        batch_size =batch_size, 
        num_workers=n_workers, 
        pin_memory =True,
    )

    return dataloader

def batch_prec1(
    logits: torch.tensor, 
    targs:  torch.Tensor,
) -> torch.Tensor:
    """
    Compute batch Top-1 Precision given logits and targets.

    Args:
        logits --- Class scores of shape (B, L)
        targs ---- Targets

    Returns:
        Scalar tensor of mean Prec@1 over mini-batch
    """
    preds      = logits.argmax(dim=1)
    prec1_mean = (preds == targs).float().mean()

    return prec1_mean

def init_resnet50(
    device: torch.device,
) -> Tuple[nn.Module, Callable]:
    """
    Load pretrained ResNet-50 and corresponding image preprocessor.

    Args:
        device --- Device to put model on

    Returns:
        ResNet-50 and corresponding image preprocessor
    """
    model_wts = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    img_pp    = model_wts.transforms()
    model     = torchvision.models.resnet50(weights=model_wts).to(device).eval()

    return model, img_pp

def init_vlm(
    model_id:   str, 
    pretrained: str, 
    quick_gelu: bool, 
    device:     torch.device,
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Load pretrained VLM and corresponding image preprocessor + tokenizer.

    Args:
        model_id ----- open_clip-native model identifier
        pretrained --- Pretraining dataset
        quick_gelu --- Whether to initialize the VLM with QuickGeLU (True) or GeLU (False) activations
        device ------- Device to put model on

    Returns:
        VLM and corresponding image preprocessor + tokenizer
    """
    model, _, img_pp = open_clip.create_model_and_transforms(
        model_id, 
        pretrained=pretrained, 
        quick_gelu=quick_gelu, 
        device    =device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_id)

    return model, img_pp, tokenizer

def print_eval_header(
    model_name: str,
) -> None:
    """
    Print a centered evaluation header for a given model name.

    Args:
        model_name --- Model display name
    """
    print(
        f"{' ' + model_name + ' ':=^{90}}",
        f"",
        sep="\n"
    )
