import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


def load_model(checkpoint_path, num_classes, device):

    weights = Swin3D_T_Weights.KINETICS400_V1

    model = swin3d_t(weights=weights)

    # Replace original 400-class head
    model.head = nn.Linear(
        model.head.in_features,
        num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state"])

    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    print("Validation accuracy:",
          checkpoint.get("val_acc", "unknown"))

    return model