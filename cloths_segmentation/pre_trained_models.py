from collections import namedtuple
from torch import nn
from torch.utils import model_zoo
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers

from segmentation_models_pytorch import Unet

model = namedtuple("model", ["url", "model"])

models = {
    "Unet_2020-10-30": model(
        url="https://github.com/ternaus/cloths_segmentation/releases/download/0.0.1/weights.zip",
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model
