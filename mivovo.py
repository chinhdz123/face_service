import cv2
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from mivolo.model.mivolo_model import *  # noqa: F403, F401
from timm.layers import set_layer_config
from timm.models._factory import parse_model_name
from timm.models._hub import load_model_config_from_hf
from timm.models._pretrained import PretrainedCfg, split_model_name_tag
from timm.models._registry import is_model, model_entrypoint
import os
from timm.models._helpers import load_state_dict, remap_checkpoint
import timm
from config import *
def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: str = "",
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    filter_keys=None,
    state_dict_map=None,
    **kwargs,
):
    """Create a model
    Lookup model's entrypoint function and pass relevant args to create a new model.
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        assert not pretrained_cfg, "pretrained_cfg should not be set when sourcing model from Hugging Face Hub."
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError("Unknown model (%s)" % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, filter_keys=filter_keys, state_dict_map=state_dict_map)
    model = model.to(DEVICE)
    model.eval()
    return model

def load_checkpoint(
    model, checkpoint_path, use_ema=True, strict=True, remap=False, filter_keys=None, state_dict_map=None
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            timm.models._model_builder.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    if remap:
        state_dict = remap_checkpoint(model, state_dict)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            for filter_key in filter_keys:
                if filter_key in sd_key:
                    if sd_key in state_dict:
                        del state_dict[sd_key]

    rep = []
    if state_dict_map is not None:
        # 'patch_embed.conv1.' : 'patch_embed.conv.'
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    target_name = state_k.replace(target_v, target_k)
                    state_dict[target_name] = state_dict[state_k]
                    rep.append(state_k)
        for r in rep:
            if r in state_dict:
                del state_dict[r]

    incompatible_keys = model.load_state_dict(state_dict, strict=strict if filter_keys is None else False)
    return incompatible_keys


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class Model():
    def __init__(self):
        self.model = create_model(model_name = 'mivolo_d1_224',num_classes=3,in_chans=3,pretrained=False,  checkpoint_path='ai_models\model_imdb_age_gender_4.22.pth.tar',filter_keys=["fds."])
        
    def predict(self, img):
        img = class_letterbox(img, new_shape=(224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        prepared_images: List[torch.tensor] = []
        prepared_images.append(img)
        prepared_input = torch.concat(prepared_images)
        prepared_input = prepared_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(prepared_input)

        age_output = output[:, 2]
        gender_output = output[:, :2].softmax(-1)
        gender_probs, gender_indx = gender_output.topk(1)
        age = age_output[0].item()
        age = age * (95 - 1) + 48
        age = round(age, 2)
        gender = "male" if gender_indx[0].item() == 0 else "female"
        return age, gender
    
model_cls = Model()
