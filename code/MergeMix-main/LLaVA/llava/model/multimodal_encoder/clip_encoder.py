import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder
from llava.model.multimodal_encoder.modeling_tome_clip import CLIPToMeVisionModel, CLIPToMeAttention, CLIPToMeEncoderLayer
from llava.model.augment.merge import parse_r, check_parse_r

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, return_attn=False):
        if type(images) is list:       # FOR VEDIO ?
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        if return_attn:
            return attn_weights
        else:
            return image_features
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)

# ------ CLIP Modeling with Token Merge ------ #
class CLIPToMeVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # TOKEN MERGE RATIO
        self.r = args.tome_ratio
        self.merge_num = args.tome_merge_num
        print(f"merge token: {self.merge_num}")
        self.num_layers = 24

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    # def load_model(self, device_map=None):
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPToMeVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

        self.vision_tower = self.apply_tome(self.vision_tower, self.r, self.merge_num)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # defaluts -2
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, return_attn=False):
        # TODO Vedio + Token Merge
        if type(images) is list:       # FOR VEDIO ?
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-1]
            if return_attn:
                return attn_weights
            if self.vision_tower._tome_info["source"] is not None and return_attn:
                source = self.vision_tower._tome_info["source"]
                return (attn_weights, source)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    def make_tome_class(self, transformer_class):
        class ToMeTransformer(transformer_class):
            """
            Modifications:
            - Initialize r, token size, and token sources.
            """
            def forward(self, *args, **kwdargs) -> torch.Tensor:
                self._tome_info["r"] = parse_r(  # CLIP blocks => layers == 24
                    24, self.r, self._tome_info["total_merge"])
                self._tome_info["size"] = None
                self._tome_info["source"] = None

                return super().forward(*args, **kwdargs)

        return ToMeTransformer

    def apply_tome(self, model, merge_ratio=None, merge_num=None):
        curr_res = int(self.config.image_size / self.config.patch_size)
        inflect, num_layers = -0.5, 0

        # check ToMe settings
        for module in model.modules():
            if isinstance(module, (CLIPAttention, CLIPToMeAttention)):
                num_layers += 1
        if merge_ratio > 0 and merge_num == 0:
            if isinstance(merge_ratio, tuple):
                merge_ratio, inflect = merge_ratio
            merge_num = sum(parse_r(num_layers, r=(merge_ratio, inflect)))
        elif merge_ratio == 0 and merge_num > 0:
            merge_ratio = check_parse_r(num_layers, merge_num, curr_res ** 2, inflect)
        # apply ToMe
        self.apply_patch(model)
        model.r = (merge_ratio, -0.5)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = merge_num

        return model

    def apply_patch(self, model, trace_source: bool = True, prop_attn: bool = True):
        CLIPToMeTransformer = self.make_tome_class(model.__class__)

        model.__class__ = CLIPToMeTransformer
        model.r = 0  
        model._tome_info = {
            "r": model.r,
            "size": None,
            "source": None,
            "total_merge": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": True,
        }
        for module in model.modules():
            if isinstance(module, CLIPToMeEncoderLayer):
                module.__class__ = CLIPToMeEncoderLayer
                module._tome_info = model._tome_info
            elif isinstance(module, CLIPAttention):
                module.__class__ = CLIPToMeAttention
