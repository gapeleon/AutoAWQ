import functools
from collections import defaultdict
from tqdm import auto as tqdm_lib

import torch
from awq.quantize.quantizer import AwqQuantizer
from awq.quantize.scale import apply_scale, apply_clip
from awq.models.llama4 import Llama4TextMoe
from awq.utils.utils import clear_memory, get_best_device
from awq.utils.calib_data import get_calib_dataset
from awq.modules.act import ScaledActivation
from transformers.activations import ACT2FN

from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe as OldLlama4TextMoe
from transformers.feature_extraction_utils import BatchFeature

from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_op_by_name, 
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)

class Llama4AwqQuantizer(AwqQuantizer):

    def _preprocess(self):
        pass

    def _preprocess_layer_iter(self, layer_index):
        self.modules[layer_index].load_state_dict(
            torch.load(f"/workspace/hf_cache/marverick/lang_layer{layer_index:03}.pth"),
            assign=True,
        )
        self.modules[layer_index] = self.modules[layer_index].to(torch.float16)
        if isinstance(self.modules[layer_index].feed_forward, OldLlama4TextMoe):
            self.modules[layer_index].feed_forward = Llama4TextMoe.replace(self.modules[layer_index].feed_forward.to('cpu'))
        
        common_device = next(self.modules[layer_index].parameters()).device
        if common_device is None or str(common_device) == "cpu":
            if torch.cuda.is_available():
                best_device = "cuda:" + str(layer_index % torch.cuda.device_count())
            else:
                best_device = get_best_device()

            self.modules[layer_index] = self.modules[layer_index].to(best_device)
            common_device = next(self.modules[layer_index].parameters()).device

        if self.module_kwargs.get("position_ids") is not None:
            self.module_kwargs["position_ids"] = self.module_kwargs[
                "position_ids"
            ].to(common_device)

        if self.module_kwargs.get("attention_mask") is not None:
            self.module_kwargs["attention_mask"] = self.module_kwargs[
                "attention_mask"
            ].to(common_device)

        self.inps = self.inps.to(common_device)

    def _postprocess_layer_iter(self, layer_index):
        self.modules[layer_index] = self.modules[layer_index].to("cpu")
        torch.save(self.modules[layer_index].state_dict(), f"/workspace/hf_cache/marverick/q{layer_index:03}.pt")
        self.modules[layer_index] = None
        clear_memory()
    
    def quantize(self):
        self._preprocess()
        for i in tqdm_lib.tqdm(range(len(self.modules)), desc="AWQ"):
            self._preprocess_layer_iter(i)
            
            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)
            self._postprocess_layer_iter(i)

            clear_memory()
    