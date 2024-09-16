import torch

from typing import Any, Dict, Optional
from einops import rearrange

from model_lib.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def register_reference_hooks(
    unet, do_classifier_free_guidance, mode
):
    unet = unet
    do_classifier_free_guidance = do_classifier_free_guidance
    mode = mode
    if do_classifier_free_guidance:
        uc_mask = (
            torch.Tensor([1] * 1 * 16 + [0] * 1 * 16)
            .to(unet.device)
            .bool()
        )
    else:
        uc_mask = (
            torch.Tensor([0] *1 * 2)
            .to(unet.device)
            .bool()
        )

    def hacked_basic_transformer_inner_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        video_length = None
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.norm1(
                hidden_states,
                timestep,
                class_labels,
                hidden_dtype=hidden_states.dtype,
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        if self.only_cross_attention:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention
                else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        else:
            if mode == "reference":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            if mode == "denoising":
                bank_fea = [
                    rearrange(
                        d.unsqueeze(1).repeat(1, video_length, 1, 1),
                        "b t l c -> (b t) l c",
                    )
                    for d in self.bank
                ]
                modify_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + bank_fea, dim=1
                )
                hidden_states_uc = (
                    self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=modify_norm_hidden_states,
                        attention_mask=attention_mask,
                    )
                    + hidden_states
                )
                if do_classifier_free_guidance:
                    hidden_states_c = hidden_states_uc.clone()
                    _uc_mask = uc_mask.clone()
                    if hidden_states.shape[0] != _uc_mask.shape[0]:
                        _uc_mask = (
                            torch.Tensor(
                                [1] * (hidden_states.shape[0] // 2)
                                + [0] * (hidden_states.shape[0] // 2)
                            )
                            .to(unet.device)
                            .bool()
                        )
                    hidden_states_c[_uc_mask] = (
                        self.attn1(
                            norm_hidden_states[_uc_mask],
                            encoder_hidden_states=norm_hidden_states[_uc_mask],
                            attention_mask=attention_mask,
                        )
                        + hidden_states[_uc_mask]
                    )
                    hidden_states = hidden_states_c.clone()
                else:
                    hidden_states = hidden_states_uc

                # self.bank.clear()
                if self.attn2 is not None:
                    # Cross-Attention
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep)
                        if self.use_ada_layer_norm
                        else self.norm2(hidden_states)
                    )
                    hidden_states = (
                        self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )

                # Feed-forward
                hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                # Temporal-Attention
                if self.unet_use_temporal_attention:
                    d = hidden_states.shape[1]
                    hidden_states = rearrange(
                        hidden_states, "(b f) d c -> (b d) f c", f=video_length
                    )
                    norm_hidden_states = (
                        self.norm_temp(hidden_states, timestep)
                        if self.use_ada_layer_norm
                        else self.norm_temp(hidden_states)
                    )
                    hidden_states = (
                        self.attn_temp(norm_hidden_states) + hidden_states
                    )
                    hidden_states = rearrange(
                        hidden_states, "(b d) f c -> (b f) d c", d=d
                    )

                return hidden_states

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

    attn_modules = [
        module
        for module in torch_dfs(unet)
        #for module in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks))
        if isinstance(module, BasicTransformerBlock)
        or isinstance(module, TemporalBasicTransformerBlock)
    ]
    attn_modules = sorted(
        attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
    )

    for i, module in enumerate(attn_modules):
        module._original_inner_forward = module.forward
        if isinstance(module, BasicTransformerBlock):
            module.forward = hacked_basic_transformer_inner_forward.__get__(
                module, BasicTransformerBlock
            )
        if isinstance(module, TemporalBasicTransformerBlock):
            module.forward = hacked_basic_transformer_inner_forward.__get__(
                module, TemporalBasicTransformerBlock
            )

        module.bank = []
        module.attn_weight = float(i) / float(len(attn_modules))

def get_reference_features(unet, dtype=torch.float16):
    attn_modules = [
        module
        for module in torch_dfs(unet) #16
        #for module in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks)) #10
        if isinstance(module, BasicTransformerBlock)
    ]

    attn_modules = sorted(
        attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
    )

    reference_features = [f.clone().to(dtype) for attn_module in attn_modules for f in attn_module.bank]

    return reference_features

def update_reference_features(unet, reference_features, dtype=torch.float16):
    attn_modules = [
        module
        for module in torch_dfs(unet)
        #for module in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks))
        if isinstance(module, TemporalBasicTransformerBlock)
    ]
    attn_modules = sorted(
        attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
    )
    for attn_module, reference_feature in zip(attn_modules, reference_features):
        attn_module.bank = [reference_feature.clone().to(dtype)]

def clear_reference(unet):
    attn_modules = [
        module
        for module in torch_dfs(unet)
        #for module in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks))
        if isinstance(module, BasicTransformerBlock)
        or isinstance(module, TemporalBasicTransformerBlock)
    ]
    attn_modules = sorted(
        attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
    )
    for attn_module in attn_modules:
        attn_module.bank.clear()