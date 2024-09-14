import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import inspect
import gc

from tqdm import tqdm
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from typing import List

from .reference_feature import register_reference_hooks, get_reference_features, update_reference_features, clear_reference
from .appearance_feature import register_appearance_hooks, get_appearance_features, clear_appearance
from .util import decode_latents, save_video, get_memory_cost, transform

class linfpgdattack():
    def __init__(self, clip_encoder, vae, denoising_unet, reference_unet, appearance_encoder, appearance_control_model, pose_guider, scheduler, generator, args, clip_min = -1., clip_max = 1.):
        self.clip_encoder = clip_encoder
        self.vae = vae
        self.denoising_unet = denoising_unet
        self.reference_unet = reference_unet
        self.appearance_encoder = appearance_encoder
        self.appearance_control_model = appearance_control_model
        self.pose_guider = pose_guider
        self.scheduler = scheduler
        self.generator = generator
        self.lpips = lpips.LPIPS(net='vgg').to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)

        self.guidance_scale = args.cfg
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.timesteps = self.scheduler.timesteps
        self.width, self.height, self.length= args.W, args.H, args.L

        self.gpu_id = args.gpu_id

        self.eps = args.eps / 255.0
        self.steps = args.pgd_steps
        self.step_size = args.step_size / 255.0
        self.clip_min = clip_min
        self.clip_max = clip_max

        register_reference_hooks(self.reference_unet, self.do_classifier_free_guidance, mode="reference")
        register_appearance_hooks(self.appearance_encoder, self.do_classifier_free_guidance, mode="reference")
        register_reference_hooks(self.denoising_unet, self.do_classifier_free_guidance, mode="denoising")

    def perturb(self, x, pose_cond_tensor):
        clip_image_embeds_ori = self.clip_encoder(
            F.interpolate(x, size=(224, 224))
        ).image_embeds
        encoder_hidden_states_ori = clip_image_embeds_ori.unsqueeze(1)
        uncond_encoder_hidden_states_ori = torch.zeros_like(encoder_hidden_states_ori)

        if self.do_classifier_free_guidance:
            encoder_hidden_states_ori = torch.cat(
                [uncond_encoder_hidden_states_ori, encoder_hidden_states_ori], dim=0
            )

        ref_image_latents_ori = self.vae.encode(x).latent_dist.mean * 0.18215  # (b, 4, h, w)

        self.reference_unet(
            ref_image_latents_ori.repeat(
                (2 if self.do_classifier_free_guidance else 1), 1, 1, 1
            ),
            torch.zeros_like(self.timesteps[0]),
            # t,
            encoder_hidden_states=encoder_hidden_states_ori,
            return_dict=False,
        )
        reference_features_ori = get_reference_features(self.reference_unet, self.reference_unet.dtype)

        pose_fea = self.pose_guider(pose_cond_tensor)
        extra_step_kwargs = prepare_extra_step_kwargs(self.scheduler, self.generator)
        text_embeddings = get_text_embeddings(self.appearance_encoder.device, self.do_classifier_free_guidance)

        del encoder_hidden_states_ori, uncond_encoder_hidden_states_ori
        del self.pose_guider
        clear_reference(self.reference_unet)
        gc.collect()
        torch.cuda.empty_cache()
        #get_memory_cost(self.gpu_id, "Get original features")

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.data.uniform_(-1, 1) * self.eps
        delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data
        delta.requires_grad_(True)

        decay = 0.5
        momentum = torch.zeros_like(x)

        pbar = tqdm(range(self.steps))
        for ii in pbar:
            x_adv_trans = transform(x + delta)
            clip_image_embeds = self.clip_encoder(F.interpolate(x_adv_trans, size=(224, 224))).image_embeds
            encoder_hidden_states = clip_image_embeds.unsqueeze(1)
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            if self.do_classifier_free_guidance:
                encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)
            loss_clip = F.mse_loss(clip_image_embeds_ori, clip_image_embeds)

            ref_image_latents = self.vae.encode(x_adv_trans).latent_dist.mean * 0.18215
            loss_vae = F.mse_loss(ref_image_latents_ori, ref_image_latents)

            self.reference_unet(ref_image_latents.repeat((2 if self.do_classifier_free_guidance else 1), 1, 1, 1),torch.zeros_like(self.timesteps[0]),encoder_hidden_states=encoder_hidden_states,return_dict=False,)
            reference_features = get_reference_features(self.reference_unet, self.reference_unet.dtype)
            loss_ref = [F.mse_loss(reference_feature_ori, reference_feature) for reference_feature_ori, reference_feature in zip(reference_features_ori, reference_features)]
            loss_ref = sum(loss_ref) / len(loss_ref)
            #get_memory_cost(self.gpu_id, "Compute reference feature loss")

            latents = randn_tensor((1, self.denoising_unet.config.in_channels, self.length, self.height // self.vae_scale_factor, self.width // self.vae_scale_factor), generator=self.generator, device=self.denoising_unet.device, dtype=self.denoising_unet.dtype)
            latents = latents * self.scheduler.init_noise_sigma

            update_reference_features(self.denoising_unet, reference_features, self.denoising_unet.dtype)
            index = torch.randint(len(self.timesteps) - 10, len(self.timesteps), (1,))
            t = self.timesteps[index[0]]

            self.appearance_encoder(
                ref_image_latents_ori.repeat((2 if self.do_classifier_free_guidance else 1), 1, 1, 1),
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            appearance_features_ori = get_appearance_features(self.appearance_encoder, self.appearance_encoder.dtype)
            clear_appearance(self.appearance_encoder)

            self.appearance_encoder(
                ref_image_latents.repeat((2 if self.do_classifier_free_guidance else 1), 1, 1, 1),
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            appearance_features = get_appearance_features(self.appearance_encoder, self.appearance_encoder.dtype)
            clear_appearance(self.appearance_encoder)
            loss_app = [F.mse_loss(appearance_feature_ori, appearance_feature) for appearance_feature_ori, appearance_feature in zip(appearance_features_ori, appearance_features)]
            loss_app = sum(loss_app) / len(loss_app)
            #get_memory_cost(self.gpu_id, "Compute appearance feature loss")

            appearance_control_features_ori = []
            self.appearance_control_model.appearance_control_model(x=ref_image_latents_ori, hint=None, timesteps=t.unsqueeze(0), context=text_embeddings[-1].unsqueeze(0), attention_bank=appearance_control_features_ori, attention_mode='write', uc=False)

            appearance_control_features = []
            self.appearance_control_model.appearance_control_model(x=ref_image_latents, hint=None, timesteps=t.unsqueeze(0), context=text_embeddings[-1].unsqueeze(0), attention_bank=appearance_control_features, attention_mode='write', uc=False)
            loss_appctrl = [F.mse_loss(appearance_control_feature_ori[0], appearance_control_feature[0]) for appearance_control_feature_ori, appearance_control_feature in zip(appearance_control_features_ori, appearance_control_features)]
            loss_appctrl = sum(loss_appctrl) / len(loss_appctrl)
            #get_memory_cost(self.gpu_id, "Compute appearance control feature loss")

            noise_pred = torch.zeros(
                (
                    latents.shape[0] * (2 if self.do_classifier_free_guidance else 1),
                    *latents.shape[1:],
                ),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )

            context = [[_ for _ in range(latents.shape[2])]]

            latent_model_input = (
                torch.cat([latents[:, :, c] for c in context])
                .to(self.denoising_unet.device)
                .repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            b, c, f, h, w = latent_model_input.shape
            latent_pose_input = torch.cat(
                [pose_fea[:, :, c] for c in context]
            ).repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)

            pred = self.denoising_unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states[:b],
                pose_cond_fea=latent_pose_input,
                return_dict=False,
            )[0]

            for j, c in enumerate(context):
                noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).pred_original_sample

            #if ii == self.steps - 1:
            #    images = decode_latents(self.vae, latents)
            #    save_video(images)
            #    del images

            del noise_pred, counter, context, latent_model_input, latent_pose_input, pred, noise_pred_uncond, noise_pred_text
            gc.collect()
            torch.cuda.empty_cache()
            #get_memory_cost(self.gpu_id, "Get denoising unet output")

            latents = rearrange(latents, "b c f h w -> (b f) c h w")

            loss_neigh = []
            loss_sim = []
            for i in range(len(latents)):
                loss_sim.append(F.mse_loss(latents[i], ref_image_latents_ori))
                for j in range(i+1, len(latents)):
                    loss_neigh.append(F.mse_loss(latents[i], latents[j]))

            loss_sim = sum(loss_sim) / len(loss_sim)
            loss_neigh = sum(loss_neigh) / len(loss_neigh)

            loss_lpips = self.lpips(x, x + delta)
            loss_lpips = max(loss_lpips - self.eps, 0)
            #get_memory_cost(self.gpu_id, "Compute lpips loss")

            loss_embd = loss_clip + loss_vae
            loss_ensemble = loss_ref + loss_app + loss_appctrl
            loss_consist = loss_sim + loss_neigh

            alpha1, alpha2, alpha3, alpha4 = 10, 100, 1, 10
            loss = alpha1 * loss_embd + alpha2 * loss_ensemble + alpha3 * loss_consist - alpha4 * loss_lpips
            pbar.set_description(f"[Running attack]: Clip Loss {loss_clip.item():.5f} | Vae Loss {loss_vae.item():.5f} | Ref Loss {loss_ref.item():.5f} | App Loss {loss_app.item():.5f} | AppCtrl Loss {loss_appctrl.item():.5f} | Sim Loss {loss_sim.item():.5f} | Neigh Loss {loss_neigh.item():.5f} | Lpips Loss {loss_lpips.item():.5f}")

            grad = torch.autograd.grad(loss, [delta])[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * decay
            momentum = grad

            delta.data = delta.data + self.step_size * grad.data.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta.data, self.clip_min, self.clip_max) - x.data

            del loss, loss_clip, loss_vae, loss_ref, loss_sim, loss_neigh, loss_lpips, loss_app, loss_appctrl, loss_ensemble, loss_consist, clip_image_embeds, reference_features, ref_image_latents, encoder_hidden_states, uncond_encoder_hidden_states, latents, x_adv_trans, grad, appearance_features, appearance_features_ori, appearance_control_features, appearance_control_features_ori
            gc.collect()
            torch.cuda.empty_cache()
            clear_reference(self.reference_unet)
            clear_reference(self.denoising_unet)
            #get_memory_cost(self.gpu_id, "After backward")

        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        return x_adv

def prepare_extra_step_kwargs(scheduler, generator, eta = 0.0):
    accepts_eta = "eta" in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def get_text_embeddings(device, do_classifier_free_guidance, prompt=[''], negative_prompt=[''], num_videos_per_prompt=1):

    def encode_prompt(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, tokenizer, text_encoder):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("../../../data/models/SD-1-5/", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("../../../data/models/SD-1-5/", subfolder="text_encoder").to(dtype=torch.float16, device=device)
    text_embeddings = encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, tokenizer, text_encoder
        )

    del tokenizer
    del text_encoder
    return text_embeddings