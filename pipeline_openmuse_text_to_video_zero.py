from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...image_processor import VaeImageProcessor
from ...models import UVit2DModel, VQModel
from ...schedulers import OpenMuseScheduler
from ..pipeline_utils import BaseOutput, DiffusionPipeline


@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        images (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]


def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    # warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    # warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="border")
    warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="zeros")
    return warped


def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype):
    """
    Create translation motion field

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        device: device
        dtype: dtype

    Returns:

    """
    seq_length = len(frame_ids)
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    for fr_idx in range(seq_length):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx])
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx])
    return reference_flow


def create_motion_field_and_warp_latents(motion_field_strength_x, motion_field_strength_y, frame_ids, latents):
    """
    Creates translation motion and warps the latents accordingly

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        latents: latent codes of frames

    Returns:
        warped_latents: warped latents
    """
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        frame_ids=frame_ids,
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


class OpenMuseTextToVideoZeroPipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: UVit2DModel
    scheduler: OpenMuseScheduler

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: UVit2DModel,
        scheduler: OpenMuseScheduler,
    ):
        super().__init__()

        embedding_layer = vqvae.quantize.embedding
        n_new_embeddings = scheduler.config.mask_token_id - embedding_layer.num_embeddings + 1
        new_embeddings = torch.randn(
            n_new_embeddings, embedding_layer.embedding_dim, device=embedding_layer.weight.device
        )
        extended_weight = torch.cat([embedding_layer.weight, new_embeddings], 0)
        embedding_layer.num_embeddings += n_new_embeddings
        embedding_layer.weight = torch.nn.Parameter(extended_weight)

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    def backward_loop(
        self,
        latents,
        guidance_scale,
        callback,
        callback_steps,
        prompt_embeds,
        micro_conds,
        encoder_hidden_states,
        generator,
        masking_schedule="cos",
        cross_attention_kwargs=None,
    ):
        """
        Perform backward process given list of time steps.

        Args:
            latents:
                Latents at time timesteps[0].
            timesteps:
                Time steps along which to perform backward process.
            prompt_embeds:
                Pre-generated text embeddings.
            guidance_scale:
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            extra_step_kwargs:
                Extra_step_kwargs.
            cross_attention_kwargs:
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            num_warmup_steps:
                number of warmup steps.

        Returns:
            latents:
                Latents of backward process output at time timesteps[-1].
        """
        starting_mask_ratio = float((latents == self.scheduler.config.mask_token_id).sum()) / float(latents.numel())

        do_classifier_free_guidance = guidance_scale > 1.0

        num_warmup_steps = 0
        timesteps = self.scheduler.timesteps
        num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order

        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    micro_conds=micro_conds,
                    cond_embeds=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=generator,
                    starting_mask_ratio=starting_mask_ratio,
                    masking_schedule=masking_schedule,
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        return latents.clone().detach()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int] = 8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        guidance_scale=10.0,
        num_inference_steps=12,
        negative_prompt_embeds=None,
        generator=None,
        generator2=None,
        latents=None,
        follow_on_frames_mask_ratio=0.13,
        end_latents=None,
        end_addition_ratio=None,
    ):
        assert video_length > 0

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        batch_size = len(prompt)

        device = self._execution_device

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids.to(self._execution_device)

        outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.text_embeds
        encoder_hidden_states = outputs.hidden_states[-2]

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                input_ids = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)

                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                negative_prompt_embeds = outputs.text_embeds
                negative_encoder_hidden_states = outputs.hidden_states[-2]

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        micro_conds = torch.tensor(
            [height, width, 0, 0, 6], device=self._execution_device, dtype=encoder_hidden_states.dtype
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        if latents is None:
            latents = torch.full(
                (batch_size, (height // self.vae_scale_factor) * (width // self.vae_scale_factor)),
                self.scheduler.config.mask_token_id,
                dtype=torch.long,
                device=self._execution_device,
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        frame0 = self.backward_loop(
            prompt_embeds=prompt_embeds,
            micro_conds=micro_conds,
            encoder_hidden_states=encoder_hidden_states,
            generator=generator,
            latents=latents,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            masking_schedule="cos",
        )

        all_frames = [frame0]

        for i in range(video_length - 1):
            frame_n = all_frames[-1].clone()

            # de-quantize
            frame_n = self.vqvae.quantize.get_codebook_entry(
                frame_n,
                (
                    frame_n.shape[0],
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                    self.vqvae.latent_channels,
                ),
            )

            # warp
            frame_n = create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x,
                motion_field_strength_y=motion_field_strength_y,
                latents=frame_n,
                frame_ids=[1],
            )

            # replace padding with mask
            padded_indices = (frame_n == 0).all(dim=1, keepdim=True)
            masked_embedding = self.vqvae.quantize.embedding.weight[-1]
            n_channels = frame_n.shape[1]
            frame_n = frame_n.permute(0, 2, 3, 1)
            frame_n[padded_indices.repeat(1, n_channels, 1, 1).permute(0, 2, 3, 1)] = masked_embedding.repeat(
                padded_indices.sum()
            )
            frame_n = frame_n.permute(0, 3, 1, 2)

            # quantize
            frame_n = self.vqvae.quantize(frame_n)[2][2].reshape(frame_n.shape[0], -1)

            if isinstance(follow_on_frames_mask_ratio, list):
                if i < len(follow_on_frames_mask_ratio):
                    mask_ratio = follow_on_frames_mask_ratio[i]
                else:
                    mask_ratio = 0.13
            else:
                mask_ratio = follow_on_frames_mask_ratio

            if isinstance(end_addition_ratio, list) and i < len(end_addition_ratio):
                end_addition_ratio_ = end_addition_ratio[i]
            else:
                end_addition_ratio_ = None

            frame_n = self.scheduler.add_noise(frame_n, mask_ratio=mask_ratio, generator=generator, end_addition_ratio=end_addition_ratio_, end_latents=end_latents)

            frame_n = self.backward_loop(
                prompt_embeds=prompt_embeds,
                micro_conds=micro_conds,
                encoder_hidden_states=encoder_hidden_states,
                generator=generator2 if generator2 is not None else generator,
                latents=frame_n,
                guidance_scale=guidance_scale,
                callback=callback,
                callback_steps=callback_steps,
                masking_schedule="lin",
            )

            all_frames.append(frame_n)

        all_frames = torch.concat(all_frames)

        images = self.vqvae.decode(
            all_frames,
            force_not_quantize=True,
            shape=(
                all_frames.shape[0],
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
                self.vqvae.config.latent_channels,
            ),
        ).sample
        images = self.image_processor.postprocess(images, output_type)

        if not return_dict:
            return (images,)

        return TextToVideoPipelineOutput(images=images)