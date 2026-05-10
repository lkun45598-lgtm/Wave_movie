"""
ResShift Trainer (masked version).

@author Leizheng
@contributors Leizheng
@date 2026-02-06
@version 4.1.0

@changelog
  - 2026-02-11 Leizheng: v4.1.0 修复 inference() 死代码
    - 清除无效的 tt 变量（两处）、indices 变量、无效 self._unwrap() 调用
    - model 改用 self._unwrap() 返回值，避免 DDP 包装传入 p_sample_loop
    - 保留全步采样（p_sample 后验系数不支持跳步，需 SpacedDiffusion 才行）
  - 2026-02-07 Leizheng: v4.0.0 inference 自动对齐尺寸 + crop 回原尺寸
    - interpolate 到 model_divisor 对齐后的尺寸
    - 采样完成后 crop 回原始 y 尺寸
  - 2026-02-07 Leizheng: v3.0.0 AMP 混合精度 + Gradient Checkpointing
    - train() 使用 autocast + GradScaler
    - gradient checkpointing 包装 training_losses forward
  - 2026-02-06 Leizheng: v2.0.0 masked 版本
    - 训练 loss 来自 base_diffusion.training_losses()，无法直接注入 mask
    - 评估阶段通过继承 BaseTrainer.evaluate() 使用 masked metrics
  - 原始版本: v1.0.0
"""

import torch
import torch.distributed as dist
import torch.utils.checkpoint
import functools
import numpy as np
from models import _ddpm_dict
from utils.loss import LossRecord
import torch.nn.functional as F
from .base import BaseTrainer, apply_sparse_known_constraint

from utils.metrics import get_obj_from_str

class ResshiftTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def build_model(self, **kwargs):

        self.resshift_cfg = self.args['resshift']

        params = self.resshift_cfg["model"]['params']
        model =get_obj_from_str(self.resshift_cfg['model']['target'])(**params)

        params = self.resshift_cfg["diffusion"]['params']
        self.base_diffusion = get_obj_from_str(self.resshift_cfg['diffusion']['target'])(**params)
        self.resshift_lq_channels = self.model_args.get('resshift_lq_channels', None)
        self.resshift_diffusion_y_channels = self.model_args.get(
            'resshift_diffusion_y_channels',
            None,
        )
        self.resshift_aux_loss_weight = float(
            self.optim_args.get('resshift_aux_loss_weight', 0.0)
        )
        return model

    @staticmethod
    def _select_channels_nhwc(tensor, channels):
        if channels is None:
            return tensor
        if isinstance(channels, int):
            channels = [channels]
        return tensor[..., list(channels)]

    def _prepare_condition_and_degraded(self, x, target_hw=None):
        """Return NCHW tensors for the ResShift condition and degraded baseline."""
        lq = self._select_channels_nhwc(x, self.resshift_lq_channels)
        degraded = self._select_channels_nhwc(x, self.resshift_diffusion_y_channels)
        lq = lq.permute(0, 3, 1, 2)
        degraded = degraded.permute(0, 3, 1, 2)
        if target_hw is not None:
            lq = F.interpolate(lq, size=target_hw, mode='bicubic', align_corners=False)
            degraded = F.interpolate(
                degraded,
                size=target_hw,
                mode='bicubic',
                align_corners=False,
            )
        return lq, degraded

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss", "diffusion_loss", "aux_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            # Patch 训练时 batch = (x, y, mask_hr_patch)，否则 (x, y)
            if len(batch) == 3:
                x, y, _mask_hr = batch
            else:
                x, y = batch
                _mask_hr = None
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if _mask_hr is not None:
                _mask_hr = _mask_hr.to(self.device, non_blocking=True)
            x_nhwc = x
            y_nhwc = y
            y = y.permute(0, 3, 1, 2)
            lq, degraded = self._prepare_condition_and_degraded(
                x_nhwc,
                target_hw=y.shape[2:],
            )

            B, C, H, W = y.shape

            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(y.shape[0],),
                    device=y.device,
                    )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.gradient_checkpointing:
                    def _forward(gt, low_quality, diffusion_y, timesteps):
                        model_kwargs = {'lq': low_quality}
                        losses, _z_t, pred_zstart = self.base_diffusion.training_losses(
                            self.model, gt, diffusion_y, timesteps,
                            first_stage_model=None, model_kwargs=model_kwargs, noise=None,
                        )
                        return losses['mse'], pred_zstart
                    diffusion_loss, pred_zstart = torch.utils.checkpoint.checkpoint(
                        _forward, y, lq, degraded, tt, use_reentrant=False
                    )
                else:
                    model_kwargs = {'lq': lq}
                    compute_losses = functools.partial(
                        self.base_diffusion.training_losses,
                        self.model, y, degraded, tt,
                        first_stage_model=None,
                        model_kwargs=model_kwargs,
                        noise=None,
                    )
                    losses, _z_t, pred_zstart = compute_losses()
                    diffusion_loss = losses['mse']

                aux_loss = y.new_tensor(0.0)
                loss = diffusion_loss
                if self.resshift_aux_loss_weight > 0:
                    pred_nhwc = pred_zstart.permute(0, 2, 3, 1)
                    base_mask = _mask_hr if _mask_hr is not None else self.mask_hr
                    loss_mask = self.build_loss_mask(base_mask, x_nhwc, y_nhwc)
                    aux_loss = self.loss_fn(
                        pred_nhwc,
                        y_nhwc,
                        mask=loss_mask,
                        input_tensor=x_nhwc,
                        observed_mask_channel=self.loss_mask_observed_channel,
                    )
                    loss = loss + self.resshift_aux_loss_weight * aux_loss

            loss_record.update(
                {
                    "train_loss": loss.item(),
                    "diffusion_loss": diffusion_loss.item(),
                    "aux_loss": aux_loss.item(),
                },
                n=B,
            )
            # 实时更新内层 batch 进度条的 loss 显示
            if getattr(self, '_batch_bar', None) is not None:
                self._batch_bar.set_postfix(
                    {
                        "loss": f"{loss_record.to_dict()['train_loss']:.4f}",
                        "diff": f"{loss_record.to_dict()['diffusion_loss']:.4f}",
                    },
                    refresh=False,
                )
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record

    def inference(self, x, y, **kwargs):
        x_input = x
        y = y.permute(0, 3, 1, 2)

        orig_h, orig_w = y.shape[2], y.shape[3]

        # interpolate 到对齐后的尺寸（能被 model_divisor 整除）
        d = self.model_divisor
        if d > 1:
            aligned_h = ((orig_h + d - 1) // d) * d
            aligned_w = ((orig_w + d - 1) // d) * d
        else:
            aligned_h, aligned_w = orig_h, orig_w
        lq, degraded = self._prepare_condition_and_degraded(
            x_input,
            target_hw=(aligned_h, aligned_w),
        )

        model_kwargs = {'lq': lq}
        model = self._unwrap()

        # NOTE: ResShift p_sample() 的后验系数是按连续步 t→t-1 推导的，
        # 不支持跳步采样。若要加速推理，应在 diffusion 创建时设置
        # timestep_respacing 参数（通过 SpacedDiffusion 重算后验系数）。
        # 当前配置 steps=15 已经很快，保持全步采样。
        y_pred = self.base_diffusion.p_sample_loop(
                        y=degraded,
                        model=model,
                        first_stage_model=None,
                        noise=None,
                        clip_denoised=None,
                        model_kwargs=model_kwargs,
                        device=degraded.device,
                        progress=True,
                        )

        # crop 回原始尺寸
        y_pred = y_pred[:, :, :orig_h, :orig_w]
        y_pred = y_pred.permute(0, 2, 3, 1)
        if self.sparse_known_constraint and kwargs.get('apply_known_constraint', True):
            y_pred = apply_sparse_known_constraint(
                y_pred,
                x_input,
                observed_value_channels=self.sparse_known_value_channels,
                mask_channel=self.sparse_known_mask_channel,
            )
        return y_pred
