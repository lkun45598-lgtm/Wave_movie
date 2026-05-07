import os
import yaml
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

from models import _model_dict
from datasets import _dataset_dict
from utils.loss import LossRecord
from utils.metrics import Evaluator
from utils.sr import make_lr_blur
from scipy.io import savemat,loadmat

import numpy as np
from scipy import stats



class BaseForecaster(object):
    def __init__(self, path,device='cpu'):
        self.saving_path = path

        args_path = os.path.join(self.saving_path, 'config.yaml')
        self.args = yaml.load(open(args_path, 'r'), Loader=yaml.FullLoader)
        self.model_args = self.args['model']
        self.train_args = self.args['train']
        self.data_args = self.args['data']

        self.shape = self.data_args['shape']

        torch.manual_seed(self.train_args.get('seed', 42))
        self.device = device

        self.model_name = self.model_args['name']
        self.model = self.build_model()
        self.load_model()
        self.build_evaluator()
        self.model.to(self.device)

    def build_model(self, **kwargs):
        if self.model_name not in _model_dict.keys():
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        print("Building model: {}".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model

    def load_model(self, **kwargs):
        model_path = os.path.join(self.saving_path, 'best_model.pth')
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def build_evaluator(self):
        self.evaluator = Evaluator(self.shape)

    def build_data(self, **kwargs):
        if self.data_name not in _dataset_dict:
            raise NotImplementedError("Dataset {} not implemented".format(self.data_name))
        dataset = _dataset_dict[self.data_name](self.data_args, **kwargs)
        self.normalizer = dataset.normalizer
        self.train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=self.data_args.get('train_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            drop_last=True,
            pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(
            dataset.valid_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            pin_memory=True)

    def forecast(self, loader, normalizer, **kwargs):
        loss_record = self.evaluator.init_record()
        all_y = []
        all_y_pred = []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                y_pred = normalizer.decode(y_pred)
                y = normalizer.decode(y)
                all_y.append(y)
                all_y_pred.append(y_pred)
        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        self.evaluator(y_pred, y, record=loss_record)
        print(loss_record)
        return loss_record

    def inference(self, x, y, **kwargs):
        return self.model(x).reshape(y.shape)

    def vis_ns(self, raw_x, raw_y, normalizer, save_path=None,
               max_error=0.5, dpi=100,cmap = 'viridis', **kwargs):
        self.model.eval()

        with torch.no_grad():
            raw_x = raw_x.to(self.device)
            pred_y = self.inference(raw_x, raw_y, **kwargs)
        pred_y = normalizer.decode(pred_y)
        raw_y = normalizer.decode(raw_y)
        raw_x = make_lr_blur(raw_y.permute(0, 3, 1, 2), scale=self.data_args.get('sample_factor', 2)).permute(0, 2, 3, 1)

        pred_y = pred_y.cpu().reshape(self.shape)
        raw_y = raw_y.cpu().reshape(self.shape)
        raw_x = raw_x.cpu().reshape(self.shape[0] // self.data_args.get('sample_factor', 2),
                                    self.shape[1] // self.data_args.get('sample_factor', 2))

        error_y = torch.abs(pred_y - raw_y)

        vmin = raw_y.min()
        vmax = raw_y.max()

        fig, axs = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True, dpi=dpi)

        axs[0].imshow(raw_x, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title('Input LR')
        axs[0].axis('off')

        axs[1].imshow(raw_y, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title('Ground Truth HR')
        axs[1].axis('off')

        heatmap = axs[2].imshow(pred_y, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[2].set_title('{} Prediction HR'.format(self.model_name))
        axs[2].axis('off')

        errormap = axs[3].imshow(error_y, cmap='hot', vmin=0, vmax=max_error)
        axs[3].set_title('Absolute Error')
        axs[3].axis('off')

        for ax, mappable in [(axs[2], heatmap), (axs[3], errormap)]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.04)  # 统一 size/pad
            fig.colorbar(mappable, cax=cax)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()


    def _calculate_radially_averaged_spectrum(self, field_2d):
        """
        计算单个二维场的径向平均功率谱。
        (修正版)

        Args:
            field_2d (torch.Tensor): 一个二维张量 (H, W)。

        Returns:
            tuple: (波数 np.ndarray, 谱 np.ndarray)
        """
        if field_2d.dim() != 2:
            raise ValueError("输入场必须是2D的")

        h, w = field_2d.shape
        device = field_2d.device

        # 1. 应用二维傅里叶变换
        f_transform = torch.fft.fft2(field_2d)
        f_transform_shifted = torch.fft.fftshift(f_transform)
        power_spectrum = torch.abs(f_transform_shifted)**2

        # 2. 创建频率/波数网格
        freq_x = torch.fft.fftshift(torch.fft.fftfreq(w, d=1., device=device))
        freq_y = torch.fft.fftshift(torch.fft.fftfreq(h, d=1., device=device))
        ky, kx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        k_norm = torch.sqrt(kx**2 + ky**2)

        # 3. 使用 SciPy 的 binned_statistic 进行高效的径向平均
        k_norm_np = k_norm.cpu().numpy().flatten()
        power_spectrum_np = power_spectrum.cpu().numpy().flatten()

        # --- 修正点在这里 ---
        # 定义波数的区间（bins）。根据图像分辨率确定分箱数量。
        # 通常分析到奈奎斯特频率，即分辨率的一半。
        num_bins = int(min(h, w) / 2)
        # k_norm 的最大值约为 0.707
        k_max = k_norm_np.max()

        # 计算每个bin中的平均能量
        mean_spectrum, bin_edges, _ = stats.binned_statistic(
            k_norm_np, power_spectrum_np, statistic='mean', bins=num_bins, range=(0, k_max)
        )

        # 计算每个bin的中心点作为波数。
        # 注意：这里的波数是归一化的，为了便于观察，我们可以将其乘以分辨率的一半。
        wavenumbers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 * num_bins * 2

        # 去除可能存在的NaN值（如果某个bin为空）
        wavenumbers = wavenumbers[~np.isnan(mean_spectrum)]
        mean_spectrum = mean_spectrum[~np.isnan(mean_spectrum)]

        return wavenumbers, mean_spectrum


    # <<<<<<<<<<<<<<< 在您的 BaseForecaster 类中添加以下两个方法 >>>>>>>>>>>>>>>

    def calculate_average_spectrum(self, loader, normalizer, **kwargs):
        """
        (计算方法)
        在给定的数据集上进行推理，计算误差的平均能量谱，并返回数据。
        这个方法只计算，不绘图。

        Args:
            loader (DataLoader): 用于评估的数据加载器。
            normalizer: 用于反归一化的 normalizer 对象。

        Returns:
            tuple: (wavenumbers, avg_spectrum)
        """
        print(f"  > 正在为模型 '{self.model_name}' 计算能量谱...")
        self.model.eval()

        all_y_true, all_y_pred = [], []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                all_y_true.append(normalizer.decode(y).cpu())
                all_y_pred.append(normalizer.decode(y_pred).cpu())

        y_true, y_pred = torch.cat(all_y_true, dim=0), torch.cat(all_y_pred, dim=0)

        if y_true.dim() == 4 and y_true.shape[-1] == 1:
            y_true, y_pred = y_true.squeeze(-1), y_pred.squeeze(-1)

        if y_true.dim() != 3:
            raise ValueError(f"期望3D张量 (B, H, W)，但得到形状 {y_true.shape}")

        error_fields = y_pred - y_true

        all_spectra, base_wavenumbers = [], None
        if error_fields.shape[0] > 0:
            base_wavenumbers, _ = self._calculate_radially_averaged_spectrum(error_fields[0])

        for i in range(error_fields.shape[0]):
            wavenumbers, spectrum = self._calculate_radially_averaged_spectrum(error_fields[i])
            interpolated_spectrum = np.interp(base_wavenumbers, wavenumbers, spectrum)
            all_spectra.append(interpolated_spectrum)

        avg_spectrum = np.mean(all_spectra, axis=0)

        print(f"  > 计算完成。")
        # <<< 新增：保存结果的逻辑 >>>
        # 1. 从参数文件中获取数据集和模型名称
        dataset_name = self.data_args['name']
        model_name = self.model_name

        # 2. 构建保存路径
        dataname = dataname = f"{dataset_name}_{self.data_args['sample_factor']}x"
        save_dir = os.path.join('energy', dataname, model_name)
        os.makedirs(save_dir, exist_ok=True) # 确保文件夹存在

        # 3. 定义文件名并保存
        file_path = os.path.join(save_dir, 'spectrum_data.npz')
        np.savez(file_path, wavenumbers=base_wavenumbers, spectrum=avg_spectrum)

        print(f"计算完成。能量谱数据已保存至: {file_path}")
        return base_wavenumbers, avg_spectrum


    # def plot_spectrum_comparison(self, results_list, save_path=None, dpi=120):
    #     """
    #     (绘图方法)
    #     接收一个包含多个模型计算结果的列表，并将它们绘制在同一张图上进行比较。

    #     Args:
    #         results_list (list of dict): 一个列表，每个字典包含一个模型的绘图信息。
    #             - 'label': 在图例中显示的名称 (e.g., 'EDSR')。
    #             - 'data': (wavenumbers, avg_spectrum) 元组，由 calculate_average_spectrum 返回。
    #             - 'color': 线的颜色 (e.g., 'purple')。
    #             - 'marker': 数据点的标记 (e.g., '^')。
    #         save_path (str): 图像保存路径。
    #         dpi (int): 图像的DPI。
    #     """
    #     plt.style.use('default')
    #     fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    #     max_freq = 0
    #     for result in results_list:
    #         label = result['label']
    #         wavenumbers, avg_spectrum = result['data']
    #         color = result.get('color', 'blue')   # .get() 允许有默认值
    #         marker = result.get('marker', 'o')

    #         if wavenumbers is not None and wavenumbers.max() > max_freq:
    #             max_freq = wavenumbers.max()

    #         ax.semilogy(
    #             wavenumbers, avg_spectrum, label=label, color=color,
    #             marker=marker, linestyle='-', markersize=6
    #         )

    #     ax.set_xlabel('Frequency', fontsize=14, fontweight='bold')
    #     ax.set_ylabel('Average Error Energy', fontsize=14, fontweight='bold')
    #     ax.grid(True, which="both", ls="--", color='0.65')
    #     ax.tick_params(axis='both', which='major', labelsize=12)
    #     plt.setp(ax.spines.values(), linewidth=2)
    #     ax.legend(fontsize=12)
    #     ax.set_xlim(0, max_freq)

    #     plt.tight_layout()

    #     if save_path:
    #         print(f"保存比较图至 {save_path}")
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')

    #     plt.show()

    # # 确保 _calculate_radially_averaged_spectrum 辅助函数仍然存在于您的类中



    def plot_spectrum_from_path(self, dataset_path, save_path=None, dpi=120):
        """
        (从路径读取并绘图的方法)
        从指定的数据集结果路径加载所有模型的能量谱数据，并将它们绘制在同一张图上进行比较。

        Args:
            dataset_path (str): 数据集结果的根路径 (e.g., './energy/your_dataset_name').
            save_path (str, optional): 图像保存路径。如果为None，则只显示不保存。
            dpi (int): 图像的DPI。
        """
        if not os.path.isdir(dataset_path):
            print(f"错误：路径 '{dataset_path}' 不存在或不是一个文件夹。")
            return

        print(f"正在从 '{dataset_path}' 加载数据并绘制比较图...")

        # 定义一些颜色和标记，让图更好看
        colors = plt.cm.get_cmap('tab10', 10).colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        # 1. 遍历给定路径下的所有子文件夹（每个子文件夹代表一个模型）
        model_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

        if not model_dirs:
            print(f"警告：在 '{dataset_path}' 中没有找到任何模型的子文件夹。")
            return

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        max_freq = 0

        # 2. 加载每个模型的数据并准备绘图
        for i, model_name in enumerate(model_dirs):
            data_file = os.path.join(dataset_path, model_name, 'spectrum_data.npz')

            if os.path.exists(data_file):
                # 读取保存的 .npz 文件
                data = np.load(data_file)
                wavenumbers = data['wavenumbers']
                avg_spectrum = data['spectrum']

                if wavenumbers is not None and len(wavenumbers) > 0 and wavenumbers.max() > max_freq:
                    max_freq = wavenumbers.max()

                # 绘制该模型的数据
                ax.semilogy(
                    wavenumbers, avg_spectrum,
                    label=model_name,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    linestyle='-',
                    markersize=5
                )
                print(f"  - 已加载并绘制模型: {model_name}")
            else:
                print(f"  - 警告: 在 {os.path.join(dataset_path, model_name)} 中未找到 spectrum_data.npz 文件。")

        # 3. 设置图像的各种属性
        ax.set_xlabel('Frequency (k)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Error Energy Spectrum E(k)', fontsize=14, fontweight='bold')
        ax.set_title(f'Error Energy Spectrum Comparison on {os.path.basename(dataset_path)}', fontsize=16, fontweight='bold')
        ax.grid(True, which="both", ls="--", color='0.65')
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(ax.spines.values(), linewidth=1.5)
        ax.legend(fontsize=12)
        if max_freq > 0:
            ax.set_xlim(0, max_freq)

        plt.tight_layout()

        # 4. 保存图像
        if save_path:
            # 确保保存路径的文件夹存在
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            print(f"保存比较图至 {save_path}")
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')

        plt.show()


    def read_and_print_spectrum_data(self, energy_dataset_path):
        """
        从指定的 energy/数据集 路径读取所有已保存的能量谱数据，
        并按模型名称打印相关信息。

        Args:
            energy_dataset_path (str): 数据集结果的根路径 (e.g., './energy/your_dataset_name').
        """
        if not os.path.isdir(energy_dataset_path):
            print(f"错误：路径 '{energy_dataset_path}' 不存在或不是一个文件夹。")
            return

        dataset_name = os.path.basename(energy_dataset_path)
        print(f"--- 开始从 '{dataset_name}' 数据集路径读取能量谱数据 ---\n")

        # 1. 遍历给定路径下的所有子文件夹（每个子文件夹代表一个模型）
        model_dirs = sorted([d for d in os.listdir(energy_dataset_path) if os.path.isdir(os.path.join(energy_dataset_path, d))])

        if not model_dirs:
            print(f"警告：在 '{energy_dataset_path}' 中没有找到任何模型的子文件夹。")
            return

        # 2. 加载每个模型的数据并打印信息
        for model_name in model_dirs:
            data_file = os.path.join(energy_dataset_path, model_name, 'spectrum_data.npz')

            print(f"[*] 检查模型: {model_name}")

            if os.path.exists(data_file):
                try:
                    # 读取保存的 .npz 文件
                    data = np.load(data_file)
                    wavenumbers = data['wavenumbers']
                    avg_spectrum = data['spectrum']

                    # 打印相关信息
                    print(f"  - 成功加载数据文件: {data_file}")
                    print(f"  - 波数 (wavenumbers) 维度: {wavenumbers.shape}")
                    print(f"  - 平均能谱 (spectrum) 维度: {avg_spectrum.shape}")
                    print(f"  - 波数数据 : {wavenumbers}")
                    print(f"  - 能谱数据 : {avg_spectrum}\n")

                except Exception as e:
                    print(f"  - 错误: 加载或处理文件 {data_file} 时出错: {e}\n")
            else:
                print(f"  - 警告: 未找到 spectrum_data.npz 文件。\n")

        print(f"--- '{dataset_name}' 数据集路径下的所有模型检查完毕 ---")



# ====================================================================================
# >> 最终正确版 v4（修正 clip_denoised 参数）：在您的 BaseForecaster 类中粘贴此方法 <<
# ====================================================================================

    def vis_diffusion_process(self, raw_x, raw_y, normalizer, save_dir,
                              num_snapshots=5, dpi=1000, cmap='viridis', **kwargs):
        """
        可视化扩散模型的逆向去噪过程。此版本通过修正 clip_denoised 参数，
        确保了 progressive 循环中的数据处理与最终推理完全一致，从而解决了数值范围错误的问题。
        """
        # 1. 检查兼容性
        if not hasattr(self, 'base_diffusion') or not hasattr(self.base_diffusion, 'p_sample_loop_progressive'):
            print("错误: 'vis_diffusion_process' 方法仅与包含 'base_diffusion' 属性的扩散模型兼容 (例如 RemgForecaster)。")
            return

        print(f"--- 开始执行扩散过程可视化 ---")
        print(f"所有图像将保存在主文件夹: '{save_dir}'")

        # 2. 准备工作：创建文件夹
        save_dir_samples = os.path.join(save_dir, 'denoised_samples')
        save_dir_xstarts = os.path.join(save_dir, 'xstart_predictions')
        os.makedirs(save_dir_samples, exist_ok=True)
        os.makedirs(save_dir_xstarts, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            # 3. 准备数据并计算正确的颜色范围
            raw_y_decoded = normalizer.decode(raw_y.clone())
            raw_y_for_vlim = raw_y_decoded.cpu().reshape(self.shape)

            vmin = raw_y_for_vlim.min()
            vmax = raw_y_for_vlim.max()
            print(f"  > 已设定全局颜色范围: vmin={vmin:.4f}, vmax={vmax:.4f}")

            # 4. 保存参考图像 (此部分逻辑正确，无需修改)
            gt_image_to_save = raw_y_for_vlim
            fig, ax = plt.subplots(figsize=(gt_image_to_save.shape[1]/100, gt_image_to_save.shape[0]/100))
            ax.imshow(gt_image_to_save, cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')
            plt.savefig(os.path.join(save_dir, "00_ground_truth_high_res.png"), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # ... (其他参考图的保存逻辑也是正确的) ...
            simulated_lr = make_lr_blur(raw_y_decoded.clone().permute(0, 3, 1, 2),
                                        scale=self.data_args.get('sample_factor', 2)).permute(0, 2, 3, 1)
            simulated_lr_image_to_save = simulated_lr[0].cpu().numpy()
            if simulated_lr_image_to_save.shape[-1] == 1: simulated_lr_image_to_save = simulated_lr_image_to_save.squeeze(-1)
            fig, ax = plt.subplots(figsize=(simulated_lr_image_to_save.shape[1]/100, simulated_lr_image_to_save.shape[0]/100))
            ax.imshow(simulated_lr_image_to_save, cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')
            plt.savefig(os.path.join(save_dir, "01_simulated_low_res.png"), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            x = raw_x.to(self.device).permute(0, 3, 1, 2)
            y_for_shape = raw_y.to(self.device).permute(0, 3, 1, 2)
            im_lq = F.interpolate(x, size=y_for_shape.shape[2:], mode='bicubic', align_corners=False)
            lr_unnormalized = normalizer.decode(im_lq.clone().permute(0, 2, 3, 1))
            lr_image_to_save = lr_unnormalized[0].cpu().numpy()
            if lr_image_to_save.shape[-1] == 1: lr_image_to_save = lr_image_to_save.squeeze(-1)
            fig, ax = plt.subplots(figsize=(lr_image_to_save.shape[1]/100, lr_image_to_save.shape[0]/100))
            ax.imshow(lr_image_to_save, cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')
            plt.savefig(os.path.join(save_dir, "02_input_low_res_upscaled.png"), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # 5. 准备循环
            total_timesteps = self.base_diffusion.num_timesteps
            snapshot_indices = np.linspace(0, total_timesteps - 1, num_snapshots, dtype=int).tolist()
            snapshot_indices = sorted(list(set(snapshot_indices)))
            model_kwargs = {'lq': im_lq}
            num_iters = 0

            # 6. 核心：使用正确的 clip_denoised=None 调用循环
            for sample_dict in self.base_diffusion.p_sample_loop_progressive(
                    y=im_lq,
                    model=self.model,
                    first_stage_model=None,
                    noise=None,
                    clip_denoised=None,  # <<< 核心修正点：与 inference 的调用保持完全一致
                    model_kwargs=model_kwargs,
                    device=self.device,
                    progress=True,
            ):
                if num_iters in snapshot_indices:
                    current_timestep = total_timesteps - 1 - num_iters
                    print(f"  > 正在处理快照: 迭代次数 {num_iters} (时间步 t={current_timestep})...")

                    # --- (A) 处理并保存 'sample' ---
                    latent_sample = sample_dict['sample']
                    decoded_tensor = self.base_diffusion.decode_first_stage(latent_sample, first_stage_model=None).permute(0, 2, 3, 1)
                    unnormalized_tensor = normalizer.decode(decoded_tensor)
                    image_to_save = unnormalized_tensor[0].cpu().numpy()
                    if image_to_save.shape[-1] == 1: image_to_save = image_to_save.squeeze(-1)

                    # (打印最终t=0步的数值范围，用于验证)
                    if current_timestep == 0:
                        print(f"    - [t=0 'sample' final range] min: {image_to_save.min():.4f}, max: {image_to_save.max():.4f}")

                    fig, ax = plt.subplots(figsize=(image_to_save.shape[1]/100, image_to_save.shape[0]/100))
                    ax.imshow(image_to_save, cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')

                    save_filename = f"sample_iter_{num_iters:04d}_time_{current_timestep:04d}.png"
                    plt.savefig(os.path.join(save_dir_samples, save_filename), dpi=dpi, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    # --- (B) 处理并保存 'pred_xstart' ---
                    latent_xstart = sample_dict['pred_xstart']
                    decoded_xstart = self.base_diffusion.decode_first_stage(latent_xstart, first_stage_model=None).permute(0, 2, 3, 1)
                    unnormalized_xstart = normalizer.decode(decoded_xstart)
                    xstart_to_save = unnormalized_xstart[0].cpu().numpy()
                    if xstart_to_save.shape[-1] == 1: xstart_to_save = xstart_to_save.squeeze(-1)

                    fig, ax = plt.subplots(figsize=(xstart_to_save.shape[1]/100, xstart_to_save.shape[0]/100))
                    ax.imshow(xstart_to_save, cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')

                    save_filename_xstart = f"pred_x0_iter_{num_iters:04d}_time_{current_timestep:04d}.png"
                    plt.savefig(os.path.join(save_dir_xstarts, save_filename_xstart), dpi=dpi, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                num_iters += 1

        print("--- 可视化流程执行完毕 ---")

    # ==============================================================================
    # >>>>> 在您的 BaseForecaster 类中粘贴以下两个新方法 <<<<<
    # ==============================================================================

    def save_sr_results_separately(self, raw_x, raw_y, normalizer, save_dir, **kwargs):
        """
        (新) 分别保存超分结果。
        对单个样本进行超分，并将以下四种图像分别保存为独立的 .mat 文件：
        1. 模拟低分辨率 (simulated_lr.mat)
        2. 插值低分辨率 (interpolated_lr.mat)
        3. 真实高分辨率 (ground_truth_hr.mat)
        4. 模型超分结果 (super_resolved_hr.mat)

        Args:
            raw_x (torch.Tensor): 模型的输入，低分辨率张量 (1, H_lr, W_lr, C)。
            raw_y (torch.Tensor): 对应的真实高分辨率张量 (1, H_hr, W_hr, C)。
            normalizer: 用于反归一化的 normalizer 对象。
            save_dir (str): 用于保存四个 .mat 文件的文件夹路径 (例如: 'results/sample_01/').
        """
        self.model.eval()

        with torch.no_grad():
            # 将输入数据移动到指定设备
            input_x = raw_x.to(self.device)
            input_y = raw_y.to(self.device)

            # 1. 执行模型推理，得到超分结果 (Super-Resolved HR)
            pred_y_normalized = self.inference(input_x, input_y, **kwargs)
            super_resolved_hr = normalizer.decode(pred_y_normalized)

            # 2. 获取原始高分辨率图像 (Ground Truth HR)
            ground_truth_hr = normalizer.decode(input_y)

            # 3. 生成模拟的低分辨率图像 (Simulated LR)
            scale_factor = self.data_args.get('sample_factor', 2)
            simulated_lr = make_lr_blur(ground_truth_hr.permute(0, 3, 1, 2),
                                        scale=scale_factor).permute(0, 2, 3, 1)

            # 4. 生成插值后的低分辨率图像 (Interpolated LR)
            x_for_interp = input_x.permute(0, 3, 1, 2)
            interpolated_lr_normalized = F.interpolate(
                x_for_interp,
                size=ground_truth_hr.shape[1:3],
                mode='bicubic',
                align_corners=False
            )
            interpolated_lr = normalizer.decode(interpolated_lr_normalized.permute(0, 2, 3, 1))

        # --- 数据后处理：转为Numpy数组并移除批次和通道维度 ---
        simulated_lr_np = simulated_lr[0, :, :, 0].cpu().numpy()
        interpolated_lr_np = interpolated_lr[0, :, :, 0].cpu().numpy()
        ground_truth_hr_np = ground_truth_hr[0, :, :, 0].cpu().numpy()
        super_resolved_hr_np = super_resolved_hr[0, :, :, 0].cpu().numpy()

        # --- 分别保存为 .mat 文件 ---
        os.makedirs(save_dir, exist_ok=True)

        # 定义文件名和对应的数据
        files_to_save = {
            '01_simulated_lr.mat': simulated_lr_np,
            '02_interpolated_lr.mat': interpolated_lr_np,
            '03_ground_truth_hr.mat': ground_truth_hr_np,
            '04_super_resolved_hr.mat': super_resolved_hr_np,
        }

        print(f"--- 开始将结果分别保存至文件夹: {save_dir} ---")
        for filename, data_array in files_to_save.items():
            file_path = os.path.join(save_dir, filename)
            # 每个.mat文件内部的变量名统一为 'data'，方便读取
            savemat(file_path, {'data': data_array})
            print(f"  - 已保存: {filename}")

        print("--- 所有文件保存完毕 ---")


    def visualize_saved_mat_results(self, load_dir, save_path=None, dpi=200, cmap='viridis'):
        """
        (新) 可视化已保存的结果。
        从指定文件夹加载四个 .mat 文件，并将它们并排绘制出来以供检查。

        Args:
            load_dir (str): 存放 .mat 文件的文件夹路径 (例如: 'results/sample_01/').
            save_path (str, optional): 可视化结果图像的保存路径。如果为None，则只显示不保存。
            dpi (int): 图像的DPI。
            cmap (str): 使用的颜色映射。
        """
        print(f"--- 正在从文件夹加载并可视化: {load_dir} ---")

        # 定义需要加载的文件名
        filenames = [
            '01_simulated_lr.mat',
            '02_interpolated_lr.mat',
            '03_ground_truth_hr.mat',
            '04_super_resolved_hr.mat',
        ]

        # 定义图像标题
        titles = [
            'Simulated LR\n(from .mat)',
            'Interpolated LR\n(from .mat)',
            'Ground Truth HR\n(from .mat)',
            'Super-Resolved HR\n(from .mat)'
        ]

        loaded_data = []
        for filename in filenames:
            file_path = os.path.join(load_dir, filename)
            try:
                mat_content = loadmat(file_path)
                # 假设每个 .mat 文件中的变量名都是 'data'
                loaded_data.append(mat_content['data'])
                print(f"  - 成功加载: {filename}")
            except FileNotFoundError:
                print(f"  - 错误: 未找到文件 {file_path}")
                return # 如果有文件缺失，则终止函数
            except KeyError:
                print(f"  - 错误: 在文件 {file_path} 中未找到名为 'data' 的变量。")
                return

        # --- 开始绘图 ---
        # 使用 Ground Truth HR 来设定统一的颜色范围
        vmin = loaded_data[2].min()
        vmax = loaded_data[2].max()

        fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True, dpi=dpi)

        for i, (ax, data, title) in enumerate(zip(axs, loaded_data, titles)):
            # 对 HR 图像使用统一的 vmin 和 vmax
            if i >= 1: # 插值LR, 真值HR, 超分HR
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            else: # 模拟LR，尺寸不同，单独显示
                im = ax.imshow(data, cmap=cmap)

            ax.set_title(title, fontsize=14)
            ax.axis('off')

        # 为 HR 图像添加一个共享的 colorbar
        fig.colorbar(im, ax=axs[1:], shrink=0.8, pad=0.02)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi)
            print(f"--- 可视化图像已保存至: {save_path} ---")

        plt.show()