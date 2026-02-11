import numpy as np
import torch
from PIL import Image
import random
import os

def save_dataset_grid_clean(dataset, resolution, num_classes, 
                            output_size=576,
                            grid_96_override=6,  # 96分辨率强制6×6
                            save_dir='./dataset_grids',
                            prefix='ds'):
    """
    生成严格576×576正方形网格（无边框/无文字/无白边）
    - 96分辨率: 强制6×6网格 (96×6=576)
    - 其他分辨率: 自动计算网格数，必要时resize单图填满576px
    - 保证同类样本严格位于同一行
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # === 1. 确定网格大小和单图显示尺寸 ===
    if resolution == 96 and grid_96_override is not None:
        grid_size = grid_96_override  # 6
    else:
        # 计算最接近的整数网格（优先向下取整避免溢出）
        grid_size = output_size // resolution
        if grid_size < 1:
            grid_size = 1
    
    # 单图在输出中的实际尺寸（确保 grid_size * display_size = output_size）
    display_size = output_size // grid_size
    
    # === 2. 按类别分配行数（严格保证每行单一类别）===
    base_rows = grid_size // num_classes
    extra_rows = grid_size % num_classes
    rows_per_class = [base_rows + (1 if i < extra_rows else 0) for i in range(num_classes)]
    assert sum(rows_per_class) == grid_size, "行分配错误"
    
    # === 3. 获取类别索引 ===
    # 支持多种targets属性名
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif hasattr(dataset, 'imgs'):  # ImageFolder
        targets = np.array([lbl for _, lbl in dataset.imgs])
    else:
        # 通用方案：遍历前10k样本构建targets
        targets = np.array([dataset[i][1] for i in range(min(10000, len(dataset)))])
    
    # === 4. 为每类采样并构建画布 ===
    canvas = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255  # 白色背景 [0,255]
    
    current_row = 0
    for cls in range(num_classes):
        # 获取该类所有索引
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) == 0:
            raise ValueError(f"类别 {cls} 无样本")
        
        rows_for_cls = rows_per_class[cls]
        samples_needed = rows_for_cls * grid_size
        
        # 采样（可重复）
        sampled_indices = [random.choice(cls_indices) for _ in range(samples_needed)]
        
        # 填充该类对应的行
        for row_offset in range(rows_for_cls):
            global_row = current_row + row_offset
            for col in range(grid_size):
                idx = row_offset * grid_size + col
                img_idx = sampled_indices[idx % len(sampled_indices)]
                
                # 获取图像
                img, _ = dataset[img_idx]
                
                # 转换为HWC numpy [0,255]
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).numpy()
                    if img_np.dtype == np.float32:
                        img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.array(img)
                    if img_np.ndim == 2:  # 灰度转RGB
                        img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                    if img_np.dtype != np.uint8:
                        img_np = img_np.astype(np.uint8)
                
                # Resize到display_size（保持原始比例，但填满正方形）
                # 使用PIL高质量resize
                img_pil = Image.fromarray(img_np)
                img_resized = img_pil.resize((display_size, display_size), Image.BICUBIC)
                img_final = np.array(img_resized)
                
                # 粘贴到画布
                y0 = global_row * display_size
                x0 = col * display_size
                canvas[y0:y0+display_size, x0:x0+display_size] = img_final
        
        current_row += rows_for_cls
    
    # === 5. 保存纯图像（无边框/无文字）===
    img_pil = Image.fromarray(canvas)
    # 严格裁剪到576×576（防御性）
    img_pil = img_pil.crop((0, 0, output_size, output_size))
    
    filename = f"{prefix}_{resolution}x{resolution}_{num_classes}cls_{grid_size}x{grid_size}.png"
    save_path = os.path.join(save_dir, filename)
    img_pil.save(save_path, format='PNG', dpi=(300, 300))
    
    print(f"✓ {filename}")
    print(f"  输出: {output_size}×{output_size} | 网格: {grid_size}×{grid_size} | 单图显示尺寸: {display_size}×{display_size}")
    print(f"  每类行数分配: {rows_per_class}")
    return save_path


# ============ 批量处理四个数据集 ============
def batch_save_clean(datasets_info, output_size=576, save_dir='./dataset_grids'):
    """
    批量生成无白边正方形网格
    
    Args:
        datasets_info: [(dataset, resolution, num_classes), ...]
        output_size: 统一输出尺寸（96×6=576）
    """
    saved_paths = []
    for i, (dataset, res, n_cls) in enumerate(datasets_info):
        path = save_dataset_grid_clean(
            dataset=dataset,
            resolution=res,
            num_classes=n_cls,
            output_size=output_size,
            grid_96_override=6 if res == 96 else None,  # 仅96强制6×6
            save_dir=save_dir,
            prefix=f'ds{i+1}'
        )
        saved_paths.append(path)
    
    print("\n" + "="*60)
    print(f"✅ 全部 {len(saved_paths)} 张图像已保存至: {os.path.abspath(save_dir)}")
    print("   所有图像严格 576×576 正方形，无白边、无文字、无坐标轴")
    print("="*60)
    return saved_paths


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例：替换为您的实际数据集
    # 示例：加载您的四个数据集
    from data.dataset_loader import load_data, MemmapDataset

    def load(path, c=3, size=32, nc=10):
        if os.path.exists(os.path.join(path, "gen.npz")):
            syn = np.load(os.path.join(path, "gen.npz"))

            syn_data, syn_labels = syn["x"], syn["y"]

            np.save(os.path.join(path, "syn_images.npy"), syn_data)
            np.save(os.path.join(path, "syn_labels.npy"), syn_labels)

            os.remove(os.path.join(path, "gen.npz"))

            del syn_data, syn_labels, syn
            import gc; gc.collect()
        syn_dataset = MemmapDataset(os.path.join(path, "syn_images.npy"), os.path.join(path, "syn_labels.npy"), c=c, size=size, num_classes=nc)
        return syn_dataset

    cifar = load("/p/fzv6enresearch/gap/exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=32, nc=10)
    camelyon = load("/p/fzv6enresearch/gap/exp/lora_camelyon_96_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=96, nc=2)
    oct = load("/p/fzv6enresearch/gap/exp/lora_octmnist_128_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=128, nc=4)
    celeba = load("/p/fzv6enresearch/gap/exp/lora_celeba_male_256_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=256, nc=2)

    # 批量生成
    datasets_info = [
        (cifar, 32, 10),
        (camelyon, 96, 2),
        (oct, 128, 4),
        (celeba, 256, 2),
    ]
    
    for i in range(1):
        batch_save_clean(datasets_info, output_size=512, save_dir='./dataset_grids_{}'.format(i))