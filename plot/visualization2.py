import os
import random
import numpy as np
from PIL import Image
import torch

def save_dataset_grid_clean(dataset, resolution, num_classes, 
                            output_width=512,
                            output_height=256,
                            border_width=2,  # 新增参数：黑边宽度（像素）
                            save_dir='./dataset_grids',
                            prefix='ds'):
    """
    生成 288×576 网格图，同类图像位于同一列（可多列同属一类）
    - 图像不拉伸：每个单元格为正方形
    - 每张小图周围添加黑色边框
    - 自动计算最大可能的单元格尺寸
    """
    import os
    import numpy as np
    import random
    from PIL import Image
    import torch

    os.makedirs(save_dir, exist_ok=True)
    
    # === 1. 确定单元格尺寸和网格行列数 ===
    if resolution == 96:
        cols = 6  # 576 / 96 = 6
    elif resolution == 32:
        cols = 10
    else:
        cols = output_width // resolution
        if cols < 1: cols = 1
    
    cell_size = output_width // cols
    rows = output_height // cell_size
    if rows < 1:
        rows = 1
        cell_size = output_height

    cols = output_width // cell_size
    actual_width = cols * cell_size
    actual_height = rows * cell_size

    # === 2. 按列分配类别 ===
    base_cols = cols // num_classes
    extra_cols = cols % num_classes
    cols_per_class = [base_cols + (1 if i < extra_cols else 0) for i in range(num_classes)]
    assert sum(cols_per_class) == cols, "列分配错误"

    # === 3. 获取标签 ===
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif hasattr(dataset, 'imgs'):
        targets = np.array([lbl for _, lbl in dataset.imgs])
    else:
        targets = np.array([dataset[i][1] for i in range(min(10000, len(dataset)))])

    # === 4. 创建画布 ===
    canvas = np.ones((actual_height, actual_width, 3), dtype=np.uint8) * 255

    current_col = 0
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) == 0:
            raise ValueError(f"类别 {cls} 无样本")
        
        n_cols_for_cls = cols_per_class[cls]
        total_cells_needed = n_cols_for_cls * rows
        sampled_indices = [random.choice(cls_indices) for _ in range(total_cells_needed)]

        # 填充这些列
        for col_offset in range(n_cols_for_cls):
            col_idx = current_col + col_offset
            for row_idx in range(rows):
                sample_idx = col_offset * rows + row_idx
                img_idx = sampled_indices[sample_idx % len(sampled_indices)]
                
                img, _ = dataset[img_idx]
                
                # 转为 HWC uint8
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).numpy()
                    if img_np.dtype == np.float32:
                        img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.array(img)
                    if img_np.ndim == 2:
                        img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                    if img_np.dtype != np.uint8:
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                
                # 处理通道顺序：确保是 HWC 格式
                if img_np.shape[0] in [1, 3] and img_np.shape[0] < img_np.shape[1]:
                    img_np = img_np.transpose(1, 2, 0)
                
                # === 关键修改：添加黑边 ===
                # 1. 创建带黑边的单元格（黑色背景）
                cell_with_border = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                
                # 2. 计算内部图像尺寸（扣除黑边）
                inner_size = cell_size - 2 * border_width
                if inner_size <= 0:
                    raise ValueError(f"border_width ({border_width}) 过大，导致内部图像尺寸无效 (cell_size={cell_size})")
                
                # 3. 调整图像到内部尺寸
                img_pil = Image.fromarray(img_np)
                img_resized = img_pil.resize((inner_size, inner_size), Image.BICUBIC)
                img_inner = np.array(img_resized)
                
                # 4. 将调整后的图像居中放置在黑色背景上
                y_start = border_width
                x_start = border_width
                cell_with_border[y_start:y_start+inner_size, x_start:x_start+inner_size] = img_inner

                # === 放入主画布 ===
                y0 = row_idx * cell_size
                x0 = col_idx * cell_size
                canvas[y0:y0+cell_size, x0:x0+cell_size] = cell_with_border

        current_col += n_cols_for_cls

    # === 5. 裁剪并保存 ===
    img_pil = Image.fromarray(canvas)
    img_pil = img_pil.crop((0, 0, output_width, output_height))
    
    filename = f"{prefix}_{resolution}x{resolution}_{num_classes}cls_{output_height}x{output_width}_bw{border_width}.png"
    save_path = os.path.join(save_dir, filename)
    img_pil.save(save_path, format='PNG', dpi=(300, 300))

    print(f"✓ {filename}")
    print(f"  输出: {output_height}×{output_width} | 网格: {rows}行 × {cols}列 | 单元格: {cell_size}×{cell_size} | 黑边: {border_width}px")
    print(f"  每类列数: {cols_per_class}")
    return save_path

def save_dataset_grid_clean_(dataset, resolution, num_classes, 
                            output_width=576,
                            output_height=288,
                            save_dir='./dataset_grids',
                            prefix='ds'):
    """
    生成 288×576 网格图，同类图像位于同一列（可多列同属一类）
    - 图像不拉伸：每个单元格为正方形（边长 = min(output_height, output_width) 的约数）
    - 自动计算最大可能的单元格尺寸
    - 每列垂直堆叠多个样本（最多 rows_per_col 个）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # === 1. 确定单元格尺寸和网格行列数 ===
    # 优先让单元格尽可能大，且为正方形
    max_cell_size = min(output_height, output_width)
    # 找到能整除 output_width 和 output_height 的最大 cell_size <= 原始 resolution（可选）
    # 但为通用性，我们直接基于输出尺寸计算
    # 尝试从 max_cell_size 向下找能整除两者的值，或近似
    # 简化：固定 cell_size 为 output_width 的因数，并使 rows = output_height // cell_size
    
    # 更简单策略：先确定列数（基于 width），再确定行数
    if resolution == 96:
        cols = 6  # 576 / 96 = 6
    if resolution == 32:
        cols = 10
    else:
        cols = output_width // resolution
        if cols < 1: cols = 1
    
    cell_size = output_width // cols  # 单元格宽度（也是高度，正方形）
    rows = output_height // cell_size
    if rows < 1:
        rows = 1
        cell_size = output_height  # fallback

    # 重新调整 cols 以匹配新 cell_size（如果 needed）
    cols = output_width // cell_size
    actual_width = cols * cell_size
    actual_height = rows * cell_size

    # === 2. 按列分配类别（每列一个类别，可重复）===
    base_cols = cols // num_classes
    extra_cols = cols % num_classes
    cols_per_class = [base_cols + (1 if i < extra_cols else 0) for i in range(num_classes)]
    assert sum(cols_per_class) == cols, "列分配错误"

    # === 3. 获取标签 ===
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif hasattr(dataset, 'imgs'):
        targets = np.array([lbl for _, lbl in dataset.imgs])
    else:
        targets = np.array([dataset[i][1] for i in range(min(10000, len(dataset)))])

    # === 4. 创建画布 ===
    canvas = np.ones((actual_height, actual_width, 3), dtype=np.uint8) * 255

    current_col = 0
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) == 0:
            raise ValueError(f"类别 {cls} 无样本")
        
        n_cols_for_cls = cols_per_class[cls]
        total_cells_needed = n_cols_for_cls * rows
        # 采样足够多的图像（可重复）
        sampled_indices = [random.choice(cls_indices) for _ in range(total_cells_needed)]

        # 填充这些列
        for col_offset in range(n_cols_for_cls):
            col_idx = current_col + col_offset
            for row_idx in range(rows):
                sample_idx = col_offset * rows + row_idx
                img_idx = sampled_indices[sample_idx % len(sampled_indices)]
                
                img, _ = dataset[img_idx]
                
                # 转为 HWC uint8
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).numpy()
                    if img_np.dtype == np.float32:
                        img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.array(img)
                    if img_np.ndim == 2:
                        img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                    img_np *= 255
                    if img_np.dtype != np.uint8:
                        img_np = img_np.astype(np.uint8)
                    img_np = img_np.transpose(1, 2, 0) 
                # print("img_np.shape:", img_np.shape)
                # print("img_np.dtype:", img_np.dtype)
                # print("img_np.min(), max():", img_np.min(), img_np.max())
                # Resize 到 cell_size × cell_size（保持比例？这里用填充或裁剪）
                # 方案：先 resize 到 (cell_size, cell_size) with BICUBIC —— 允许轻微变形，但简单
                # 若需保持比例，可用 letterbox/pad，但会引入空白。此处按原逻辑使用 resize（可能轻微变形）
                # 但至少是正方形，不会像之前那样 32x288 极端拉伸
                img_pil = Image.fromarray(img_np)
                img_resized = img_pil.resize((cell_size, cell_size), Image.BICUBIC)
                img_final = np.array(img_resized)

                # 放入画布
                y0 = row_idx * cell_size
                x0 = col_idx * cell_size
                canvas[y0:y0+cell_size, x0:x0+cell_size] = img_final

        current_col += n_cols_for_cls

    # === 5. 裁剪并保存 ===
    img_pil = Image.fromarray(canvas)
    # 裁剪到目标尺寸（防御性）
    img_pil = img_pil.crop((0, 0, output_width, output_height))
    
    filename = f"{prefix}_{resolution}x{resolution}_{num_classes}cls_{output_height}x{output_width}.png"
    save_path = os.path.join(save_dir, filename)
    img_pil.save(save_path, format='PNG', dpi=(300, 300))

    print(f"✓ {filename}")
    print(f"  输出: {output_height}×{output_width} | 网格: {rows}行 × {cols}列 | 单元格: {cell_size}×{cell_size}")
    print(f"  每类列数: {cols_per_class}")
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
            # output_size=output_size,
            # grid_96_override=6 if res == 96 else None,  # 仅96强制6×6
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

    cifar = load("exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=32, nc=10)
    camelyon = load("exp/lora_camelyon_96_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=96, nc=2)
    oct = load("exp/lora_octmnist_128_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=128, nc=4)
    celeba = load("exp/lora_celeba_male_256_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen", size=256, nc=2)

    from data.stylegan3.dataset import ImageFolderDataset

    cifar = ImageFolderDataset("dataset/cifar10/train_32.zip", 32, 3, use_labels=True)
    camelyon = ImageFolderDataset("exp/train_96.zip", 96, 3, use_labels=True)
    oct = ImageFolderDataset("dataset/octmnist/train_128.zip", 128, 3, use_labels=True)
    celeba = ImageFolderDataset("dataset/celeba/train_256_Male.zip", 256, 3, use_labels=True)

    from torch.utils.data import Subset

    indices = torch.randperm(len(cifar)).tolist()
    cifar = Subset(cifar, indices)
    indices = torch.randperm(len(camelyon)).tolist()
    camelyon = Subset(camelyon, indices)
    indices = torch.randperm(len(oct)).tolist()
    oct = Subset(oct, indices)
    indices = torch.randperm(len(celeba)).tolist()
    celeba = Subset(celeba, indices)

    # 批量生成
    datasets_info = [
        (cifar, 32, 10),
        (camelyon, 96, 2),
        (oct, 128, 4),
        (celeba, 128, 2),
    ]
    
    for i in range(10):
        batch_save_clean(datasets_info, output_size=512, save_dir='./real_dataset_grids_{}'.format(i))