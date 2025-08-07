import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.ufgvc import UFGVCDataset

def visualize_samples(dataset_name="cotton80", split="train", num_samples=6, figsize=(15, 10)):
    """
    可視化資料集中的樣本，顯示圖片、label 和 class_name
    
    Args:
        dataset_name (str): 資料集名稱
        split (str): 資料集分割 (train/val/test)
        num_samples (int): 要顯示的樣本數量
        figsize (tuple): 圖片大小
    """
    try:
        # 載入資料集
        print(f"載入 {dataset_name} 資料集 ({split} split)...")
        dataset = UFGVCDataset(
            dataset_name=dataset_name,
            root="./data",
            split=split,
            transform=None  # 不使用轉換，直接顯示原始圖片
        )
        
        print(f"資料集載入成功！總共 {len(dataset)} 個樣本")
        
        # 隨機選擇樣本
        if len(dataset) < num_samples:
            num_samples = len(dataset)
            print(f"資料集樣本不足，調整為顯示 {num_samples} 個樣本")
        
        # 隨機選擇索引
        selected_indices = random.sample(range(len(dataset)), num_samples)
        selected_indices.sort()  # 排序以便查看
        
        print(f"選擇的樣本索引: {selected_indices}")
        
        # 設置圖片顯示
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'{dataset_name.upper()} Dataset - {split.upper()} Split\n隨機選擇 {num_samples} 個樣本', 
                     fontsize=16, fontweight='bold')
        
        # 確保 axes 是 2D 陣列
        if num_samples == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, idx in enumerate(selected_indices):
            row = i // cols
            col = i % cols
            
            # 獲取樣本
            image, label = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            # 顯示圖片
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                # 如果是 tensor，轉換為 numpy
                if hasattr(image, 'numpy'):
                    image_array = image.numpy()
                    if image_array.shape[0] == 3:  # CHW -> HWC
                        image_array = np.transpose(image_array, (1, 2, 0))
                else:
                    image_array = image
            
            # 正規化圖片數值到 [0, 1] 範圍
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            axes[row, col].imshow(image_array)
            axes[row, col].axis('off')
            
            # 設置標題
            title = f"Index: {idx}\n"
            title += f"Label: {label}\n"
            title += f"Class: '{sample_info['class_name']}'\n"
            title += f"Original Label: {sample_info['label']}"
            
            axes[row, col].set_title(title, fontsize=10, ha='center')
        
        # 隱藏多餘的子圖
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存圖片
        output_path = f"sample_visualization_{dataset_name}_{split}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖片已保存至: {output_path}")
        
        plt.show()
        
        # 打印詳細資訊
        print("\n" + "="*60)
        print("詳細樣本資訊:")
        print("="*60)
        
        for i, idx in enumerate(selected_indices):
            sample_info = dataset.get_sample_info(idx)
            image, label = dataset[idx]
            
            print(f"\n樣本 {i+1} (Index: {idx}):")
            print(f"  - 資料集: {sample_info['dataset']}")
            print(f"  - 分割: {sample_info['split']}")
            print(f"  - 原始 Label: {sample_info['label']}")
            print(f"  - Class Name: '{sample_info['class_name']}'")
            print(f"  - 轉換後 Label: {label}")
            print(f"  - 圖片尺寸: {np.array(dataset[idx][0]).shape if hasattr(dataset[idx][0], 'shape') else 'PIL Image'}")
        
        # 顯示資料集統計資訊
        print(f"\n資料集統計:")
        info = dataset.get_dataset_info()
        print(f"  - 總類別數: {info['current_classes']}")
        print(f"  - 當前分割樣本數: {info['current_samples']}")
        print(f"  - 類別範圍: {info['classes'][0]} ~ {info['classes'][-1]}")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

def visualize_multiple_datasets(dataset_names=None, split="train", samples_per_dataset=2):
    """
    可視化多個資料集的樣本
    
    Args:
        dataset_names (list): 資料集名稱列表
        split (str): 資料集分割
        samples_per_dataset (int): 每個資料集顯示的樣本數
    """
    if dataset_names is None:
        dataset_names = ['cotton80', 'soybean']
    
    print(f"比較多個資料集: {dataset_names}")
    
    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"顯示 {dataset_name} 資料集")
        print(f"{'='*50}")
        try:
            visualize_samples(dataset_name, split, samples_per_dataset, figsize=(10, 6))
        except Exception as e:
            print(f"無法載入 {dataset_name}: {e}")

if __name__ == "__main__":
    # 設置中文字體（如果需要）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 設置隨機種子以便重現結果
    random.seed(42)
    np.random.seed(42)
    
    print("UFGVC 資料集樣本可視化工具")
    print("="*50)
    
    # 主要範例：顯示 cotton80 資料集的 6 個樣本
    print("1. 顯示 Cotton80 資料集的 6 個隨機樣本")
    visualize_samples("cotton80", "train", 6)
    
    print("\n" + "="*50)
    
    # 額外範例：顯示 soybean 資料集的樣本
    print("2. 顯示 Soybean 資料集的 6 個隨機樣本")
    try:
        visualize_samples("soybean", "train", 6)
    except Exception as e:
        print(f"無法載入 soybean 資料集: {e}")
    
    print("\n程式執行完成！")
