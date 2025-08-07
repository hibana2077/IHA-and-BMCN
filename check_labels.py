import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.ufgvc import UFGVCDataset

def check_label_mapping(dataset_name="cotton80"):
    """檢查資料集的 label 和 class_name 對應關係"""
    print(f"=== 檢查 {dataset_name} 資料集 ===")
    
    try:
        # 載入資料集
        dataset = UFGVCDataset(dataset_name=dataset_name, root="./data", split="train")
        
        # 讀取原始 parquet 檔案
        df = pd.read_parquet(dataset.filepath)
        
        print(f"總樣本數: {len(df)}")
        print(f"欄位: {list(df.columns)}")
        
        # 檢查 label 和 class_name 的對應關係
        print("\n=== 原始資料中的 label 和 class_name 對應 ===")
        label_class_mapping = df[['label', 'class_name']].drop_duplicates().sort_values('label')
        print(label_class_mapping)
        
        print(f"\nlabel 範圍: {df['label'].min()} ~ {df['label'].max()}")
        print(f"class_name 範圍: {df['class_name'].min()} ~ {df['class_name'].max()}")
        print(f"unique labels: {len(df['label'].unique())}")
        print(f"unique class_names: {len(df['class_name'].unique())}")
        
        # 檢查資料集物件中的類別對應
        print(f"\n=== UFGVCDataset 中的類別對應 ===")
        print(f"classes (排序後): {dataset.classes}")
        print(f"class_to_idx mapping: {dataset.class_to_idx}")
        
        # 檢查是否有問題
        print(f"\n=== 潛在問題檢查 ===")
        
        # 檢查原始 label 是否從 0 開始
        original_labels = sorted(df['label'].unique())
        print(f"原始 labels: {original_labels}")
        if original_labels[0] != 0:
            print(f"⚠️  警告: 原始 label 不是從 0 開始，而是從 {original_labels[0]} 開始")
        
        # 檢查 class_name 是否為數字且從 1 開始
        try:
            class_names_as_int = [int(x) for x in df['class_name'].unique()]
            class_names_sorted = sorted(class_names_as_int)
            print(f"class_name 作為數字: {class_names_sorted}")
            if class_names_sorted[0] == 1:
                print(f"⚠️  警告: class_name 從 1 開始，但 PyTorch 通常期望從 0 開始")
        except ValueError:
            print("class_name 不是純數字")
        
        # 檢查實際資料載入時的標籤
        print(f"\n=== 實際資料載入檢查 ===")
        for i in range(min(5, len(dataset))):
            image, label = dataset[i]
            sample_info = dataset.get_sample_info(i)
            print(f"樣本 {i}: 原始label={sample_info['label']}, class_name='{sample_info['class_name']}', 轉換後label={label}")
        
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    # 檢查 cotton80 資料集
    check_label_mapping("cotton80")
    
    print("\n" + "="*50)
    
    # 如果有其他資料集也檢查
    try:
        check_label_mapping("soybean")
    except:
        print("無法載入 soybean 資料集")
