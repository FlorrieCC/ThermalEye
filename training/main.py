from utils import set_random_seeds
from trainer import build_trainer, build_dataloaders
from constants import *
from dataset import ThermalBlinkDataset
from models.get_model import get_model 
from model import OptimizedBlinkClassifier  
import os
import pandas as pd, json
from evaluate import evaluate_model 
import torch

def main():
    set_random_seeds()

    # ====================== [DEBUG] 数据集验证用 ======================
    print("🛠️ 正在加载训练数据...")
    dataset = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=False,
        center_size=CENTER_SIZE,
    )
    print(f"✅ 数据集加载完毕，共 {len(dataset)} 帧")
    print("\n==== [DEBUG] 样本详细信息（前3条）====")
    for i in range(3):
        sample = dataset[i]
        if "timestamp" in sample:
            print(f"\n第 {i+1} 个样本时间戳: {sample['timestamp']:.2f} ms, 标签: {sample['y'].tolist()}")
        else:
            print(f"\n第 {i+1} 个样本 标签: {sample['y'].tolist()}")


    # Ground truth 验证
    print("\n==== [DEBUG] Ground Truth 区间验证（使用验证集）====")
    csv_files = [
        os.path.join(VAL_CSV_DIR, f)
        for f in os.listdir(VAL_CSV_DIR)
        if f.endswith(".csv")
    ]
    for csv_file in csv_files:
        print(f"\n[DEBUG] 检查 CSV 文件: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
        print("▶️ start_offsets:", offsets["start_offsets"])
        print("▶️ end_offsets:", offsets["end_offsets"])

    # ====================== [正式训练部分] ======================
    print("\n🚀 初始化模型: ", MODEL_NAME)
    model = OptimizedBlinkClassifier()
    print(model)
    train_loader, val_loader = build_dataloaders()
    trainer = build_trainer()
    trainer.fit(model, train_loader, val_loader)
    
    # ✅ 保存为统一路径
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.save(model.model.state_dict(), os.path.join(CHECKPOINT_PATH, "tcn_final.pth"))
    
    print("\n📊 开始模型评估...")
    ckpt_path = os.path.join(CHECKPOINT_PATH, "tcn_final.pth")
    evaluate_model(ckpt_path)

if __name__ == '__main__':
    main()
