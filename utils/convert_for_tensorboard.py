import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

detect_dir = 'model/runs/detect'
train_dirs = [d for d in os.listdir(detect_dir) if d.startswith('train')]

for train_dir in train_dirs:
    csv_path = os.path.join(detect_dir, train_dir, 'results.csv')
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    writer = SummaryWriter(log_dir=os.path.join(detect_dir, train_dir, 'tblog'))

    for epoch, row in df.iterrows():
        for metric, val in row.items():
            writer.add_scalar(metric, val, epoch)

    writer.close()
    print(f'✅ Created TensorBoard logs for {train_dir}')
