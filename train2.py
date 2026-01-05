import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from model2 import TreatmentResponseModel

class TreatmentDataset10(Dataset):
    def __init__(self, path, label_col="response"):
        self.df = pd.read_csv(path)

        self.df.columns = self.df.columns.astype(str).str.strip()
        if label_col not in self.df.columns:
            raise ValueError(f"'{label_col}' not found. Columns: {list(self.df.columns)}")

        self.label_col = label_col
        self.feature_cols = [f"GENE_{i}" for i in range(10)]  # ✅ only 10 genes

        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing genes: {missing}")

        self.df[self.feature_cols] = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors="coerce").fillna(0).astype(int)

        # normalize
        self.mean = self.df[self.feature_cols].mean()
        self.std = self.df[self.feature_cols].std().replace(0, 1.0)
        self.df[self.feature_cols] = (self.df[self.feature_cols] - self.mean) / self.std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = int(row[self.label_col])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def main():
    here = os.path.dirname(__file__)
    csv_path = os.path.abspath(os.path.join(here, "..", "..", "data", "synthetic", "stage2.csv"))
    print("📄 Loading:", csv_path)

    ds = TreatmentDataset10(csv_path)
    print("✅ Stage-2 CSV Loaded | Rows:", len(ds), "| Features:", len(ds.feature_cols))

    torch.manual_seed(42)
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = TreatmentResponseModel(in_dim=10)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    def acc(m, dl):
        m.eval()
        c, t = 0, 0
        with torch.no_grad():
            for x, y in dl:
                pred = m(x).argmax(dim=1)
                c += (pred == y).sum().item()
                t += y.numel()
        return c / t if t else 0.0

    model.train()
    for epoch in range(60):
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, "done")

    train_acc = acc(model, train_dl)
    test_acc = acc(model, test_dl)

    print(f"Stage-2 Train Accuracy: {train_acc*100:.2f}%")
    print(f"Stage-2 Test Accuracy:  {test_acc*100:.2f}%")

    save_path = os.path.join(here, "stage2_model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "mean": ds.mean.to_dict(),
        "std": ds.std.to_dict(),
        "feature_cols": ds.feature_cols
    }, save_path)
    print("✅ Saved:", save_path)

if __name__ == "__main__":
    main()
