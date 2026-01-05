import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from model import DiseaseModel  # DiseaseModel = DiseaseModel_WideDeepBN in model.py


# ----------------- Dataset -----------------
class ImmuneDataset10(Dataset):
    """
    Uses ONLY these 10 features (same as frontend):
    Age, Gender, WBC, CRP, ESR, ANA, RF, Anti-dsDNA, IL-6, TNF-α
    """

    def __init__(self, path, label_col="Diagnosis"):
        self.df = pd.read_csv(path, encoding="utf-8-sig")

        # clean column names
        self.df.columns = (
            self.df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # fix label col
        if label_col not in self.df.columns:
            candidates = [c for c in self.df.columns if c.strip().lower() == label_col.lower()]
            if candidates:
                label_col = candidates[0]
            else:
                raise ValueError(f"Label column '{label_col}' not found!")

        self.label_col = label_col
        self.df[self.label_col] = self.df[self.label_col].astype(str).str.strip()

        # classes
        classes = sorted(self.df[self.label_col].unique().tolist())
        self.class_to_id = {c: i for i, c in enumerate(classes)}
        self.classes = classes

        # gender -> numeric
        if "Gender" in self.df.columns:
            g = self.df["Gender"].astype(str).str.strip().str.lower()
            self.df["Gender"] = (
                g.map({"male": 0, "m": 0, "female": 1, "f": 1})
                .fillna(-1)
                .astype(float)
            )

        # ---- IMPORTANT: choose 10 columns from your dataset ----
        # If your dataset column names are different, update these names exactly.
        self.feature_cols = [
    "Age",
    "Gender",
    "WBC",
    "Neutrophils",
    "Lymphocytes",
    "PLT_Count",
    "ANA",
    "RF",
    "ESR",
    "CRP"
]


        # Try to auto-fix common variations in column names
        rename_map = {}
        for c in self.df.columns:
            cl = c.lower()
            if cl in ["wbc_count", "wbc count", "wbc"]:
                rename_map[c] = "WBC"
            elif cl in ["crp (mg/l)", "crp_mg_l", "crp"]:
                rename_map[c] = "CRP"
            elif cl in ["esr (mm/hr)", "esr_mm_hr", "esr"]:
                rename_map[c] = "ESR"
            elif cl in ["ana (0=negative, 1=positive)", "ana"]:
                rename_map[c] = "ANA"
            elif cl in ["rheumatoid factor", "rf"]:
                rename_map[c] = "RF"
            elif cl in ["anti-dsdna", "dsdna", "dsdna (0/1)", "anti dsdna"]:
                rename_map[c] = "dsDNA"
            elif cl in ["il-6", "il6"]:
                rename_map[c] = "IL6"
            elif cl in ["tnf-α", "tnf", "tnf alpha", "tnf-a"]:
                rename_map[c] = "TNF"

        self.df = self.df.rename(columns=rename_map)

        # Ensure all 10 exist
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in CSV: {missing}\n"
                f"Available columns: {list(self.df.columns)}\n"
                f"Fix names in feature_cols/rename_map."
            )

        # numeric coercion
        self.df[self.feature_cols] = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # normalization (z-score)
        self.mean = self.df[self.feature_cols].mean()
        self.std = self.df[self.feature_cols].std().replace(0, 1.0)
        self.df[self.feature_cols] = (self.df[self.feature_cols] - self.mean) / self.std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = row[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = self.class_to_id[row[self.label_col]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ----------------- TRAIN -----------------
def main():
    here = os.path.dirname(__file__)
    csv_path = os.path.abspath(os.path.join(here, "..", "..", "data", "synthetic", "stage1.csv"))

    print("📄 Loading CSV from:", csv_path)

    ds = ImmuneDataset10(csv_path)
    print("✅ CSV Loaded")
    print("Rows:", len(ds))
    print("Features:", len(ds.feature_cols))
    print("Classes:", ds.classes)

    torch.manual_seed(42)
    np.random.seed(42)

    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = DiseaseModel(in_dim=10, num_classes=len(ds.classes))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # class weights
    labels = torch.tensor([ds[i][1].item() for i in range(len(ds))], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(ds.classes)).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

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
    for epoch in range(120):
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} done")

    train_acc = acc(model, train_dl)
    test_acc = acc(model, test_dl)

    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Test accuracy:  {test_acc*100:.2f}%")

    # ✅ save everything needed for prediction
    save_path = os.path.join(here, "stage1_model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "classes": ds.classes,
        "feature_cols": ds.feature_cols,
        "mean": ds.mean.to_dict(),
        "std": ds.std.to_dict(),
    }, save_path)

    print("✅ Model saved:", save_path)


if __name__ == "__main__":
    main()
