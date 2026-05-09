# NOTE:
# This project must be run in Google Colab.
# The model downloads English and Spanish speech-command datasets directly
# from online sources and relies on Colab's cloud storage to extract and access
# large audio archives. Colab also provides GPU acceleration for training.
# Running this code locally may cause file-path errors, missing dependencies,
# or storage issues, especially due to the Spanish dataset's nested ZIP structure.
# For reliable and reproducible results, run this notebook exclusively in Colab.

%pip install TorchCodec

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


MSC_REPO = "/content/multilingual-speech-commands-15lang-zip"  # change for local
# Spanish metadata has paths like "data_all/left/es_XXX.wav", so we join with MSC_REPO
# The audio should be under: MSC_REPO/data_all/<command>/es_*.wav

# REPRODUCIBILITY & DEVICE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# COMMAND CLASSES
TARGET_KEYWORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go"
]
CLASSES = TARGET_KEYWORDS[:]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
print("Command classes:", CLASSES)

# 1. LOAD ENGLISH SPEECH COMMANDS
root = "."
base_en = SPEECHCOMMANDS(root=root, download=True)
print(f"Total English audio files: {len(base_en)}")

# 2. LOAD SPANISH METADATA FROM HUGGINGFACE

msc = load_dataset("artur-muratov/multilingual-speech-commands-15lang-zip", "default")

def add_label_and_lang(example):
    """
    Example 'text': "./data_all/go/es_XXXX_nohash_0.wav"
    We parse:
      command = 'go'
      lang    = 'es'
    """
    path = example["text"]
    parts = path.split("/")     
    cmd = parts[2]
    fname = parts[3]
    lang = fname.split("_")[0]
    example["command"] = cmd
    example["lang"] = lang
    return example

msc = msc.map(add_label_and_lang)

def is_spanish_keyword(ex):
    return (ex["lang"] == "es") and (ex["command"] in TARGET_KEYWORDS)

msc_es = {split: ds.filter(is_spanish_keyword) for split, ds in msc.items()}
msc_es_all = concatenate_datasets([msc_es["train"], msc_es["validation"], msc_es["test"]])

print(f"Total Spanish keyword samples (all splits combined): {len(msc_es_all)}")

valid_es_indices = []

for i in range(len(msc_es_all)):
    row = msc_es_all[i]
    rel_path = row["text"].lstrip("./")
    wav_path = os.path.join(MSC_REPO, rel_path)

    if os.path.exists(wav_path):
        valid_es_indices.append(i)

print(f"Spanish samples BEFORE file check: {len(msc_es_all)}")

msc_es_all = msc_es_all.select(valid_es_indices)


# 3. BUILD COMBINED METADATA (EN + ES) + 80/10/10 SPLIT

examples = []  
by_class = {i: [] for i in range(len(CLASSES))}

# English examples
for idx in range(len(base_en)):
    _, _, label_str, *_ = base_en[idx]
    if label_str in TARGET_KEYWORDS:
        c_idx = CLASS_TO_IDX[label_str]
        ex_id = len(examples)
        examples.append({
            "src": "en",
            "src_idx": idx,
            "class_idx": c_idx,
            "label": label_str,
            "lang": "en",
        })
        by_class[c_idx].append(ex_id)

# Spanish examples
for idx in range(len(msc_es_all)):
    row = msc_es_all[idx]
    cmd = row["command"]
    if cmd in TARGET_KEYWORDS:
        c_idx = CLASS_TO_IDX[cmd]
        ex_id = len(examples)
        examples.append({
            "src": "es",
            "src_idx": idx,
            "class_idx": c_idx,
            "label": cmd,
            "lang": "es",
        })
        by_class[c_idx].append(ex_id)

print(f"\nTotal combined examples (EN + ES): {len(examples)}")

# 80/10/10 stratified split
train_ids, val_ids, test_ids = [], [], []

for c_idx, ex_ids in by_class.items():
    random.shuffle(ex_ids)
    n = len(ex_ids)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_ids.extend(ex_ids[:n_train])
    val_ids.extend(ex_ids[n_train:n_train + n_val])
    test_ids.extend(ex_ids[n_train + n_val:])

random.shuffle(train_ids)
random.shuffle(val_ids)
random.shuffle(test_ids)

print("\nStratified 80/10/10 over EN+ES:")
print(f"Train examples: {len(train_ids)}")
print(f"Val examples:   {len(val_ids)}")
print(f"Test examples:  {len(test_ids)}")

# 4. AUDIO → MEL-SPECTROGRAM PIPELINE
SAMPLE_RATE = 16000
N_MELS = 40
MAX_FRAMES = 128

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=160,
    n_mels=N_MELS,
)
db_transform = torchaudio.transforms.AmplitudeToDB()

# 5. BILINGUAL DATASET CLASS (FIXED SPANISH BRANCH)

class BilingualCommandsDataset(Dataset):
    """
    Mixed English (torchaudio) + Spanish (HF) keyword dataset.
    """

    def __init__(self, examples, id_list, base_en, ds_es, subset: str):
        self.examples = examples
        self.id_list = id_list
        self.base_en = base_en
        self.ds_es = ds_es
        self.subset = subset  

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, i):
        ex = self.examples[self.id_list[i]]
        src = ex["src"]          
        src_idx = ex["src_idx"]
        class_idx = ex["class_idx"]

        if src == "en":
            waveform, sample_rate, _, *_ = self.base_en[src_idx]
        else:
            # Spanish sample: load WAV using path from HuggingFace metadata
            row = self.ds_es[src_idx]
            rel_path = row["text"].lstrip("./")     
            wav_path = os.path.join(MSC_REPO, rel_path)

            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Spanish wav not found: {wav_path}")

            waveform, sample_rate = torchaudio.load(wav_path)

        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )

        
        if self.subset == "train":
            shift = int(0.1 * SAMPLE_RATE)
            offset = random.randint(-shift, shift)
            waveform = torch.roll(waveform, shifts=offset, dims=-1)

        # Mel spectrogram → log-mel
        mel = mel_transform(waveform)
        mel_db = db_transform(mel)

        # Normalize per sample
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

        # Fix time dimension
        _, _, T = mel_db.shape
        if T < MAX_FRAMES:
            pad = MAX_FRAMES - T
            mel_db = F.pad(mel_db, (0, pad))
        elif T > MAX_FRAMES:
            start = random.randint(0, T - MAX_FRAMES)
            mel_db = mel_db[:, :, start:start+MAX_FRAMES]

        y = torch.tensor(class_idx, dtype=torch.long)
        return mel_db, y

# build DataLoaders
train_ds = BilingualCommandsDataset(examples, train_ids, base_en, msc_es_all, subset="train")
val_ds   = BilingualCommandsDataset(examples, val_ids,   base_en, msc_es_all, subset="val")
test_ds  = BilingualCommandsDataset(examples, test_ids,  base_en, msc_es_all, subset="test")

print(f"\nTrain samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")
print(f"Test samples:  {len(test_ds)}")

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 6. CNN MODEL + TRAINING

class ElderlyAssistCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.mean(dim=[2, 3])  
        x = self.dropout(x)
        x = self.fc(x)
        return x

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples

EPOCHS = 10
LR = 1e-3

model = ElderlyAssistCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
best_state = None

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer)
    val_loss,   val_acc   = run_epoch(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    print(f"Epoch {epoch:02d} | "
          f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

if best_state is not None:
    model.load_state_dict(best_state)


# 7. FINAL TEST + CONFUSION MATRIX + CURVES

test_loss, test_acc = run_epoch(model, test_loader)
print(f"\nFinal Test: loss={test_loss:.4f}, acc={test_acc*100:.2f}%")
print(f"Total bilingual test cases used: {len(test_ds)}")

# curves
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs_range, history["train_loss"], label="Train loss")
plt.plot(epochs_range, history["val_loss"],   label="Val loss")
plt.axhline(y=test_loss, linestyle="--", label=f"Test loss ({test_loss:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs (Train / Val / Test)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train acc")
plt.plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val acc")
plt.axhline(y=test_acc*100, linestyle="--",
            label=f"Test acc ({test_acc*100:.1f}%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs (Train / Val / Test)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# confusion matrix
model.eval()
all_targets, all_preds = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(1)
        all_targets.extend(yb.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_targets = np.array(all_targets)
all_preds   = np.array(all_preds)

cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)
ax.set_title("Confusion Matrix – Test Set", fontsize=14)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)

tick_marks = np.arange(len(CLASSES))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CLASSES, fontsize=8)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        if val == 0:
            continue
        color = "white" if val > cm.max() / 2.0 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                color=color, fontsize=6)

fig.tight_layout()
plt.show()

# 8. COLAB DEMO: UPLOAD WAV & PREDICT

try:
    from google.colab import files

    def preprocess_waveform_for_model(waveform, sample_rate):
        # waveform: (channels, T) or (T,)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )

        # add batch dimension for CNN
        waveform = waveform.unsqueeze(0) 
        mel = mel_transform(waveform)
        mel_db = db_transform(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

        _, _, _, T = mel_db.shape
        if T < MAX_FRAMES:
            pad = MAX_FRAMES - T
            mel_db = F.pad(mel_db, (0, pad))
        elif T > MAX_FRAMES:
            start = max(0, T - MAX_FRAMES)
            mel_db = mel_db[:, :, :, start:start + MAX_FRAMES]

        return mel_db.to(DEVICE)

    def predict_file(path):
        model.eval()
        waveform, sr = torchaudio.load(path)
        x = preprocess_waveform_for_model(waveform, sr)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top3_p, top3_i = torch.topk(probs, 3)
        print(f"\nFile: {path}")
        print("Top predictions:")
        for rank in range(3):
            idx = int(top3_i[rank].item())
            cls = CLASSES[idx]
            conf = float(top3_p[rank].item()) * 100
            print(f"  {rank+1}. {cls:8s}  ({conf:5.1f}%)")

        best_idx = int(torch.argmax(probs).item())
        best_cls = CLASSES[best_idx]
        print(f"\nPredicted command: {best_cls}")

    def upload_and_predict():
        print("Upload a WAV file of you saying one command (yes/no/up/...):")
        uploaded = files.upload()
        for fname in uploaded.keys():
            predict_file(fname)

    upload_and_predict()

except ImportError:
    pass