"""
Dataset for keyboard LM training.

Each sample mirrors exactly what FUTO's runtime feeds the model at inference
(org_futo_inputmethod_latin_xlm_LanguageModel.cpp PredictNextWord /
PredictCorrection): the context is tokenized, BOS (token id 1) is prepended,
and every word carries a whitespace-suffix token. We therefore tokenize each
line as [BOS] + sp.EncodeAsIds(line + " ") and train the causal next-token
objective over it.
"""
import torch
import sentencepiece as spm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

BOS_ID = 1   # FUTO hardcodes BOS to token id 1


class KeyboardDataset(Dataset):
    def __init__(
        self,
        text_file: Path,
        sp_model_path: Path,
        max_seq_len: int = 256,
        max_samples: int | None = None,   # None = use the whole corpus
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(sp_model_path))
        self.max_seq_len = max_seq_len

        self.samples: list[list[int]] = []
        with open(text_file, encoding="utf-8") as f:
            for line in f:
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                # trailing space so the final word gets its ▁ suffix, exactly
                # like the runtime's tokenize(trim(context) + " ")
                ids = self.sp.EncodeAsIds(line + " ")
                ids = [BOS_ID] + ids
                for start in range(0, len(ids) - 1, max_seq_len - 1):
                    chunk = ids[start:start + max_seq_len]
                    if len(chunk) >= 3:
                        self.samples.append(chunk)
                        if max_samples is not None and len(self.samples) >= max_samples:
                            break
        print(f"  loaded {len(self.samples):,} samples from {Path(text_file).name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.full((len(xs), max_len), 0, dtype=torch.long)  # pad id 0
    y_pad = torch.full((len(ys), max_len), -1, dtype=torch.long)  # ignored in loss
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    return x_pad, y_pad


def make_loader(
    text_file: Path,
    sp_model_path: Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    ds = KeyboardDataset(text_file, sp_model_path, **kwargs)
    pin = torch.cuda.is_available()   # MPS doesn't support pin_memory
    # MPS: multiprocessing workers cause spawn overhead — run single-threaded
    workers = 0 if torch.backends.mps.is_available() else num_workers
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_fn, pin_memory=pin)
