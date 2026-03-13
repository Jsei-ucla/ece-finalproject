from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty.transforms import Compose, LogSpectrogram, ToTensor

checkpoint_path = Path(
    "/Users/Suhas/emg2qwerty/relevant_checkpoints/40e_base_tds_best.ckpt"
)

data_root = Path("./data")
train_sessions = [
    "2021-07-21-1626915176-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-05-1622884635-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-05-1622885888-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-05-1622889105-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-04-1622863166-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-04-1622861066-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-02-1622681518-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-03-1622766673-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-03-1622764398-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    "2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
]
hdf5_paths = [data_root / s for s in train_sessions]

num_batches = 500
window_length = 8000
padding = (1800, 200)
batch_size = 32

module = TDSConvCTCModule.load_from_checkpoint(
    str(checkpoint_path), map_location="cpu"
)
module.eval()


transform = Compose([
    ToTensor(fields=["emg_left", "emg_right"]),
    LogSpectrogram(),
])

datasets = [WindowedEMGDataset(hdf5_path=path, transform=transform, window_length=window_length, padding=padding, jitter=False) for path in hdf5_paths]


concat_dataset = ConcatDataset(datasets)

dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=WindowedEMGDataset.collate)

accumulated_scores = []

def _hook(module, inputs, output):
    # x = inputs[0]

    # score = x.abs().mean(dim=(0, 1, 4)).detach().cpu()
    score = output.abs().mean(dim=(0, 1, 4)).detach().cpu()
    accumulated_scores.append(score)

hook_handle = module.model[0].register_forward_hook(_hook)

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs_tensor = batch["inputs"]
        _ = module.model(inputs_tensor)


hook_handle.remove()

scores = torch.stack(accumulated_scores).mean(dim=0).numpy()
ranked = np.argsort(scores, axis=1)[:, ::-1]

print("\n=== Channel Importance Scores ===")
print("Left Band (ranked):")
for rank, ch_idx in enumerate(ranked[0]):   
    print("  Rank", rank+1, ": electrode", ch_idx, ": score=", scores[0, ch_idx])

print("Right Band (ranked):")
for rank, ch_idx in enumerate(ranked[1]):
    print("  Rank", rank+1, ": electrode", ch_idx, ": score=", scores[1, ch_idx])
