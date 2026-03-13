from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from emg2qwerty.data import EMGSessionData, WindowedEMGDataset
from emg2qwerty.transforms import ToTensor, Compose
import matplotlib.pyplot as plt

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

datasets = [WindowedEMGDataset(hdf5_path=path, transform=ToTensor(), window_length=8000, padding=(0,0), jitter=False) for path in hdf5_paths]

concat_dataset = ConcatDataset(datasets)

dataloader = DataLoader(concat_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=WindowedEMGDataset.collate)

left_data_batches = []
right_data_batches = []

for batch in dataloader:
    inputs = batch["inputs"]
    T, N, B, C = inputs.shape
    flat = inputs.reshape(T * N, B, C).numpy()
    left_data_batches.append(flat[:, 0, :])
    right_data_batches.append(flat[:, 1, :])

left_data = np.concatenate(left_data_batches, axis=0)
right_data = np.concatenate(right_data_batches, axis=0)
global_data = np.hstack((left_data, right_data))

left_data = left_data[:2400000]
right_data = right_data[:2400000]
global_data = global_data[:2400000]

left_data_mean = np.mean(left_data, axis=0)
left_data_std = np.std(left_data, axis=0)

right_data_mean = np.mean(right_data, axis=0)
right_data_std = np.std(right_data, axis=0)

global_data_mean = np.mean(global_data, axis=0)
global_data_std = np.std(global_data, axis=0)

left_data_centered = (left_data - left_data_mean)
right_data_centered = (right_data - right_data_mean)
global_data_centered = (global_data - global_data_mean)

_, l_S, l_Vt = np.linalg.svd(left_data_centered, full_matrices=False)
_, r_S, r_Vt = np.linalg.svd(right_data_centered, full_matrices=False)
_, g_S, g_Vt = np.linalg.svd(global_data_centered, full_matrices=False)

l_V = l_Vt.T
r_V = r_Vt.T
g_V = g_Vt.T

left_ratios = l_S ** 2 / np.sum(l_S ** 2)
right_ratios = r_S ** 2 / np.sum(r_S ** 2)
global_ratios = g_S ** 2 / np.sum(g_S ** 2)


fig, axs = plt.subplots(3, 1) 

axs[0].plot(np.cumsum(left_ratios))
axs[0].axhline(0.9, linestyle='--', color='blue', label='90%')

axs[1].plot(np.cumsum(right_ratios))
axs[1].axhline(0.9, linestyle='--', color='orange', label='90%')

axs[2].plot(np.cumsum(global_ratios))
axs[2].axhline(0.9, linestyle='--', color='green', label='90%')
plt.show()

# output_dir = Path("analysis")
# output_dir.mkdir(exist_ok=True)

# np.save(output_dir / "pca_components_left.npy", l_V)
# np.save(output_dir / "pca_mean_left.npy", left_data_mean)

# np.save(output_dir / "pca_components_right.npy", r_V)
# np.save(output_dir / "pca_mean_right.npy", right_data_mean)

# np.save(output_dir / "pca_components_global.npy", g_V)
# np.save(output_dir / "pca_mean_global.npy", global_data_mean)