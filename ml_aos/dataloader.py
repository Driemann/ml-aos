import glob
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class Donuts(Dataset):
 
    def __init__(
        self,
        mode: str = "train",
        nval: int = 256,
        ntest: int = 2 ** 10,
        data_dir: str = "/astro/users/driemann/auxtel_aos/data",
        **kwargs: Any,
    ) -> None:
        
        data_files = glob.glob(f"{data_dir}/*")
        rng = np.random.default_rng(0)
        rng.shuffle(data_files)
        
        test_set = data_files[-ntest:]
        rest=data_files[:-ntest]
        
        val_set = rest[-nval:]
        train_set=rest[:-nval]
              
        if mode == "train":
            self._data_files = train_set
        elif mode == "val":
            self._data_files = val_set
        elif mode == "test":
            self._data_files = test_set
            
    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
         # get the image file
        data_file = self._data_files[idx]

        # load the image
        data = np.load(data_file)
        
        image = data['image'][None, 22:278, 22:278]
        intrafocal = 0 if "extra" in data_file.split("/")[-1] else 1
        dof=data['dof']
        temp=data['temp']
        flux=data['flux']
        background=data['background']

        dof[0] /= 0.001 / np.sqrt(3)
        dof[1] /= 0.001 / np.sqrt(3)
        dof[2] /= 0.0001 / np.sqrt(3)
        output= {
            "image": torch.from_numpy(image).float(),
            "intrafocal": torch.FloatTensor([intrafocal]),
            "dof": torch.from_numpy(dof).float(),
            "temp": torch.from_numpy(temp).float(), 
            "flux": torch.from_numpy(flux).float(), 
            "background": torch.from_numpy(background).float()
        }
        
        return output
