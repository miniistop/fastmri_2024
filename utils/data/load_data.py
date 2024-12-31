import h5py
import random
from utils.data.transforms import DataTransform, DataTransform_img
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from utils.data.mask import get_mask


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, train = True):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key ## 원래 그냥 target_eky
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            if train:
                image_files_aug = list(Path("../result/aug_image/").iterdir())
                part_image = sorted(image_files_aug) + sorted(image_files)
            else:
                part_image = sorted(image_files)
            for fname in part_image:
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        if train:
            kspace_files_aug = list(Path("../result/aug_kspace/").iterdir())
            part_kspace = sorted(kspace_files_aug) + sorted(kspace_files)
        else:
            part_kspace = sorted(kspace_files)
        for fname in part_kspace:
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
            


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:

            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders( data_path, args, isimage = False, input_key = "kspace", shuffle=False, isforward=False, acc=0, to_acc = 0, train = True):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = args.max_key ############원래 -1이었던 부분
        target_key_ = args.target_key
        
    if isimage:
        data_storage = SliceData_img(
            root=data_path,
            transform=DataTransform_img(isforward, max_key_),
            input_key=input_key,
            target_key=target_key_,
            forward = isforward
        )
    else:
        data_storage = SliceData(
            root=data_path,
            transform=DataTransform(isforward, max_key_, acc, to_acc, train),
            input_key=input_key,
            target_key=target_key_,
            forward = isforward,
            train = train
        )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle
    )
    return data_loader


def synthesis_data(root):
    """
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    kspace_files = list(Path( root / "kspace").iterdir())
    part_kspace = sorted(kspace_files)
    for fname in part_kspace:
        with h5py.File(fname, 'r') as original_h5:
            new_h5_path = Path(f"../result/aug_kspace/aug_{fname.name}" )
            attrs = dict(original_h5.attrs)
            with h5py.File(new_h5_path, 'w') as new_h5:
                for key in original_h5.keys():
                    if key == 'mask':
                        if "acc8" in str(fname.name):
                            new_data = get_mask(len(original_h5['mask']), 4)
                        elif "acc4" in str(fname.name):
                            new_data = get_mask(len(original_h5['mask']), 8)
                        
                        new_h5.create_dataset(key, data=new_data, dtype='float32')
                    else:
                        original_h5.copy(original_h5[key], new_h5)
                for key, value in attrs.items():
                    new_h5.attrs[key] = value  
                        
    image_files = list(Path(root / "image").iterdir())
    part_image = sorted(image_files)
    for fname in part_image:
        with h5py.File(fname, 'r') as original_h5:
            new_h5_path = Path(f"../result/aug_image/aug_{fname.name}" )
            attrs = dict(original_h5.attrs)
            with h5py.File(new_h5_path, 'w') as new_h5:
                for key in original_h5.keys():
                    original_h5.copy(original_h5[key], new_h5)
                for key, value in attrs.items():
                    new_h5.attrs[key] = value    
    
    print("=================================================================================")
    print("===========================DATA_AUGMENTATION_COMPLETE============================")
    print("=================================================================================")