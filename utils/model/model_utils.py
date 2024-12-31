import torch
import pathlib
from pathlib import Path

def load_best_model(args, model_index, model, optimizer=None):
    model_path = args.exp_dir / f'best_model{model_index}.pt'
    posix_path = Path(model_path).as_posix()
    checkpoint = torch.load(posix_path, map_location='cpu')
    print("======================================================================================================================")
    print(f"|| load trained model... || file name : best_model{model_index}.pt  ||")
    print(f"|| load trained model... || checkpoint epoch : {checkpoint['epoch']} || best validation loss : {checkpoint['best_val_loss'].item()}  ||")
    print("======================================================================================================================")
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        return epoch
    
    
    

    