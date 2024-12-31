import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from utils.model.fgdnet import FGDNet
from utils.model.mynet import MyNet
from utils.model.CombinedModel import CombinedModel, VarNoiseNet, VarMWCNN, ADNet
from utils.model.denoiser.mwcnn import make_model
from utils.model.model_utils import load_best_model
from utils.data.mask import get_mask
from utils.model.eamri import EAMRI
import fastmri

def test(args, model, data_loader, model_num, previous_model):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, _, kspace,_, _, fnames, slices) in data_loader: 
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            if model_num == 2:
                result  = previous_model(kspace, mask)
                # output = model(pred_kspace, sens_maps)
                output = model(result)
                
            elif model_num == 1:
                output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args, model_num):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())
    
    
    if model_num == 1:
        # model = EAMRI()
        model = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
        model.to(device=device)
        load_best_model(args, 1, model)
        previous_model = -1
        
    elif model_num == 2:
        previous_model = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
        previous_model.to(device=device)
        load_best_model(args, model_num -1 , previous_model)
        model = MyNet(num_cascades=args.cascade2, chans=9)##두번째로 학습시킬 모델
        # model = FGDNet()
        model.to(device=device)
        load_best_model(args, model_num, model)
        
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True, isimage = False, input_key = "kspace", acc=8, to_acc = 4, train=False)
    reconstructions, inputs = test(args, model, forward_loader, model_num, previous_model)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
    
################################################################################################################################################################################################################################

