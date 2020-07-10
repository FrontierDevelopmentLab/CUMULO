import numpy as np
import torch
import os
import sys

from src.prediction_utils import save_labels
from src.loader import CumuloDataset
from src.utils import make_directory, Normalizer, TileExtractor

def load_iresnet(model_dir, use_cuda):
    
    # load model
    model_path = os.path.join(model_dir, "model.t7")
    model = torch.load(model_path)["model"]

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        torch.backends.cudnn.enabled = False

    model.eval()

    return model

def predict_tiles(model, tiles, use_cuda):
    
    inputs = torch.from_numpy(tiles).float()

    if use_cuda:
        inputs = inputs.cuda()
    
    logits, z, _, _ = model(inputs)

    labels = torch.argmax(logits.data, 1)

    return labels.cpu().detach().numpy(), z[-1].cpu().detach().numpy()

if __name__ == "__main__":

    data_dir = "../DATA/nc/"
    model_dir = "results/iresnet/best/"
    save_dir = "results/iresnet/best/"

    save_dir_labels = os.path.join(save_dir, "predicted-label-masks")
    make_directory(save_dir_labels)

    save_dir_z = os.path.join(save_dir, "predicted-z", "z")
    save_dir_loc = os.path.join(save_dir, "predicted-z", "locations")
    
    make_directory(save_dir_z)
    make_directory(save_dir_loc)
    
    m = np.load(os.path.join(model_dir, "../mean.npy"))
    s = np.load(os.path.join(model_dir, "../std.npy"))

    # dataset loader
    tile_extr = TileExtractor()
    normalizer = Normalizer(m, s)
    dataset = CumuloDataset(root_dir="../DATA/nc/", ext="nc", label_preproc=None, normalizer=normalizer, tiler=tile_extr)

    use_cuda = torch.cuda.is_available()
    print("using GPUs?", use_cuda)

    # load model
    model = load_iresnet(model_dir, use_cuda)
    
    for swath in dataset:
        
        filename, tiles, locations, _, rois, _ = swath

        base = os.path.basename(filename)
        base_npy = base.replace(".nc", ".npy")

        print("processing ", filename)
    
        labels, z = predict_tiles(model, tiles, use_cuda)

        save_path_labels = os.path.join(save_dir_labels, base_npy)
        save_path_z = os.path.join(save_dir_z, base_npy)
        save_path_loc = os.path.join(save_dir_loc, base_npy)
        
        save_labels(labels, locations, rois.squeeze(), save_path_labels)
        np.save(save_path_z, z)
        np.save(save_path_loc, locations.astype(np.uint16))

        print("{} processed".format(base))
