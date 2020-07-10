import lightgbm as lgb
import numpy as np

from src.prediction_utils import save_labels
from src.loader import CumuloDataset
from src.utils import make_directory, TileExtractor

def load_lgbm(filename):

    loaded_model = lgb.Booster(model_file=filename)

    return loaded_model

def predict_tiles(model, tiles):
    
    t_shape = tiles.shape
    vec_tiles = tiles[:, :13].reshape(-1, 13 * t_shape[2] * t_shape[3])

    labels = np.argmax(model.predict(vec_tiles, num_iteration=model.best_iteration), 1).astype(int)

    return labels

def predict_and_save(save_dir, model_path, swath):

    filename, tiles, locations, _, rois, _ = swath
    
    print("processing", filename)

    model = load_lgbm(model_path)

    labels = predict_tiles(model, tiles)

    save_path = os.path.join(save_dir, os.path.basename(filename)).replace(".nc", ".npy")

    save_labels(labels, locations, rois.squeeze(), save_path)

    print(save_path, "processed")

if __name__ == "__main__":

    import os

    model_path = "results/lgbm/lightgbm-model.txt"
    
    save_dir = os.path.join("results/lgbm/predicted-label-masks/")
    make_directory(save_dir)

    tile_extr = TileExtractor()
    dataset = CumuloDataset(root_dir="../DATA/nc/", ext="nc", label_preproc=None, tiler=tile_extr)
    
    for swath in dataset:
        predict_and_save(save_dir, model_path, swath)
