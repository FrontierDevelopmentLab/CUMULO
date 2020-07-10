import numpy as np

def get_class_mask(labels, locations, cloud_mask):

    shape = cloud_mask.shape
    assert len(shape) == 2, "shape should be 2d"

    overlay = np.full(shape, np.nan)
    
    for label, loc in zip(labels, locations):
        (x1, x2), (y1, y2) = loc
        overlay[x1:x2, y1:y2] = label

    overlay[cloud_mask == 0] = np.nan
    
    return overlay

def save_labels(labels, locations, cloud_mask, save_file):

    """ save tile predictions as a 2d array """
    class_mask = get_class_mask(labels, locations, cloud_mask)

    np.save(save_file, class_mask)
