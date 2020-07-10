import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import visdom

from src.iresnet import multiscale_conv_iResNet
from src.iresnet_utils import get_init_batch, train, test, save_model
from src.loader import CumuloDataset
from src.metrics import scores_per_class
from src.utils import Normalizer, get_dataset_statistics, get_hms, make_directory, get_tile_sampler, tile_collate

# monitor training
try:
    port = int(sys.argv[1]) # chosen port for visdom monitoring
except:
    raise Exception("Run first $visdom --port <port>")

viz = visdom.Visdom(port=port, server="http://localhost")
assert viz.check_connection(), "Could not connect to visdom"

nb_epochs = 100
t_size = 3
nb_classes = 8
batch_size = 256 # number of tiles per batch 
lr = 0.001
weight_decay = 5e-4

root_dir = os.path.join("../DATA/npz/")

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True

# iResNet parameters

# specify for each res block
nb_blocks = [4, 4, 4, 4, 4]
nb_strides = [1, 2, 2, 2, 2]
nb_channels = [4, 16, 32, 32, 32]

inj_pad = 0
coeff = 0.97

nb_trace_samples = 1
nb_series_terms = 1
nb_iter_norm = 5

# shape of the input (channels, height, width)
in_shape = (13, t_size, t_size)

use_cuda = torch.cuda.is_available()
print("using GPUs?", use_cuda)

classification_weight = in_shape[0] * in_shape[1] * in_shape[2]

save_dir = "results/iresnet"

save_dir_best = os.path.join(save_dir, "best")
save_dir_last = os.path.join(save_dir, "last")

make_directory(save_dir_best)
make_directory(save_dir_last)

train_log = open(os.path.join(save_dir, "train_log.txt"), 'w')
val_log = open(os.path.join(save_dir, "val_log.txt"), 'w')
test_log = open(os.path.join(save_dir, "test_log.txt"), 'w')

# compute class weights and normalizer
try:
    class_weights = np.load(os.path.join(save_dir, "class-weights.npy"))
    m = np.load(os.path.join(save_dir, "mean.npy"))
    s = np.load(os.path.join(save_dir, "std.npy"))

except:
    # load dataset characteristics
    print("Computing dataset mean, standard deviation and class ratios")

    dataset = CumuloDataset(os.path.join(root_dir, "label/")) # change to unlabelled dir
    class_weights, m, s = get_dataset_statistics(dataset, nb_classes, collate=tile_collate, batch_size=40, use_cuda=use_cuda)

    np.save(os.path.join(save_dir, "class-weights.npy"), class_weights)
    np.save(os.path.join(save_dir, "mean.npy"), m)
    np.save(os.path.join(save_dir, "std.npy"), s)

normalizer = Normalizer(m, s)
class_weights = torch.from_numpy(class_weights).float()

# get train, validation, test sets by randomly splitting the set of swaths
nb_swaths = len(os.listdir(os.path.join(root_dir, "label/")))
idx = np.arange(nb_swaths)
np.random.shuffle(idx)
train_idx, val_idx, test_idx = np.split(idx, [int(.7 * nb_swaths), int(.8 * nb_swaths)])

train_dataset = CumuloDataset(os.path.join(root_dir, "label/"), normalizer=normalizer, indices=train_idx) # change to labelled dir
val_dataset = CumuloDataset(os.path.join(root_dir, "label/"), "npz", normalizer=normalizer, indices=val_idx) # change to labelled dir
test_dataset = CumuloDataset(os.path.join(root_dir, "label/"), "npz", normalizer=normalizer, indices=test_idx) # change to labelled dir

# samplers
train_sampler = get_tile_sampler(train_dataset)

# data loaders, batch_size = number of tiles
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

# in the following, batch_size corresponds to number of swaths (each swath contain multiple tiles)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=40, collate_fn=tile_collate, shuffle=False, num_workers=30)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=40, collate_fn=tile_collate, shuffle=False, num_workers=30)

# create new one and initialize it
model = multiscale_conv_iResNet(in_shape, nb_blocks, nb_strides, nb_channels, False, inj_pad, coeff, nb_classes, nb_trace_samples, nb_series_terms, nb_iter_norm, actnorm=True, learn_prior=True, nonlin="elu", lin_classifier=True)
# init actnrom parameters
init_batch = get_init_batch(trainloader, 1)
print("initializing actnorm parameters...")
with torch.no_grad():
    model(init_batch, ignore_logdet=True)
print("initialized")

# uses all available GPUs
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    cudnn.benchmark = True
    in_shapes = model.module.get_in_shapes()
else:
    in_shapes = model.get_in_shapes()

print('|  Train Epochs: ' + str(nb_epochs))
print('|  Initial Learning Rate: ' + str(lr))

elapsed_time = 0
test_objective = -np.inf

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

state = {
    'tile-size': t_size,
    'batch-size': batch_size,
    'classifier': 'linear',
    'loss-function': 'cross-entropy',
    'learning-rate': lr,
    'class-weight': classification_weight,
    'Lipschitz-coeff': coeff
}

best_accuracy, best_f1 = 0., 0.

try:

    for epoch in range(1, 1 + nb_epochs):
        start_time = time.time()
        train_cm, train_acc = train(model, optimizer, epoch, lr, trainloader, viz, train_log, class_weights, use_cuda, classification_weight)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
        print('Training accuracy', train_acc)
 
        val_cm, val_acc = test(model, epoch, valloader, viz, val_log, use_cuda, "val ", classification_weight)
        print('Validation accuracy', val_acc)
        _, val_f1 = scores_per_class(val_cm)
        val_f1 = val_f1.mean()
        
        # save state if accuracy or f1 improved
        if val_acc > best_accuracy or val_f1 > best_f1:
            save_model(model, optimizer, train_cm, val_cm, state, save_dir_best, epoch=epoch, train_accuracy=train_acc, val_accuracy=val_acc)
            best_accuracy = val_acc
            best_f1 = val_f1

except KeyboardInterrupt:
    pass
    print("\nWait for the program to save current state")
    print("stopping at epoch", epoch)

except Exception as e:
    raise e

# save last-model
test_cm, _ = test(model, epoch, testloader, viz, test_log, use_cuda, "test ", classification_weight)
save_model(model, optimizer, train_cm, val_cm, state, save_dir_last, epoch=epoch, train_accuracy=train_acc, val_accuracy=val_acc)
np.save(os.path.join(save_dir_last, "test-confusion-matrix.npy"), test_cm)

# test with best-model and save
model = torch.load(os.path.join(save_dir_best, "model.t7"))["model"]

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    cudnn.benchmark = True

test_cm, _ = test(model, epoch, testloader, viz, test_log, use_cuda, "test ", classification_weight)
np.save(os.path.join(save_dir_best, "test-confusion-matrix.npy"), test_cm)
