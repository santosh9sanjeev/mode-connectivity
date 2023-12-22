import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

import data
# import models
import curves
import utils
import random
import torchxrayvision as xrv

import torch.nn as nn
class CheXpertModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CheXpertModel, self).__init__()
        
        self.feature_extractor = base_model.features
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024,num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        # print(x.shape)
        # Global average pooling (GAP)
        x = self.global_avg_pooling(x)  # Assuming the output shape is (batch_size, 512, H, W)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # Forward pass through the linear classification layer
        x = self.classifier(x)
        return x

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

seed = 0

random.seed(seed)
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Computes values for plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                    help='training directory (default: /tmp/plane)')

parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                    help='number of points in the grid (default: 21)')
parser.add_argument('--margin_left', type=float, default=0.8, metavar='M',
                    help='left margin (default: 0.2)')
parser.add_argument('--margin_right', type=float, default=0.8, metavar='M',
                    help='right margin (default: 0.2)')
parser.add_argument('--margin_bottom', type=float, default=0.8, metavar='M',
                    help='bottom margin (default: 0.)')
parser.add_argument('--margin_top', type=float, default=0.8, metavar='M',
                    help='top margin (default: 0.2)')

parser.add_argument('--curve_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False,
    dataset_selection = 'rsna'
)

# architecture = getattr(models, args.model)
# curve = getattr(curves, args.curve)

# curve_model = curves.CurveNet(
#     num_classes,
#     curve,
#     architecture.curve,
#     args.num_bends,
#     architecture_kwargs=architecture.kwargs,
# )
# curve_model.cuda()

# checkpoint = torch.load(args.ckpt)
# curve_model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(args.wd)


def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

# full_model = CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2) # CheXpert (Stanford)

model1 =  CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2)
# model1.fc = nn.Linear(model1.fc.in_features, 2)
model1 = model1.to('cuda')
# /home/santoshsanjeev/dnn-mode-connectivity/models_model_soups/rsna_cnn_v1/model_4.pth
# checkpoint1 = torch.load('/home/santoshsanjeev/model_soups/dnn-mode-connectivity/cnn_v1/model_2.pth')
# checkpoint1 = torch.load('/home/santoshsanjeev/dnn-mode-connectivity/models_model_soups/rsna_imagenet_pre_with_augs/LP.pth')
checkpoint1 = torch.load('/home/santoshsanjeev/dnn-mode-connectivity/models_model_soups/rsna_chexpert/model_6.pth')
model1.load_state_dict(checkpoint1['model_state_dict'])
test_res = utils.test(loaders['test'], model1, criterion, regularizer)
print('111111111',test_res['accuracy'])



model2 =  CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2)
# model2.fc = nn.Linear(model2.fc.in_features, 2)
model2 = model2.to('cuda')
# checkpoint2 = torch.load('/home/santoshsanjeev/model_soups/dnn-mode-connectivity/cnn_v1/model_4.pth')
checkpoint2 = torch.load('/home/santoshsanjeev/dnn-mode-connectivity/models_model_soups/rsna_chexpert/model_23.pth')
model2.load_state_dict(checkpoint2['model_state_dict'])
test_res = utils.test(loaders['test'], model2, criterion, regularizer)
print('222222222',test_res['accuracy'])



model3 =  CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2)
# model3.fc = nn.Linear(model3.fc.in_features, 2)
model3 = model3.to('cuda')
# checkpoint3 = torch.load('/home/santoshsanjeev/model_soups/dnn-mode-connectivity/cnn_v1/model_6.pth')
checkpoint3 = torch.load('/home/santoshsanjeev/dnn-mode-connectivity/models_model_soups/rsna_chexpert/model_2.pth')
model3.load_state_dict(checkpoint3['model_state_dict'])
test_res = utils.test(loaders['test'], model3, criterion, regularizer)
print('3333333',test_res['accuracy'])
# exit()

w = list()
#curve_parameters = []#list(curve_model.net.parameters())
curve_parameters = [model1.parameters(),model2.parameters(),model3.parameters()]#list(curve_model.net.parameters())

for i in range(args.num_bends):
    w.append(np.concatenate([
        p.data.cpu().numpy().ravel() for p in curve_parameters[i]#[i::args.num_bends]
    ]))

print('Weight space dimensionality: %d' % w[0].shape[0])

u = w[2] - w[0]
dx = np.linalg.norm(u)
u /= dx

v = w[1] - w[0]
v -= np.dot(u, v) * u
dy = np.linalg.norm(v)
v /= dy
print('dx',-0.8*dx,1.8*dx,'dy',-0.8*dx,1.8*dy)
bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
print(bend_coordinates)
ts = np.linspace(0.0, 1.0, args.curve_points)
# curve_coordinates = []
# for t in np.linspace(0.0, 1.0, args.curve_points):
#     weights = curve_model.weights(torch.Tensor([t]).cuda())
#     curve_coordinates.append(get_xy(weights, w[0], u, v))
# curve_coordinates = np.stack(curve_coordinates)

G = args.grid_points
alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

tr_loss = np.zeros((G, G))
tr_nll = np.zeros((G, G))
tr_acc = np.zeros((G, G))
tr_err = np.zeros((G, G))

te_loss = np.zeros((G, G))
te_nll = np.zeros((G, G))
te_acc = np.zeros((G, G))
te_err = np.zeros((G, G))

grid = np.zeros((G, G, 2))

base_model =  CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2)
# base_model.fc = nn.Linear(base_model.fc.in_features, 2)
base_model.to('cuda')
columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        print()
        p = w[0] + alpha * dx * u + beta * dy * v

        offset = 0
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = p[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        utils.update_bn(loaders['train'], base_model)

        tr_res = utils.test(loaders['train'], base_model, criterion, regularizer)
        te_res = utils.test(loaders['test'], base_model, criterion, regularizer)

        tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
        te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']

        c = get_xy(p, w[0], u, v)
        grid[i, j] = [alpha * dx, beta * dy]

        tr_loss[i, j] = tr_loss_v
        tr_nll[i, j] = tr_nll_v
        tr_acc[i, j] = tr_acc_v
        tr_err[i, j] = 100.0 - tr_acc[i, j]

        te_loss[i, j] = te_loss_v
        te_nll[i, j] = te_nll_v
        te_acc[i, j] = te_acc_v
        te_err[i, j] = 100.0 - te_acc[i, j]

        values = [
            grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
            te_nll[i, j], te_err[i, j]
        ]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)







np.savez(
    os.path.join(args.dir, 'plane.npz'),
    ts=ts,
    bend_coordinates=bend_coordinates,
    # curve_coordinates=curve_coordinates,
    alphas=alphas,
    betas=betas,
    grid=grid,
    tr_loss=tr_loss,
    tr_acc=tr_acc,
    tr_nll=tr_nll,
    tr_err=tr_err,
    te_loss=te_loss,
    te_acc=te_acc,
    te_nll=te_nll,
    te_err=te_err
)
