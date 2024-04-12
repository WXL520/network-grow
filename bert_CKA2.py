from pathlib import Path
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
import timm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self._fpaths = [str(p.resolve()) for p in Path(img_dir).glob('*.JPEG')]
        self.transform = transform

    def __len__(self):
        return len(self._fpaths)

    def __getitem__(self, idx):
        img_path = self._fpaths[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = repeat(image, 'c h w-> (repeat c) h w', repeat=3)
        if self.transform:
            image = self.transform(image)
        image = image.to(DEVICE)
        return image


# using the validation transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # 用的是什么数据集？imagenet？还是sample的子集？
trans = transforms.Compose([
    transforms.ToPILImage(),  # this makes it work
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
data_path = 'imagenet-sample-images/'
dataset = CustomImageDataset(data_path, transform=trans)
print(len(dataset))  # 1000张图片
data_loader = torch.utils.data.DataLoader(dataset,
                                          # batch_size=4,
                                          batch_size=8,
                                          shuffle=False,
                                          pin_memory=False)  # I think I need this for debugging since I'm modifying the the inputs to CKA
                                          # pin_memory=True) # can't pin on GPU. Guess I shouldn't be moving the images to GPU in the dataloader?
# 在这里加载文本数据集

t_model_names = [
    'vit_base_patch32_224',
    'vit_base_patch16_224',
    'vit_large_patch16_224',
]
t_model_names_in21k = [m + '_in21k' for m in t_model_names]
t_model_names_in21k.append('vit_huge_patch14_224_in21k')  # 比较base和huge吧？
print(t_model_names_in21k)
model_t = timm.create_model(t_model_names[0], pretrained=True).to(DEVICE)
model_t2 = timm.create_model(t_model_names[2], pretrained=True).to(DEVICE)
model = model_t


# mini-batch CKA?
def gram(X):
    # ensure correct input shape
    X = rearrange(X, 'b ... -> b (...)')
    return X @ X.T


def centering_mat(n):
    v_i = torch.ones(n, 1, device=DEVICE)
    H = torch.eye(n, device=DEVICE) - (v_i @ v_i.T) / n
    return H


def centered_gram(X):
    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m)
    # logger.info(H.shape)
    # logger.info(K.shape)
    return H @ K @ H


def unbiased_hsic_xy(X, Y):
    n = X.shape[0]
    assert n > 3
    v_i = torch.ones(n, 1, device=DEVICE)
    K = centered_gram(X)
    L = centered_gram(Y)
    KL = K @ L
    iK = v_i.T @ K
    Li = L @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n - 1) * (n - 2))
    c = iK @ Li * 2 / (n - 2)

    outv = (a + b - c) / (n * (n - 3))
    return outv.long().item()


class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Introduced in: https://arxiv.org/pdf/2010.15327.pdf
        Implemented to reproduce the results in: https://arxiv.org/pdf/2108.08810v1.pdf
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, X: torch.Tensor, Y: torch.Tensor):
        # NB: torchmetrics Bootstrap resampling janks up batch shape by varying number of samples per batch
        self._xx += unbiased_hsic_xy(X, X)
        self._xy += unbiased_hsic_xy(X, Y)
        self._yy += unbiased_hsic_xy(Y, Y)

    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))


#########################
# sanity tests


# x = torch.randn(8, 7, device=DEVICE)
# y = torch.randn(8, 17, device=DEVICE)
#
# g = gram(x)
# assert g.shape == (x.shape[0], x.shape[0])
# cg = centered_gram(x)  # centered = column and row means subtracted
# assert cg.shape == g.shape
#
# cka = MinibatchCKA().to(DEVICE)
# cka.update(x, y)
# print(cka.compute())


#################################################

#################################################
class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target

        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache

    def clear(self):
        self._cache = None

    def _extract_target(self):
        for name, module in self.model.named_modules():
            if name == self.target:
                self._target = module
                return

    def _register_hook(self):
        def _hook(module, in_val, out_val):
            self._cache = out_val

        self._target.register_forward_hook(_hook)


#######################################
# # sanity tests
# target0 = 'blocks.0.attn.qkv'
# target1 = 'blocks.0.mlp.fc1'
#
# hook0 = HookedCache(model_t, target0)
# hook1 = HookedCache(model_t, target1)  # 啥意思啊？


######################################
#
######################################
def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
        for j, cka in enumerate(ckas):
            z = cka.compute().item()
            vals.append((i, j, z))

    sim_mat = torch.zeros(i + 1, j + 1)
    for i, j, z in vals:
        sim_mat[i, j] = z

    return sim_mat


def make_pairwise_metrics(mod1_hooks, mod2_hooks):
    metrics = []
    for i_ in mod1_hooks:
        metrics.append([])
        for j_ in mod2_hooks:
            metrics[-1].append(MinibatchCKA().to(DEVICE))
    return metrics


def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, do_log):
    for i, hook1 in enumerate(mod1_hooks):
        for j, hook2 in enumerate(mod2_hooks):
            cka = metrics[i][j]
            X, Y = hook1.value, hook2.value
            cka.update(X, Y)
            if do_log and 0 in (i, j):
                _metric_name = f"{metric_name}_{i}-{j}"
                v = cka.compute()
                # writer.add_scalar(_metric_name, v, it)
    if do_log:
        sim_mat = get_simmat_from_metrics(metrics)
        sim_mat = sim_mat.unsqueeze(0) * 255
        # writer.add_image(metric_name, sim_mat, it)


log_every = 10

batch_size = 8
dataset = CustomImageDataset(data_path, transform=trans)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=False
                                          )

# modc_hooks = []
# for i, stage in enumerate(model_t.stages):
#     for j, block in enumerate(stage.blocks):
#         tgt = f'stages.{i}.blocks.{j}'
#         hook = HookedCache(model_t, tgt)
#         modc_hooks.append(hook)

modt_hooks = []
for j, block in enumerate(model_t.blocks):
    tgt = f'blocks.{j}'
    hook = HookedCache(model_t, tgt)  # 把hook.value打印出来看看？
    modt_hooks.append(hook)
modt2_hooks = []
for j, block in enumerate(model_t2.blocks):
    tgt = f'blocks.{j}'
    hook = HookedCache(model_t2, tgt)
    modt2_hooks.append(hook)

# metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)
# metrics_cc = make_pairwise_metrics(modc_hooks, modc_hooks)
# metrics_tt = make_pairwise_metrics(modt_hooks, modt_hooks)
metrics_tt2 = make_pairwise_metrics(modt_hooks, modt2_hooks)

with torch.no_grad():
    for it, batch in enumerate(data_loader):
        batch = batch.to(DEVICE)
        do_log = (it % log_every == 0)
        if do_log:
            logger.debug(f"iter: {it}")
        outv_t = model_t(batch)
        outv_t2 = model_t2(batch)

        # update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", do_log)
        # update_metrics(modc_hooks, modc_hooks, metrics_cc, "cka/cc", do_log)
        # update_metrics(modt_hooks, modt_hooks, metrics_tt, "cka/tt", do_log)
        update_metrics(modt_hooks, modt2_hooks, metrics_tt2, "cka/tt2", do_log)

        for hook0 in modt_hooks:
            for hook1 in modt2_hooks:
                hook0.clear()
                hook1.clear()

sim_mat = get_simmat_from_metrics(metrics_tt2)
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns

#
# sim_mat = torch.FloatTensor([[1.0000, 0.9841, 0.9611, 0.9624, 0.9643, 0.9629, 0.9446, 0.8845, 0.8626, 0.8696, 0.8877, 0.8930],
#         [0.9841, 1.0000, 0.9928, 0.9914, 0.9877, 0.9804, 0.9550, 0.8882, 0.8680, 0.8749, 0.8918, 0.8976],
#         [0.9611, 0.9928, 1.0000, 0.9983, 0.9926, 0.9833, 0.9579, 0.8938, 0.8749, 0.8818, 0.8982, 0.9041],
#         [0.9624, 0.9914, 0.9983, 1.0000, 0.9973, 0.9906, 0.9694, 0.9088, 0.8874, 0.8945, 0.9118, 0.9177],
#         [0.9643, 0.9877, 0.9926, 0.9973, 1.0000, 0.9972, 0.9818, 0.9247, 0.8989, 0.9063, 0.9250, 0.9310],
#         [0.9629, 0.9804, 0.9833, 0.9906, 0.9972, 1.0000, 0.9918, 0.9412, 0.9117,  0.9191, 0.9384, 0.9444],
#         [0.9446, 0.9550, 0.9579, 0.9694, 0.9818, 0.9918, 1.0000, 0.9705, 0.9341, 0.9403, 0.9582, 0.9637],
#         [0.8845, 0.8882, 0.8938, 0.9088, 0.9247, 0.9412, 0.9705, 1.0000, 0.9732,  0.9739, 0.9776, 0.9800],
#         [0.8626, 0.8680, 0.8749, 0.8874, 0.8989, 0.9117, 0.9341, 0.9732, 1.0000,  0.9992, 0.9885, 0.9873],
#         [0.8696, 0.8749, 0.8818, 0.8945, 0.9063, 0.9191, 0.9403, 0.9739, 0.9992, 1.0000, 0.9918, 0.9907],
#         [0.8877, 0.8918, 0.8982, 0.9118, 0.9250, 0.9384, 0.9582, 0.9776, 0.9885, 0.9918, 1.0000, 0.9989],
#         [0.8930, 0.8976, 0.9041, 0.9177, 0.9310, 0.9444, 0.9637, 0.9800, 0.9873, 0.9907, 0.9989, 1.0000]])
# sim_mat = torch.FloatTensor([[0.9841, 0.9782, 0.9727, 0.9665, 0.9630, 0.9593, 0.9572, 0.9544, 0.9509,
#                               0.9472, 0.9418, 0.9237, 0.8991, 0.8840, 0.8755, 0.8746, 0.8765, 0.8750,
#                               0.8757, 0.8775, 0.8720, 0.8691, 0.8724, 0.8781],
#                              [0.9854, 0.9841, 0.9812, 0.9780, 0.9740, 0.9693, 0.9663, 0.9613, 0.9567,
#                               0.9520, 0.9469, 0.9283, 0.9041, 0.8897, 0.8818, 0.8812, 0.8833, 0.8820,
#                               0.8827, 0.8852, 0.8849, 0.8834, 0.8864, 0.8918],
#                              [0.9775, 0.9805, 0.9806, 0.9800, 0.9771, 0.9731, 0.9697, 0.9640, 0.9593,
#                               0.9547, 0.9502, 0.9331, 0.9108, 0.8974, 0.8901, 0.8897, 0.8919, 0.8907,
#                               0.8914, 0.8940, 0.8953, 0.8941, 0.8969, 0.9019],
#                              [0.9797, 0.9841, 0.9853, 0.9855, 0.9836, 0.9809, 0.9785, 0.9739, 0.9699,
#                               0.9660, 0.9622, 0.9472, 0.9266, 0.9139, 0.9069, 0.9066, 0.9086, 0.9074,
#                               0.9081, 0.9105, 0.9101, 0.9079, 0.9107, 0.9158],
#                              [0.9802, 0.9860, 0.9885, 0.9894, 0.9888, 0.9874, 0.9864, 0.9832, 0.9803,
#                               0.9773, 0.9743, 0.9617, 0.9429, 0.9310, 0.9242, 0.9238, 0.9257, 0.9245,
#                               0.9251, 0.9272, 0.9245, 0.9213, 0.9241, 0.9292],
#                              [0.9794, 0.9863, 0.9898, 0.9915, 0.9919, 0.9916, 0.9918, 0.9903, 0.9886,
#                               0.9867, 0.9846, 0.9745, 0.9575, 0.9463, 0.9398, 0.9393, 0.9410, 0.9399,
#                               0.9404, 0.9423, 0.9378, 0.9336, 0.9364, 0.9415],
#                              [0.9650, 0.9746, 0.9802, 0.9835, 0.9858, 0.9876, 0.9893, 0.9904, 0.9905,
#                               0.9904, 0.9902, 0.9856, 0.9734, 0.9643, 0.9588, 0.9583, 0.9596, 0.9587,
#                               0.9590, 0.9605, 0.9525, 0.9465, 0.9492, 0.9540],
#                              [0.9121, 0.9245, 0.9322, 0.9369, 0.9411, 0.9452, 0.9482, 0.9513, 0.9534,
#                               0.9552, 0.9569, 0.9582, 0.9518, 0.9456, 0.9415, 0.9412, 0.9423, 0.9416,
#                               0.9419, 0.9430, 0.9327, 0.9251, 0.9275, 0.9319],
#                              [0.8898, 0.9012, 0.9082, 0.9122, 0.9159, 0.9196, 0.9223, 0.9247, 0.9265,
#                               0.9278, 0.9289, 0.9276, 0.9193, 0.9122, 0.9077, 0.9074, 0.9086, 0.9079,
#                               0.9082, 0.9092, 0.8986, 0.8918, 0.8944, 0.8992],
#                              [0.8968, 0.9085, 0.9155, 0.9196, 0.9234, 0.9271, 0.9297, 0.9322, 0.9341,
#                               0.9353, 0.9364, 0.9352, 0.9269, 0.9197, 0.9151, 0.9148, 0.9161, 0.9154,
#                               0.9158, 0.9168, 0.9064, 0.8995, 0.9022, 0.9070],
#                              [0.9143, 0.9268, 0.9344, 0.9389, 0.9429, 0.9468, 0.9496, 0.9525, 0.9545,
#                               0.9559, 0.9571, 0.9568, 0.9488, 0.9418, 0.9372, 0.9369, 0.9382, 0.9375,
#                               0.9384, 0.9401, 0.9304, 0.9236, 0.9262, 0.9309],
#                              [0.9202, 0.9327, 0.9404, 0.9449, 0.9489, 0.9527, 0.9556, 0.9585, 0.9604,
#                               0.9619, 0.9632, 0.9627, 0.9546, 0.9475, 0.9429, 0.9425, 0.9439, 0.9432,
#                               0.9440, 0.9456, 0.9354, 0.9285, 0.9312, 0.9361]])
print('ok')
print(sim_mat.shape)
print(sim_mat)

# # creating a colormap
# colormap = sns.color_palette("Greens")
# ax = sns.heatmap(sim_mat, cmap=colormap)
# ax.invert_yaxis()
plt.imshow(sim_mat, origin='lower')
plt.title('t-t')
plt.show()  # small ViT的前6层，和large ViT的前10层表示很接近，说明说明？
#
# sim_mat = get_simmat_from_metrics(metrics_cc)
# plt.imshow(sim_mat)
# plt.title('c-c')
# plt.show()
#
# sim_mat = get_simmat_from_metrics(metrics_ct)
# plt.imshow(sim_mat)
# plt.title('c-t')
# plt.show()
