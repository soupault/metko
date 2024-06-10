# Perceptual loss paper: https://arxiv.org/pdf/1603.08155.pdf

from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


# From https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/fast_neural_style/neural_style/
class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


# From https://github.com/crowsonkb/vgg_loss
class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](weights=models.VGG16_Weights.DEFAULT).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


def perceptual_loss(image_true, image_pred, *, model="vgg16", layer=8):
    """

    Args:
        image_true: (M, N) ndarray
        image_pred: (M, N) ndarray
        model: {"vgg16", "vgg19"}
        layer: int

    Returns:
        loss: scalar
    """
    fn = VGGLoss(model=model, layer=layer)

    image_true = torch.as_tensor(image_true, dtype=torch.float32)[None, None, ...]
    image_true = image_true.repeat(1, 3, 1, 1)
    image_pred = torch.as_tensor(image_pred, dtype=torch.float32)[None, None, ...]
    image_pred = image_pred.repeat(1, 3, 1, 1)
    return fn(image_true, image_pred).cpu().numpy()


if __name__ == "__main__":
    from torchvision import io as tio
    from torchvision.transforms import functional as TF

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    crit_vgg = VGGLoss().to(device)

    target = tio.read_image('DSC00261.jpg')[None] / 255
    target = TF.resize(target, (256, 256), 3).to(device)
    target_act = crit_vgg.get_features(target)

    input = torch.rand_like(target) / 255 + 0.5
    input.requires_grad_(True)

    loss = crit_vgg(input, target_act, target_is_features=True)
    TF.to_pil_image(input[0].clamp(0, 1)).save('out.png')
