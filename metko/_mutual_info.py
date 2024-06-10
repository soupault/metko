from PIL import Image
import torch
from torch import nn
from torchvision import transforms


# Modified from https://github.com/connorlee77/pytorch-mutual-information
class MutualInformation(nn.Module):
    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(),
                                 requires_grad=False)

    def marginal_pdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def joint_pdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values,
                                  dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

    def get_mutual_info(self, input1, input2):
        """

        Args:
            input1: (b, ch, d0, d1) tensor of [0.0, 1.0]
            input2: (b, ch, d0, d1) tensor of [0.0, 1.0]

        Returns:
            out: scalar
        """
        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert (input1.shape == input2.shape)

        x1 = input1.view(B, H * W, C)
        x2 = input2.view(B, H * W, C)

        pdf_x1, kernel_values1 = self.marginal_pdf(x1)
        pdf_x2, kernel_values2 = self.marginal_pdf(x2)
        pdf_x1x2 = self.joint_pdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def forward(self, input1, input2):
        """

        Args:
            input1: (b, ch, d0, d1)
            input2: (b, ch, d0, d1)

        Returns:
            out: scalar
        """
        return self.get_mutual_info(input1, input2)


def mutual_info(image_true, image_pred, *, sigma=0.1, num_bins=256, normalize=True):
    """

    Args:
        image_true: (d0, d1) ndarray or tensor
        image_pred: (d0, d1) ndarray or tensor
        sigma: float
        num_bins: int
        normalize: bool

    Returns:
        out: float
    """
    fn = MutualInformation(sigma=sigma, num_bins=num_bins, normalize=normalize)

    image_true = torch.as_tensor(image_true, dtype=torch.float32)[None, None, ...] / 256.
    image_pred = torch.as_tensor(image_pred, dtype=torch.float32)[None, None, ...] / 256.
    return fn(image_true, image_pred).cpu().numpy()


# if __name__ == '__main__':
#     device = 'cuda:0'
#
#     # Create test cases
#     img1 = Image.open('../../p41_hypereg/src/hypereg/metrics/grad.jpg').convert('L')
#     img2 = img1.rotate(10)
#
#     img1 = transforms.ToTensor()(img1).unsqueeze(dim=0).to(device)
#     img2 = transforms.ToTensor()(img2).unsqueeze(dim=0).to(device)
#
#     MI = MutualInformation(num_bins=256, sigma=0.1, normalize=True).to(device)
#     mi_test = MI(img1, img2)
#
#     print(mi_test.cpu().numpy())
