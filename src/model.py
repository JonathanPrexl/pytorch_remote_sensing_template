import torch
from torchvision.models import vgg11
from torch.nn import Conv2d


class VGG(torch.nn.Module):
    def __init__(self, num_bands, num_classes):
        super().__init__()
        self.model = vgg11(num_classes=num_classes)
        self.model.features[0] = Conv2d(num_bands, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        return self.model(x)
    
if __name__ == "__main__":
    import hydra
    import omegaconf
    from torchinfo import summary

    cfg = omegaconf.OmegaConf.load("./configs/default.yaml")

    model = hydra.utils.instantiate(cfg.model)
    example_batch = torch.zeros(1, 13, 64, 64)
    example_output_shape = model(example_batch).shape
    print(f"With input shape {example_batch} the model outputs a shape {example_output_shape}.")
    summary(model)
