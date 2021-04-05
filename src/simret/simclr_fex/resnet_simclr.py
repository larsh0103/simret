import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def get_resnet_backbone(backbone_name: str, norm_layer=None):
    return models.resnet.__dict__[backbone_name](pretrained=False, norm_layer=norm_layer)


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = get_resnet_backbone(backbone_name=model_name)
            print("Feature extractor:", model_name)
            return model
        except:  # noqa: E722
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
