import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import pretrainedmodels


class main_model(torch.nn.Module):

    def __init__(self):
        super(main_model, self).__init__()
        se_resnet50 = pretrainedmodels.se_resnet50(pretrained=None)
        self.conv1 = torch.nn.Conv2d(1, 64, (3, 3))
        self.layer0 = se_resnet50.layer0[1:]
        self.layer1 = se_resnet50.layer1
        self.layer2 = se_resnet50.layer2
        self.layer3 = se_resnet50.layer3
        self.layer4 = se_resnet50.layer4
        self.layer5 = torch.nn.Conv2d(2048, 256, (3, 3))
        self.fc1 = torch.nn.Linear(4096, 168)
        self.fc2 = torch.nn.Linear(4096, 11)
        self.fc3 = torch.nn.Linear(4096, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return [x1, x2, x3]
