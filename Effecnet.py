
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:51:21 2019

@author: ai
"""
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from efficientnet_pytorch import EfficientNet


params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
class Effect_netI(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, pretrained=True, model_name='efficientnet-b0', device = "cuda"):
        super().__init__()
        assert num_channels == 3
        self.model_name= model_name
        self.device = device
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Efficient = EfficientNet.from_pretrained(self.model_name).to(device)

        

        self.logit = nn.Linear(1000, num_classes)
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.Efficient(inputs)

        x = self.logit(x)
        return x
"""
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Efficient = EfficientNet.from_pretrained('efficientnet-b3').to(device)
feature   = Efficient.extract_features
model = Effect_netI(num_classes=196, num_channels=3, pretrained=True, model_name='efficientnet-b3', device = device).to(device)
summary(model,(3, 300, 300))

"""


