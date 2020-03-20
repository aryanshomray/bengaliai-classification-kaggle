import torch.nn.functional as F

def cross_entropy(output,target):
    return 2*F.cross_entropy(output[0], target[0])+F.cross_entropy(output[1], target[1])+F.cross_entropy(output[2], target[2])