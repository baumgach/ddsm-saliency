import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet

class CAM(nn.Module):
    def __init__(self, net):
        super(CAM, self).__init__()
        if isinstance(net, ResNet):
            self.net_identifier = 'ResNet'
        else:
            raise ValueError(f'Grad-CAM not implemented for net class {net.__class__.__name__}')

        self.classifier = net

    def forward(self, im, label, back=0, grad_outputs=None):
        logits, activations = self.forward_through_net(im)
        output = activations[back]
        if grad_outputs is None:
            grad_outputs = torch.eye(logits.shape[1], device=label.device).index_select(dim=0, index=label.view(-1))

        grad = torch.autograd.grad(logits, output, grad_outputs=grad_outputs)[0]
        saliency = (output * grad.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True))
        cam = F.relu(saliency).sum(dim=1, keepdim=True)
        cam = F.upsample_bilinear(cam, size=im.shape[-2:]).detach()
        logits.detach()
        self.classifier.zero_grad()
        return cam

    def forward_through_net(self, im):
        if self.net_identifier == 'ResNet':
            x = self.classifier.conv1(im)
            x = self.classifier.bn1(x)
            x = self.classifier.relu(x)
            x = self.classifier.maxpool(x)

            l1 = self.classifier.layer1(x)
            l2 = self.classifier.layer2(l1)
            l3 = self.classifier.layer3(l2)
            l4 = self.classifier.layer4(l3)

            x2 = self.classifier.avgpool(l4)
            x2 = torch.flatten(x2, 1)
            logits = self.classifier.fc(x2)

            return logits, [l4, l3, l2, l1]
        else:
            raise ValueError(f'Function not implemented for net class {self.net_identifier}')