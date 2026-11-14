import torch
from torch import nn
from . import Stage_2_1


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [move_to_device(v, device) for v in value]
    return value

class GatedAttentionPooling(nn.Module):
    def __init__(self, M=384, L=128):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(M,L), nn.Linear(M,L), nn.Linear(L,1)

    def forward(self, H):
        a = self.w(torch.tanh(self.V(H))*torch.sigmoid(self.U(H))).softmax(dim=1)
        return (a*H).sum(1), a.squeeze(-1)

class LocalEncoder(nn.Module):
    def __init__(self, channels=3, dim=384):
        super().__init__()
        layers=[]
        for width in (32,64,128):
            layers.extend([nn.Conv3d(channels,width,3,stride=2,padding=1,bias=False),
                           nn.GroupNorm(8,width),nn.GELU()])
            channels=width
        self.layers=nn.Sequential(*layers,nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(128,dim))

    def forward(self,x):
        return self.layers(x)

class SwinMIL(nn.Module):
    def __init__(self, in_chans=3, num_classes=3, L=128):
        super().__init__()
        self.encoder=Swin_transforemr.Model(in_chans=in_chans,num_classes=num_classes)
        self.context_encoder=LocalEncoder(in_chans)
        self.texture_encoder=LocalEncoder(in_chans)
        self.view_embedding=nn.Parameter(torch.zeros(1,4,384))
        nn.init.normal_(self.view_embedding,std=0.02)
        self.view_norm=nn.LayerNorm(384)
        self.pool=GatedAttentionPooling(384,L)
        self.drop=nn.Dropout(0.3)
        self.classifier=nn.Linear(384,num_classes)

    def forward(self,x):
        views=x['views'] if isinstance(x,dict) else x
        if not isinstance(views,(list,tuple)) or len(views)!=4:
            raise ValueError('Expected four heterogeneous views, not stacked legacy patches')
        if tuple(views[0].shape[-3:])!=(128,128,128) or tuple(views[3].shape[-3:])!=(32,32,32):
            raise ValueError('Global and texture views must retain 128^3 and 32^3 shapes')
        features=[self.encoder.forward_features(views[0]),self.context_encoder(views[1]),
                  self.context_encoder(views[2]),self.texture_encoder(views[3])]
        H=self.view_norm(torch.stack(features,1))+self.view_embedding
        fused,attention=self.pool(H)
        logits=self.classifier(self.drop(fused))
        return logits,attention

def Model(in_chans=3,num_classes=3,L=128):
    return SwinMIL(in_chans,num_classes,L)