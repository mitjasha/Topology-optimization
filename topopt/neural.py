"""Small CPU U-Net baseline, selected by validation FEM compliance."""
import copy
from time import perf_counter
import numpy as np
import torch
from torch import nn
from .solver import Model


class SmallUNet(nn.Module):
    def __init__(self,channels=5,width=8):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        self.e1=block(channels,width);self.e2=block(width,width*2);self.bottom=block(width*2,width*4)
        self.pool=nn.MaxPool2d(2);self.up2=nn.ConvTranspose2d(width*4,width*2,2,stride=2)
        self.d2=block(width*4,width*2);self.up1=nn.ConvTranspose2d(width*2,width,2,stride=2)
        self.d1=block(width*2,width);self.out=nn.Conv2d(width,1,1)

    def forward(self,x):
        a=self.e1(x);b=self.e2(self.pool(a));z=self.bottom(self.pool(b))
        z=self.d2(torch.cat((self.up2(z),b),dim=1))
        return torch.sigmoid(self.out(self.d1(torch.cat((self.up1(z),a),dim=1))))


class UNetSurrogate:
    def fit(self,inputs,targets,valid_inputs,valid_cases,valid_references,epochs=80,seed=3017):
        start=perf_counter()
        torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        self.model=SmallUNet();optimizer=torch.optim.Adam(self.model.parameters(),lr=0.003)
        x=torch.tensor(inputs,dtype=torch.float32)
        y=torch.tensor(targets.reshape(-1,1,*inputs.shape[-2:]),dtype=torch.float32)
        best=float('inf');checkpoint=None;self.history=[]
        for epoch in range(1,epochs+1):
            self.model.train();optimizer.zero_grad()
            loss=nn.functional.mse_loss(self.model(x),y);loss.backward();optimizer.step()
            if epoch%10==0 or epoch==epochs:
                predictions=self.predict(valid_inputs)
                ratios=[]
                for case,reference,prediction in zip(valid_cases,valid_references,predictions):
                    m=Model(case,backend='sparse')
                    ratios.append(m.analyze(m.project_volume(prediction))['compliance']/reference['compliance'])
                score=float(np.mean(ratios))
                self.history.append({'epoch':epoch,'training_mse':float(loss.detach()),'validation_compliance_ratio':score})
                if score<best:
                    best=score;checkpoint=copy.deepcopy(self.model.state_dict());self.best_epoch=epoch
        self.model.load_state_dict(checkpoint);self.model.eval()
        self.seconds=perf_counter()-start
        return self

    def predict(self,inputs):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(inputs,dtype=torch.float32)).numpy().reshape(len(inputs),-1)
