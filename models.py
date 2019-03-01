import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def pts(name, ts):
    print(name + ' shape :', np.shape(ts))

class Generator(nn.Module):

    def __init__(self, feature_size=3, class_num=1, element_num=128):
        super(Generator, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(feature_size, feature_size*2)
        self.fc1_bn = nn.BatchNorm1d(element_num)
        self.fc2 = nn.Linear(feature_size*2, feature_size*2*2)
        self.fc2_bn = nn.BatchNorm1d(element_num)
        self.fc3 = nn.Linear(feature_size*2*2, feature_size*2*2)

        # Relation module1
        self.unary1 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps1 = torch.FloatTensor(torch.rand(1))
        self.ph1 = torch.FloatTensor(torch.rand(1))
        self.wr1 = torch.FloatTensor(torch.rand(1))

        # Relation module2
        self.unary2 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps2 = torch.FloatTensor(torch.rand(1))
        self.ph2 = torch.FloatTensor(torch.rand(1))
        self.wr2 = torch.FloatTensor(torch.rand(1))

        # Relation module3
        self.unary3 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps3 = torch.FloatTensor(torch.rand(1))
        self.ph3 = torch.FloatTensor(torch.rand(1))
        self.wr3 = torch.FloatTensor(torch.rand(1))

        # Relation module4
        self.unary4 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps4 = torch.FloatTensor(torch.rand(1))
        self.ph4 = torch.FloatTensor(torch.rand(1))
        self.wr4 = torch.FloatTensor(torch.rand(1))
        
        # Decoder
        self.fc4 = nn.Linear(feature_size*2*2, feature_size*2)
        self.fc4_bn = nn.BatchNorm1d(element_num)
        self.fc5 = nn.Linear(feature_size*2, feature_size)
        
        # Branch
        self.fc6 = nn.Linear(feature_size, class_num)
        self.fc7 = nn.Linear(feature_size, feature_size-class_num)

    def forward(self, x):
        
        # Encoder
        out = F.relu(self.fc1_bn(self.fc1(x)))
        out = F.relu(self.fc2_bn(self.fc2(out)))
        encoded = F.sigmoid(self.fc3(out))        
        
        # Stacked Relation Module
        rel_module_res1 = self.relation_module(encoded, self.unary1, self.ps1, self.ph1, self.wr1)
        rel_module_res2 = self.relation_module(rel_module_res1, self.unary2, self.ps2, self.ph2, self.wr2)
        rel_module_res3 = self.relation_module(rel_module_res2, self.unary3, self.ps3, self.ph3, self.wr3)        
        rel_module_res4 = self.relation_module(rel_module_res3, self.unary4, self.ps4, self.ph4, self.wr4)
        
        # Decoder
        out = F.relu(self.fc4_bn(self.fc4(rel_module_res4)))
        out = F.relu(self.fc5(out))
        
        # Branch
        cls = self.fc6(out)
        geo = self.fc7(out)
        
        # Refined layout
        res = torch.cat((cls,geo), 2)
        pts('res', res)
        
        return res

    def relation_module(self, out, unary, psi, phi, wr):
        element_num = out.size(1)
        batch_res = []
        for bdx, batch in enumerate(out):
            f_prime = []
            for idx, i in enumerate(batch):
                self_attention = torch.Tensor(torch.zeros(i.size(0)))
                for jdx, j in enumerate(batch):           
                    if idx == jdx:
                        continue

                    u = F.relu(unary(j))                        
                    iv = i.view(i.size(0),1)
                    jv = j.view(j.size(0),1)
                    dot = (torch.mm((iv*psi).t(), jv*phi)).squeeze()
                    self_attention += dot*u

                f_prime.append(wr*(self_attention/element_num) + i )
            batch_res.append(torch.stack(f_prime))
        return torch.stack(batch_res)
        
    
class Discriminator(nn.Module):

    def __init__(self, feature_size=3, class_num=1, rel_module=True, element_num=128):
        super(Discriminator, self).__init__()
        self.feature_size = feature_size
        self.rel_module = rel_module
        self.element_num = element_num
        
        # Encoder
        self.fc1 = nn.Linear(feature_size, feature_size*2)
        self.fc1_bn = nn.BatchNorm1d(element_num)
        self.fc2 = nn.Linear(feature_size*2, feature_size*2*2)
        self.fc2_bn = nn.BatchNorm1d(element_num)
        self.fc3 = nn.Linear(feature_size*2*2, feature_size*2*2)
        
        # Relation module1
        self.unary1 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps1 = torch.FloatTensor(torch.rand(1))
        self.ph1 = torch.FloatTensor(torch.rand(1))
        self.wr1 = torch.FloatTensor(torch.rand(1))

        # Relation module2
        self.unary2 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps2 = torch.FloatTensor(torch.rand(1))
        self.ph2 = torch.FloatTensor(torch.rand(1))
        self.wr2 = torch.FloatTensor(torch.rand(1))

        # Relation module3
        self.unary3 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps3 = torch.FloatTensor(torch.rand(1))
        self.ph3 = torch.FloatTensor(torch.rand(1))
        self.wr3 = torch.FloatTensor(torch.rand(1))

        # Relation module4
        self.unary4 = nn.Linear(feature_size*2*2, feature_size*2*2)
        self.ps4 = torch.FloatTensor(torch.rand(1))
        self.ph4 = torch.FloatTensor(torch.rand(1))
        self.wr4 = torch.FloatTensor(torch.rand(1))
        
        # Decoder
        self.fc4 = nn.Linear(feature_size*2*2, feature_size*2)
        self.fc4_bn = nn.BatchNorm1d(element_num)
        self.fc5 = nn.Linear(feature_size*2, feature_size)
        
        # Branch
        self.fc6 = nn.Linear(feature_size, class_num)
        self.fc7 = nn.Linear(feature_size, feature_size-class_num) 
        
        # Max pooling
        self.mp = nn.MaxPool1d(element_num, stride=2)
        
        # Logits
        self.fc8 = nn.Linear(feature_size, 1)
        
    def forward(self, x):
        
        # Encoder
        out = F.relu(self.fc1_bn(self.fc1(x)))
        out = F.relu(self.fc2_bn(self.fc2(out)))
        out = F.sigmoid(self.fc3(out))
        
        if self.rel_module:
            # Stacked Relation Module
            rel_module_res1 = self.relation_module(out, self.unary1, self.ps1, self.ph1, self.wr1)
            rel_module_res2 = self.relation_module(rel_module_res1, self.unary2, self.ps2, self.ph2, self.wr2)
            rel_module_res3 = self.relation_module(rel_module_res2, self.unary3, self.ps3, self.ph3, self.wr3)        
            out = self.relation_module(rel_module_res3, self.unary4, self.ps4, self.ph4, self.wr4)
        else :
            # Wireframe Module
            pass

        # Decoder
        out = F.relu(self.fc4_bn(self.fc4(out)))
        out = F.relu(self.fc5(out))
        
        # Branch
        cls = self.fc6(out)
        geo = self.fc7(out)
        
        # Refined layout
        res = torch.cat((cls,geo), 2)
        
        # Max Pooling
        pres = self.max_pooling(res, self.mp)
        
        # Logits
        pred = F.sigmoid(self.fc8(pres))
        pts('pred', pred)
        return pred
    
    def max_pooling(self, out, mp):
        batch_res = []
        for bdx, batch in enumerate(out):
            ns = []
            for i in range(self.feature_size):
                ns.append(batch[: , i:i+1].squeeze())
            ns = torch.stack(ns)
            ns = ns.view(1,self.feature_size,self.element_num)
            batch_res.append(mp(ns).squeeze())

        res = torch.stack(batch_res).view(-1, self.feature_size)
        return res

    def relation_module(self, out, unary, psi, phi, wr):
        element_num = out.size(1)
        batch_res = []
        for bdx, batch in enumerate(out):
            f_prime = []
            for idx, i in enumerate(batch):
                self_attention = torch.Tensor(torch.zeros(i.size(0)))
                for jdx, j in enumerate(batch):           
                    if idx == jdx:
                        continue

                    u = F.relu(unary(j))                        
                    iv = i.view(i.size(0),1)
                    jv = j.view(j.size(0),1)
                    dot = (torch.mm((iv*psi).t(), jv*phi)).squeeze()
                    self_attention += dot*u

                f_prime.append( wr*(self_attention/element_num)) # No shortcut.
            batch_res.append(torch.stack(f_prime))
        return torch.stack(batch_res)
    