import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F
import shutil

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        #x = self.visible.layer1(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        #x = self.thermal.layer1(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)        
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)
        
        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)
        
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3.apply(weights_init_classifier)
        
        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)  # no shift
        self.bottleneck4.apply(weights_init_kaiming)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4.apply(weights_init_classifier)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encode1 = nn.Conv2d(3, 1, 1)
        self.encode1.apply(my_weights_init)
        self.fc1 = nn.Conv2d(1, 1, 1)
        self.fc1.apply(my_weights_init)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn1.apply(weights_init_kaiming)

        
        self.encode2 = nn.Conv2d(3, 1, 1)
        self.encode2.apply(my_weights_init)
        self.fc2 = nn.Conv2d(1, 1, 1)
        self.fc2.apply(my_weights_init)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn2.apply(weights_init_kaiming)


        self.decode = nn.Conv2d(1, 3, 1)
        self.decode.apply(my_weights_init)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            gray1 = F.relu(self.encode1(x1))
            gray1 = self.bn1(F.relu(self.fc1(gray1)))

            gray2 = F.relu(self.encode2(x2))
            gray2 = self.bn2(F.relu(self.fc2(gray2)))            
            
            gray = F.relu(self.decode(torch.cat((gray1, gray2),0)))
            
            gray1, gray2 = torch.chunk(gray, 2, 0)
            xo = torch.cat((x1, x2), 0)

            x1 = self.visible_module(torch.cat((x1, gray1),0))
            x2 = self.thermal_module(torch.cat((x2, gray2),0))

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            gray1 = F.relu(self.encode1(x1))
            gray1 = self.bn1(F.relu(self.fc1(gray1)))
            gray1 = F.relu(self.decode(gray1))

            x = self.visible_module(torch.cat((x1, gray1),0))
        elif modal == 2:
            gray2 = F.relu(self.encode2(x2))
            gray2 = self.bn2(F.relu(self.fc2(gray2)))
            gray2 = F.relu(self.decode(gray2))

            x = self.thermal_module(torch.cat((x2, gray2),0))


        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        x = self.base_resnet.base.layer4(x)
        x41, x42, x43, x44 = torch.chunk(x, 4, 2) #四个模态
        
        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))

        feat41 = self.bottleneck1(x41)
        feat42 = self.bottleneck2(x42)
        feat43 = self.bottleneck3(x43)
        feat44 = self.bottleneck4(x44)

        if self.training:
            return x41, x42, x43, x44, self.classifier1(feat41), self.classifier2(feat42), self.classifier3(feat43), self.classifier4(feat44), [xo, gray]
        else:
            return self.l2norm(torch.cat((x41, x42, x43, x44),1)), self.l2norm(torch.cat((feat41, feat42, feat43, feat44),1))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Linear(channel, channel//reduction,bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel//reduction,channel, bias=False),
        nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(channel, track_running_stats=False)
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.training:
            return x * y + (1 - y) * self.IN(x), y.view(b, c)
        return x * y + (1 - y) * self.IN(x) #,y.view(b,c)

class cro_chnal_net(nn.Module):
    def __init__(self,channel):
        super(cro_chnal_net, self).__init__()
        self.IN = nn.InstanceNorm2d(channel, track_running_stats=False)

    def forward(self, x1, x2):
        b,c,h,w = x1.size()
        x1 = x1.view(b, c, -1)
        x1 = F.normalize(x1, p=2, dim = -1)
        x2 = x2.view(b, c, -1)
        x2 = F.normalize(x2, p=2, dim=-1)
        relation = torch.matmul(x1,x2.permute(0,2,1))
        chnal_v = torch.max(relation,dim = -1)[0]
        chnal_x = torch.max(relation.permute(0,2,1),dim = -1)[0]
        return chnal_v, chnal_x

class Base_net(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(Base_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.l2_loss = nn.MSELoss() #nn.KLDivLoss(reduction='batchmean')   #nn.CosineSimilarity(dim=-1, eps=1e-6)
        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.SeN1 = SELayer(channel=1024)
        self.CroC1 = cro_chnal_net(channel=1024)

    def forward(self, x1, x2, modal=0,img=None):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        b,c,h,w = x.size()
        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        if modal == 0:
            chnal_label1,chnal_label2 = self.CroC1(x[:b//2],x[b//2:])
            x, chnal1 = self.SeN1(x)
            chnal_label12 = torch.cat((chnal_label1,chnal_label2),0).detach()
            l2loss = self.l2_loss(chnal_label12, chnal1)
        else:
            x = self.SeN1(x)
        x = self.base_resnet.base.layer4(x)
        if self.training:
            return x ,l2loss
        else:
            return x

    def visualize_feat(self,featmap, img, chnal_label):
        select_id = [0, 32, 4, 36, 8, 40, 12, 44, 16, 48, 20, 52, 24, 56, 28, 60]  # [0,32,4,36,8,40,12,44] [16,48,20,52,24,56,28,60]
        b, c, h, w = featmap.size()
        featmap = featmap.data.cpu().numpy()
        _, inx = torch.sort(chnal_label,dim=-1,descending=True)
        #img = img.data.cpu().numpy()
        for i in select_id:
            im = img[i]
            # im = im.reshape(-1, 192 * 384)
            # im = im - np.min(im, axis=1, keepdims=True)
            # im = im / np.max(im, axis=1, keepdims=True)
            # ori_img = im.reshape(3, 384, 192).swapaxes(0, 1).swapaxes(1, 2)
            ori_img = cv2.cvtColor(np.uint8(im), cv2.COLOR_RGB2BGR)
            path = './heatmap/' + str(i)
            #if not os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
            # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path + '/' + 'ori.jpg', ori_img)
            for k, id in enumerate(inx[i,:100]):
                mask = featmap[i, id].reshape(-1, h * w)
                mask = mask - np.min(mask, axis=1)
                mask = mask / np.max(mask, axis=1)
                # mask = 1 - mask
                mask = mask.reshape(h, w)
                cam_img = np.uint8(255 * mask)
                #cam_img = cv2.resize(cam_img, (144, 288))
                cam_img = cv2.applyColorMap(cv2.resize(cam_img, (192, 384)), cv2.COLORMAP_JET)
                cam_img = cam_img * 0.3 + ori_img * 0.5
                cv2.imwrite(path + '/' + str(k) + '_' + str(id) + '.jpg', cam_img)

class Moco_net(nn.Module):
    def __init__(self, class_num, arch='resnet50',dim=2048, K=256, m=0.1, T=0.07, mlp=False):
        super(Moco_net, self).__init__()

        pool_dim = 2048
        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)
        #
        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)
        #
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3.apply(weights_init_classifier)
        #
        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)  # no shift
        self.bottleneck4.apply(weights_init_kaiming)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = Base_net(class_num, arch=arch)
        self.encoder_k = Base_net(class_num, arch=arch)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue1", torch.randn(K, dim))
        self.register_buffer("queue2", torch.randn(K, dim))
        self.register_buffer("queue3", torch.randn(K, dim))
        self.register_buffer("queue4", torch.randn(K, dim))
        self.register_buffer("queueg", torch.randn(K, dim))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1,keys2,keys3,keys4,keysg): #,keys5,keys6 , keysg
        batch_size = keys1.shape[0]

        assert self.K % batch_size == 0  # for simplicity

        keys1 = self.tune(keys1)
        self.queue1[batch_size:,:] = self.queue1[:self.K - batch_size,:].clone()  # keys.T
        self.queue1[:batch_size,:] = keys1
        keys2 = self.tune(keys2)
        self.queue2[batch_size:,:] = self.queue2[:self.K - batch_size,:].clone()   # keys.T
        self.queue2[:batch_size,:] = keys2
        keys3 = self.tune(keys3)
        self.queue3[batch_size:,:] = self.queue3[:self.K - batch_size,:].clone()   # keys.T
        self.queue3[:batch_size,:] = keys3
        keys4 = self.tune(keys4)
        self.queue4[batch_size:,:] = self.queue4[:self.K - batch_size,:].clone()   # keys.T
        self.queue4[:batch_size,:] = keys4
        keysg = self.tune(keysg)
        self.queueg[batch_size:, :] = self.queueg[:self.K - batch_size, :].clone()  # keys.T
        self.queueg[:batch_size, :] = keysg

    def tune(self,preds):
        self.feat_dims = preds.shape[1]
        preds = preds.reshape(2, 4, -1, 2048)
        preds = preds.permute(1, 0, 2, 3).reshape(-1, 2048)
        return preds

    def forward(self, x1, x2, modal=0,img1=None,img2=None):
        if self.training:
            q, l2loss = self.encoder_q(x1,x2,modal)  # queries: NxC, l2loss
        else:
            q = self.encoder_q(x1,x2,modal)

        if self.training:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                k, _ = self.encoder_k(x1,x2,modal)  # keys: NxC

        x41, x42, x43, x44 = torch.chunk(q, 4, 2)  # 四个模态, x45, x46
        if self.training:
            xk41, xk42, xk43, xk44 = torch.chunk(k, 4, 2) #, xk45, xk46

        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        q = self.avgpool(q)

        if self.training:
            xk41 = self.avgpool(xk41).view(x41.size(0), x41.size(1))
            xk42 = self.avgpool(xk42).view(x41.size(0), x41.size(1))
            xk43 = self.avgpool(xk43).view(x41.size(0), x41.size(1))
            xk44 = self.avgpool(xk44).view(x41.size(0), x41.size(1))
            k = self.avgpool(k).view(k.size(0),k.size(1))
            self._dequeue_and_enqueue(xk41, xk42, xk43, xk44, k) #, xk45, xk46, k

        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))
        q = q.view(q.size(0), q.size(1))

        feat41 = self.bottleneck1(x41)
        feat42 = self.bottleneck2(x42)
        feat43 = self.bottleneck3(x43)
        feat44 = self.bottleneck4(x44)
        q_b = self.bottleneck(q)

        if self.training:   #self.classifier(q_b),
            return x41, x42, x43, x44, q, self.queue1, self.queue2, self.queue3, self.queue4, self.queueg, self.classifier1(feat41), self.classifier2(feat42), self.classifier3(feat43), self.classifier4(feat44), self.classifier(q_b), l2loss #, self.classifier5(feat45), self.classifier6(feat46)
        else:
            return self.l2norm(torch.cat((q, x41, x42, x43, x44), 1)), self.l2norm(
                torch.cat((q_b, feat41, feat42, feat43, feat44), 1))