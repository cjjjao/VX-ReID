import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

         
class DCLoss(nn.Module):
    def __init__(self, num=2):
        super(DCLoss, self).__init__()
        self.num = num
        self.fc1 = nn.Sequential(nn.Linear(2048, 256, bias=False)) 
        self.fc2 = nn.Sequential(nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256), nn.BatchNorm1d(256))
        
    def forward(self, x):
        x = F.normalize(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x)
        loss = 0
        num = int(x.size(0) / self.num) #32
        for i in range(self.num):
            for j in range(self.num):
                if i<j:
                    loss += ((x[i*num:(i+1)*num,:] - x[j*num:(j+1)*num,:]).norm(dim=1,keepdim=True)).mean()
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        inputs = torch.nn.functional.normalize(inputs, dim=1)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class expATLoss(nn.Module):
    def __init__(self,bs,pre_id):
        super(expATLoss, self).__init__()
        self.bs = bs
        self.pre_id = pre_id
        self.marginloss = torch.nn.MarginRankingLoss(margin = 1)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        inputs = F.normalize(inputs, p=2, dim=-1)
        dist = F.relu(torch.matmul(inputs,inputs.t()))
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        diag = (torch.eye(n,n).eq(torch.zeros((n,n)))).cuda()
        dist_ap, dist_an = [], []
        for i in range(n):
            ap_i = torch.randint(low=0, high=self.pre_id-1,size=(1,))
            an_i = torch.randint(low=0,high = self.bs-self.pre_id,size=(1,))
            dist_ap.append((dist[i][mask[i] * diag[i]])[ap_i[0]].unsqueeze(0))
            dist_an.append((dist[i][mask[i] == 0])[an_i[0]].unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y_true = inputs.new().resize_as_(inputs).fill_(1)[:,0:1]
        return torch.exp(self.marginloss(dist_ap, dist_an.float(), y_true)) # max(

class TriLoss(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        super(TriLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        
        vis, vtm, nir, ntm = torch.chunk(inputs, 4, 0)
        
        input1 = torch.cat((vis, ntm), 0)
        n = input1.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_vm = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_vm = dist_vm + dist_vm.t()
        dist_vm.addmm_(1, -2, input1, input1.t())
        dist_vm = dist_vm.clamp(min=1e-12).sqrt()  # for numerical stability
        
        
        input2 = torch.cat((vtm, nir), 0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_nm = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_nm = dist_nm + dist_nm.t()
        dist_nm.addmm_(1, -2, input2, input2.t())
        dist_nm = dist_vm.clamp(min=1e-12).sqrt()  # for numerical stability
        

        input3 = torch.cat((vis, nir), 0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_vn = torch.pow(input3, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_vn = dist_vn + dist_vn.t()
        dist_vn.addmm_(1, -2, input3, input3.t())
        dist_vn = dist_vn.clamp(min=1e-12).sqrt()  # for numerical stability
        
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist_vn[i][mask[i]].mean().unsqueeze(0))
            dist_an1.append(dist_vn[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)
        dist_ap1 = dist_ap1.mean()
        dist_an1 = dist_an1.mean()
        

        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist_vm[i][mask[i]].mean().unsqueeze(0))
            dist_an2.append(dist_vm[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        dist_ap2 = dist_ap2.mean()
        dist_an2 = dist_an2.mean()
        

        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist_nm[i][mask[i]].mean().unsqueeze(0))
            dist_an3.append(dist_nm[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)
        dist_ap3 = dist_ap3.mean()
        dist_an3 = dist_an3.mean()
        
        #print(dist_ap1.mean(), dist_ap2.mean(), dist_ap3.mean())
        #print(dist_an1.mean(), dist_an2.mean(), dist_an3.mean())

        if  dist_ap2 > dist_ap3:
        
            loss1 = torch.abs(dist_ap2 - dist_ap3.detach())# + dist_an2.detach() - dist_an3
        else:
            loss1 = torch.abs(dist_ap2.detach() - dist_ap3)# + dist_an2.detach() - dist_an3

        return loss1# + loss2

class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class SmoothAP_MC(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.

    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.

    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:

        labels = ( A, A, A, B, B, B, C, C, C)

    (the order of the classes however does not matter)

    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.

    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings

    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar

    Examples::

        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims,gallery_size = None):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP_MC, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds, gallery, gallery_pos):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        self.gallery_size,self.feat_dims = gallery.size()
        preds = preds.reshape(2, self.num_id, -1, self.feat_dims)
        preds = preds.permute(1, 0, 2, 3).reshape(-1, self.feat_dims)

        sim_all = self.compute_aff(preds,gallery) #[bs,gs]
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.gallery_size, 1) #[bs,gs,gs]
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)# [bs,gs,gs]
        # pass through the sigmoid
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) #[bs,bs,bs]
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1)  #[bs,gs]

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims) #[n,pn,d]
        gallery_pos = gallery_pos.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)

        ########################
        nx = int(self.batch_size / self.num_id)
        distx = torch.pow(xs, 2).sum(dim=-1, keepdim=True).expand(self.num_id, nx, nx)
        disty = torch.pow(gallery_pos, 2).sum(dim=-1, keepdim=True).expand(self.num_id, nx, nx)
        sim_pos = distx + disty.permute(0, 2, 1)
        sim_pos = sim_pos - 2 * torch.bmm(xs, gallery_pos.permute(0,2,1))
        sim_pos = -sim_pos.clamp(min=1e-12).sqrt()
        ########################
        #sim_pos = torch.bmm(xs, gallery_pos.permute(0, 2, 1))#[n,pn,pn]
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)#[n,pn,pn,pn]
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)#[n,pn,pn,pn]
        # pass through the sigmoid
        sim_pos_sg = self.sigmoid(sim_pos_diff, temp=self.anneal) #[n,pn,pn,pn]
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1)  #[n,pn,pn]

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)
        return (1-ap)

    def compute_aff(self,x, y = None):
        """computes the affinity matrix between an input vector and itself"""
        nx = x.size(0)
        ny = y.size(0)
        # Compute pairwise distance, replace by the official when merged
        distx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(nx, ny)
        disty = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(ny, nx)
        dist = distx + disty.t()
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return -dist

    def sigmoid(self,tensor, temp=1.0):
        """ temperature controlled sigmoid

        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

class SmoothAP_Cross(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.

    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.

    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:

        labels = ( A, A, A, B, B, B, C, C, C)

    (the order of the classes however does not matter)

    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.

    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings

    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar

    Examples::

        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims,gallery_size = None):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP_Cross, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds, gallery):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        self.gallery_size,self.feat_dims = gallery.size()
        preds = torch.nn.functional.normalize(preds,dim=1)
        gallery = torch.nn.functional.normalize(gallery, dim=1)

        sim_all = self.compute_aff(preds,gallery) #[bs,gs]
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.gallery_size, 1) #[bs,gs,gs]
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)# [bs,gs,gs]
        # pass through the sigmoid
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) #[bs,bs,bs]
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1)  #[bs,gs]

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims) #[n,pn,d]
        gallery_pos = gallery.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)

        sim_pos = torch.bmm(xs, gallery_pos.permute(0, 2, 1))#[n,pn,pn]
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)#[n,pn,pn,pn]
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)#[n,pn,pn,pn]
        # pass through the sigmoid
        sim_pos_sg = self.sigmoid(sim_pos_diff, temp=self.anneal) #[n,pn,pn,pn]
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1)  #[n,pn,pn]

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)

    def compute_aff(self,x, y = None):
        """computes the affinity matrix between an input vector and itself"""
        if y != None:
            return torch.mm(x, y.t())
        return torch.mm(x, x.t())

    def sigmoid(self,tensor, temp=1.0):
        """ temperature controlled sigmoid

        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y
        
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx