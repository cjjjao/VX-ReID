from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData,RX01Data,ChannelAdapGray
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from Baseline import Moco_net
from utils import *
from loss import SmoothAP_MC, SmoothAP_Cross, CenterTripletLoss, TripletLoss_WRT, expATLoss
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='RX01', help='dataset name: regdb or sysu or RX01]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='RX01_agw_p8_n4_lr_0.1_seed_0_trial_camera_deleteS_best.t', type=str, help='resume from checkpoint detachMAM_MoCo_best.t')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw192', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh384', help='img height')
parser.add_argument('--batch-size', default=4, type=int, metavar='4B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=8, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='indoor', type=str, help='all or indoor')
parser.add_argument('--delta', default=0.2, type=float, metavar='delta', help='dcl weights, 0.2 for PCB, 0.5 for resnet50')
parser.add_argument('--K', default=8, type=int, metavar='kk')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    times = 10
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    times = 5
    test_mode = [2, 1]  # visible to thermal [2, 1]
elif dataset == 'RX01':
    log_path = args.log_path + 'rx01_log/'
    data_path = '../Datasets/RX2_07/train/'#'/data/cja/cja_proj/Datasets/RX2_07/train/' #'/media/data/cja/cja_proj/Datasets/RX2_07/train/'
    times = 1
    test_mode = [1, 2]  # thermal to visible

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method=='agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'sysu':
    suffix = suffix + '_trial_MoCo_bug'

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

if dataset == 'RX01':
    suffix = suffix + '_trial_camera_deleteS'

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 5 * 2048 #2048 * 5

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0]),
])
transform_train_v = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0]),
])

transform_train_x = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ChannelAdapGray(probability=0.5),
    RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'RX01':
    #trainset = RX01Data(data_path, transform=transform_train)
    trainset = RX01Data(data_path, transform_v=transform_train, transform_x=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    # testing set
    query_img, query_label, query_cam = process_query_RX01(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_RX01(data_path, mode=args.mode, trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool = 'off', arch=args.arch)
else:
    #net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
    net = Moco_net(n_class, arch=args.arch) #P_net(n_class, arch=args.arch,K = args.K) #Chnal_net(n_class, arch=args.arch) Moco_net(n_class, arch=args.arch)
#modality_classifier = Discriminator(3, 3)
#modality_classifier.apply(weights_init_classifier)

net.to(device)
#modality_classifier.to(device)
cudnn.benchmark = True

# define loss function
criterion_id = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion_sap = SmoothAP_MC(0.1,args.batch_size * 2 * args.num_pos,args.batch_size, 2048) #SmoothAP(0.1,args.batch_size * 2 * args.num_pos,args.batch_size,2048) #SmoothAP_MC(0.1,args.batch_size * 2 * args.num_pos,args.batch_size, 2048,320)

criterion_id.to(device)
criterion_sap.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck1.parameters())) \
                     + list(map(id, net.bottleneck2.parameters())) \
                     + list(map(id, net.bottleneck3.parameters())) \
                     + list(map(id, net.bottleneck4.parameters())) \
                     + list(map(id, net.classifier1.parameters())) \
                     + list(map(id, net.classifier2.parameters())) \
                     + list(map(id, net.classifier3.parameters())) \
                     + list(map(id, net.classifier4.parameters())) \
                     + list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) \

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck1.parameters(), 'lr': args.lr},
        {'params': net.bottleneck2.parameters(), 'lr': args.lr},
        {'params': net.bottleneck3.parameters(), 'lr': args.lr},
        {'params': net.bottleneck4.parameters(), 'lr': args.lr},
        {'params': net.classifier1.parameters(), 'lr': args.lr},
        {'params': net.classifier2.parameters(), 'lr': args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.classifier3.parameters(), 'lr': args.lr},
        {'params': net.classifier4.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer, epoch, net=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
        net.m = 0.8
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
        net.m = 0.8
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
        net.m = 0.8 #0.3
    elif epoch >= 50:
        lr = args.lr * 0.01
        net.m = 0.8 #0.7

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    #modal_classifier_optimizer.param_groups[0]['lr'] = lr

    return lr

start_epoch = 0
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['opti'])
        best_acc = checkpoint['cmc'][0]
        # modality_classifier.load_state_dict(checkpoint['net_m'])
        # modal_classifier_optimizer.load_state_dict(checkpoint['opti_m'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch, net)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    l2_l = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    total = 0
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)
        b = labels.shape[0]

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())

        data_time.update(time.time() - end)

        #part pooling 4
        feat1, feat2, feat3, feat4, feat, queue1, queue2, queue3, queue4, queueg, out1, out2, out3, out4, outg, l2_loss = net(input1, input2) #, out1, out2, out3, out4 outg,queue1, queue2, queue3, queue4,
         #, l2_loss , targets_a, targets_b targets=labels , cls_v, cls_r , mae_loss
        loss_tri = (criterion_sap(feat1, queue1[:4 * b], queue1[:b]) + criterion_sap(feat2, queue2[:4 * b],queue2[:b]) + criterion_sap(feat3,queue3[:4 * b],queue3[:b]) + criterion_sap(feat4, queue4[:4 * b], queue4[:b])) * 0.25 + criterion_sap(feat, queueg[:4 * b], queueg[:b]) * 0.75
        _, predicted = out1.max(1)
        best_acc = predicted.eq(labels).sum().item() / b

        loss_id = (criterion_id(out1, labels) + criterion_id(out2, labels) + criterion_id(out3, labels) + criterion_id(
            out4, labels)) * 0.25 + criterion_id(outg, labels) * 0.75

        loss = loss_id + loss_tri + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))

        tri_loss.update(loss_tri.item(), b)
        l2_l.update(l2_loss.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 200 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'triLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'LLoss: {l2_l.val:.4f} ({l2_l.avg:.4f}) '
                  'Accu: {:.2f}%'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                best_acc * 100, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, l2_l = l2_l)) #,l2_l=l2_l  #'LLoss: {l2_l.val:.4f} ({l2_l.avg:.4f}) ' 'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) ' 'wLoss: {w_loss.val:.4f} ({w_loss.avg:.4f}) '

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('l2_l', l2_l.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def extract_gall_feat(gall_loader):
    net.eval()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    #gall_img = np.zeros((ngall, 384, 192, 3))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, modal = test_mode[0])
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
            #gall_img[ptr:ptr + batch_num, :] = img
            ptr = ptr + batch_num
    #print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc #, gall_img  # , Xgall_feat_pool, Xgall_feat_fc


def extract_query_feat(query_loader):
    net.eval()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    #que_img = np.zeros((nquery, 384, 192, 3))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, modal = test_mode[1])
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
            #que_img[ptr:ptr + batch_num, :] = img
            # Xquery_feat_pool[ptr:ptr+batch_num,: ] = feat_pool[batch_num:].detach().cpu().numpy()
            # Xquery_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc[batch_num:].detach().cpu().numpy()
            ptr = ptr + batch_num
    #print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc #, que_img # , Xquery_feat_pool, Xquery_feat_fc

def test(epoch):
    # switch to evaluation mode

    query_img, query_label, query_cam = process_query_RX01(data_path, mode=args.mode)
    #query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible') #'visible'
    #query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    #gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)  # , que_img, Xquery_feat_pool, Xquery_feat_fc
    for trial in range(times):
        #gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)
        gall_img, gall_label, gall_cam = process_gallery_RX01(data_path, mode=args.mode, trial=0)
        #gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal') #'thermal'
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader) #, ga_img
        # , Xgall_feat_pool, Xgall_feat_fc
        # fc feature
        distmat_att = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        distmat = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam) #, gall_img = ga_img,que_img = que_img
        # cmc, mAP, mINP = eval_sysu(-distmat, gall_label, query_label)
        # cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, gall_label, query_label)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

            all_cmc_att = cmc_att
            all_mAP_att = mAP_att
            all_mINP_att = mINP_att

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

            all_cmc_att = all_cmc_att + cmc_att
            all_mAP_att = all_mAP_att + mAP_att
            all_mINP_att = all_mINP_att + mINP_att
    return all_cmc / times, all_mAP / times, all_mINP / times, all_cmc_att / times, all_mAP_att / times, all_mINP_att / times

# training
print('==> Start Training...')
update = 0.8
for epoch in range(start_epoch, 82):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)
    # training
    train(epoch)

    if epoch >= 0 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        #start = time.time()
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch) #, \
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
                'opti' : optimizer.state_dict(),
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 0 and (epoch % args.save_epoch == 0 or epoch == 19):
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
                'opti' : optimizer.state_dict(),
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))