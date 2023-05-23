import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from net.mfnet import MFNet
from utils.tdataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import logging
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = False

def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

    return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        # lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0, edge_map = model(images)
        lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0, = model(images)
        # lateral_map_3 = model(images)
        # lateral_map_3 = model(images)
        # ---- loss function ----
        # loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        loss0 = structure_loss(lateral_map_0, gts)
        # losse = adaptive_pixel_intensity_loss(edge_map, edges)
        loss = loss3 + loss2 + loss1 + loss0 
        # loss = loss3 + loss2 + loss1 + loss0 + losse
        # loss = loss3
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        # loss_recorde.update(losse.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                '[lateral-3: {:.4f}]'.
                format(epoch, opt.epoch, i, total_step,
                        loss_record3.avg))

            logging.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[lateral-3: {:.4f}]'.
                    format(epoch, opt.epoch, i, total_step,
                        loss_record3.avg))
        # if i % 60 == 0 or i == total_step:
        #     print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
        #         '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
        #         format(epoch, opt.epoch, i, total_step,
        #                 loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

        #     logging.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
        #             '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
        #             format(epoch, opt.epoch, i, total_step,
        #                 loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if epoch > 20:
        if epoch % 1 == 0 or epoch == opt.epoch:
            torch.save(model.state_dict(), save_path + 'BGNet-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'BGNet-%d.pth' % epoch)
    
        
def val_camo(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('CAMO_MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

def val_chameleon(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae1, best_epoch1
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('CHAMELEON_MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae1, best_epoch1))
        if epoch == 1:
            best_mae1 = mae
        else:
            if mae < best_mae1:
                best_mae1 = mae
                best_epoch1 = epoch
                # torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                # print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch1, best_mae1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=60, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=416, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--val_root', type=str, default='./data/TestDataset',
                        help='the test rgb images root')
    parser.add_argument('--train_save', type=str,
                        default='mfnet')
    opt = parser.parse_args()

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")

    # ---- build models ----
    model = MFNet().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    # val_loader = test_dataset(image_root='{}/Imgs/'.format(opt.val_root),
    #                           gt_root='{}/GT/'.format(opt.val_root),
    #                           testsize=opt.trainsize)
    val_camo_loader = test_dataset(image_root='{}/CAMO/Imgs/'.format(opt.val_root),
                              gt_root='{}/CAMO/GT/'.format(opt.val_root),
                              testsize=opt.trainsize)
    val_chameleon_loader = test_dataset(image_root='{}/CHAMELEON/Imgs/'.format(opt.val_root),
                              gt_root='{}/CHAMELEON/GT/'.format(opt.val_root),
                              testsize=opt.trainsize)
    total_step = len(train_loader)
    writer = SummaryWriter(save_path + 'summary')
    print("Start Training")
    best_mae = 1
    best_epoch = 0
    best_mae1 = 1
    best_epoch1 = 0
    for epoch in range(1, opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
        if epoch>20:
            val_camo(val_camo_loader, model, epoch, save_path, writer)
            val_chameleon(val_chameleon_loader, model, epoch, save_path, writer)

