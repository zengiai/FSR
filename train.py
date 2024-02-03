import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from models.resnet_fsr import ResNet18_FSR
from models.vgg_fsr import vgg16_FSR
from models.wideresnet34_fsr import WideResNet34_FSR
# try
from models.resnet_fsr_1 import ResNet18_FSR_1

from attacks.pgd import PGD

from tqdm.auto import tqdm

import argparse
import os


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='FSR Training')
parser.add_argument('--save_name', type=str, help='specify checkpoint save name')
parser.add_argument('--lam_sep', type=float, default=1.0, help='weight for separation loss')
parser.add_argument('--lam_rec', type=float, default=1.0, help='weight for recalibration loss')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for classifier')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
parser.add_argument('--dataset', type=str, default='cifar10', help='target dataset')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
parser.add_argument('--eps', type=float, default=8., help='perturbation constraint epsilon')
parser.add_argument('--alpha', type=float, default=0.25, help='step size alpha')
parser.add_argument('--tau', type=float, default=0.1, help='tau for Gumbel softmax')
parser.add_argument('--device', type=int, help='device id')
args = parser.parse_args()

print(args)

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
start_epoch = 1

if args.dataset == 'cifar10':
    num_classes = 10
    image_size = (32, 32)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='constant', value=0).squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)

elif args.dataset == 'svhn':
    num_classes = 10
    image_size = (32, 32)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root='../data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)

    testset = torchvision.datasets.SVHN(
        root='../data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)

models = {
    'resnet18': ResNet18_FSR(tau=args.tau, num_classes=num_classes, image_size=image_size),
    'vgg16': vgg16_FSR(tau=args.tau, num_classes=num_classes, image_size=image_size),
    'wideresnet34': WideResNet34_FSR(tau=args.tau, num_classes=num_classes, image_size=image_size),
    'resnet18_1': ResNet18_FSR_1(tau=args.tau, num_classes=num_classes, image_size=image_size),
}

model_name = args.model
net = models[model_name]
net = net.to(device)
cudnn.benchmark = True


criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 我的尝试
klLoss = nn.KLDivLoss(reduction='mean')
BCE_stable = nn.BCEWithLogitsLoss(reduction='mean')




def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)
    return adv_label


attack = PGD(net, args.eps/255.0, args.alpha * (args.eps/255.0), min_val=0, max_val=1, max_iters=10, _type='linf')


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    adv_cls_losses = 0
    sep_losses = 0
    rec_losses = 0
    adv_correct = 0
    total = 0

    r_nr_losses = 0

    adjust_learning_rate(optimizer, epoch)

    with tqdm(total=(len(trainset) - len(trainset) % args.bs)) as _tqdm:
        _tqdm.set_description('{} (Train) Epoch: {}/{}'.format(args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            net.eval()
            adv_inputs = attack.perturb(inputs, targets, True)
            net.train()

            nat_outputs, _, _, _ = net(inputs)
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = net(adv_inputs)
            # 对抗标签，获取top2标签
            adv_labels = get_pred(adv_outputs, targets)
            # 分类损失值
            adv_cls_loss = criterion(adv_outputs, targets)

            # sep损失和rec损失
            r_loss = torch.tensor(0.).to(device)
            if not len(adv_r_outputs) == 0:
                for r_out in adv_r_outputs:
                    r_loss += args.lam_sep * criterion(r_out, targets)
                r_loss /= len(adv_r_outputs)

            nr_loss = torch.tensor(0.).to(device)
            if not len(adv_nr_outputs) == 0:
                for nr_out in adv_nr_outputs:
                    nr_loss += args.lam_sep * criterion(nr_out, adv_labels)
                nr_loss /= len(adv_nr_outputs)
            sep_loss = r_loss + nr_loss

            rec_loss = torch.tensor(0.).to(device)
            if not len(adv_rec_outputs) == 0:
                for rec_out in adv_rec_outputs:
                    rec_loss += args.lam_rec * criterion(rec_out, targets)
                rec_loss /= len(adv_rec_outputs)

            # 我的尝试----r_nr_loss
            # t_real = torch.ones((adv_outputs.size(0), 1))
            # t_fake = torch.zeros((adv_outputs.size(0), 1))
            #
            # r_nr_loss = torch.tensor(0.).to(device)
            # if not len(adv_r_outputs) == 0 and len(adv_nr_outputs) == 0:
            #     for r_out, nr_out in adv_r_outputs, adv_nr_outputs:
            #         r_nr_loss += (BCE_stable(r_out - torch.mean(nr_out), t_real) +
            #                       BCE_stable(nr_out - torch.mean(r_out), t_fake)) / 2
            #     r_nr_loss /= len(adv_r_outputs)

            # 我的尝试----kl_loss
            # kl_loss = torch.tensor(0.).to(device)
            # if not len(adv_r_outputs) == 0 and len(adv_rec_outputs) == 0:
            #     for r_out, rec_out in adv_r_outputs, adv_rec_outputs:
            #         kl_loss += klLoss(F.log_softmax(r_out), F.softmax(rec_out))
            #         kl_loss += klLoss(F.log_softmax(rec_out), F.softmax(r_out))
            #     kl_loss /= len(adv_nr_outputs) * 2

            # kl_loss_1 = torch.tensor(0.).to(device)
            # if not len(adv_nr_outputs) == 0 and len(adv_r_outputs) == 0:
            #     for nr_out, r_out in adv_nr_outputs, adv_r_outputs:
            #         kl_loss_1 += klLoss(F.log_softmax(nr_out), F.softmax(r_out))
            #     kl_loss_1 /= len(adv_nr_outputs)



            kl_loss_2 = torch.tensor(0.).to(device)
            if not len(nat_outputs) == 0 and len(adv_outputs) == 0:
                for nat_output, adv_output in nat_outputs, adv_outputs:
                    kl_loss_2 += klLoss(F.log_softmax(nat_output), F.softmax(adv_output))
                    kl_loss_2 += klLoss(F.log_softmax(adv_output), F.softmax(nat_output))
                kl_loss_2 /= len(nat_outputs) * 2


            # loss = adv_cls_loss + 6 * kl_loss_2
            loss = adv_cls_loss + sep_loss + rec_loss + 6 * kl_loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_cls_losses += adv_cls_loss.item()
            # r_nr_losses += r_nr_loss.item()
            sep_losses += sep_loss.item()
            rec_losses += rec_loss.item()
            _, adv_predicted = adv_outputs.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            _tqdm.set_postfix(
                Adv_Loss='{:.3f}'.format(adv_cls_losses / (batch_idx + 1)),
                Sep_Loss='{:.3f}'.format(sep_losses / (batch_idx + 1)),
                Rec_Loss='{:.3f}'.format(rec_losses / (batch_idx + 1)),
                # R_NR_Loss='{:.3f}'.format(r_nr_losses / (batch_idx + 1)),
                Adv_Acc='{:.3f}%'.format(100. * adv_correct / total),
                KL_Loss='{:.3f}'.format(kl_loss_2 / (batch_idx + 1)),
            )
            _tqdm.update(inputs.shape[0])


best_adv_acc = 0.

def test(epoch):
    net.eval()
    ori_test_loss = 0
    adv_test_loss = 0
    ori_correct = 0
    adv_correct = 0
    total = 0

    global best_adv_acc

    with tqdm(total=(len(testset) - len(testset) % args.bs), dynamic_ncols=True) as _tqdm:
        _tqdm.set_description('{} (Test) Epoch: {}/{}'.format(args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = attack.perturb(inputs, targets, False)
            net.eval()

            ori_outputs, ori_r_outputs, ori_nr_outputs, ori_rec_outputs = net(inputs, is_eval=True)
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = net(adv_inputs, is_eval=True)

            ori_loss = criterion(ori_outputs, targets)
            ori_test_loss += ori_loss.item()
            _, ori_predicted = ori_outputs.max(1)
            ori_correct += ori_predicted.eq(targets).sum().item()

            adv_loss = criterion(adv_outputs, targets)
            adv_test_loss += adv_loss.item()
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()

            total += targets.size(0)

            _tqdm.set_postfix(
                Ori_Loss='{:.3f}'.format(ori_test_loss/(batch_idx+1)),
                Ori_Acc='{:.3f}%'.format(100.*ori_correct/total),
                Adv_Loss='{:.3f}'.format(adv_test_loss/(batch_idx+1)),
                Adv_Acc='{:.3f}%'.format(100.*adv_correct/total),
            )
            _tqdm.update(inputs.shape[0])

    if not os.path.exists('./weights/{}/{}/'.format(args.dataset, args.model)):
        os.makedirs('./weights/{}/{}/'.format(args.dataset, args.model))

    adv_acc = 100. * adv_correct / total

    if adv_acc > best_adv_acc:
        torch.save(net.state_dict(), './weights/{}/{}/best_{}.pth'.format(args.dataset, args.model, args.save_name))
        best_adv_acc = adv_acc

    torch.save(net.state_dict(), './weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name))


for epoch in range(start_epoch, args.epoch + 1):
    train(epoch)
    test(epoch)
