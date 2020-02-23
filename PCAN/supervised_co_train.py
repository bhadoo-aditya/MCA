import time
import argparse


import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import copy
import math
from supervised_co import Supervised_co
from utils import *
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class mydataset(data.Dataset):

    def __init__(self):
        super(mydataset, self).__init__()
        train_data = np.load('/data/pycharm_project_9/lunwencode20190225/data/train_data.npz')
        self.title = np.array(train_data["title"])
        print(self.title.shape,24)#10129,20
        self.ingredients = np.array(train_data["ingredients"])
        print(self.ingredients.shape,26)#10129,20
        self.instructions = np.array(train_data["instructions"])
        print(self.instructions.shape,28)#10129,20,20
        self.images = train_data["images"]
        print(self.images.shape,30)#(10129,5,1,4096)
        self.len = self.title.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # return img,title,ingredient,instruction
        return self.images[index], self.title[index], self.ingredients[index], self.instructions[index]


class mydataset_test(data.Dataset):

    def __init__(self):
        super(mydataset_test, self).__init__()
        train_data = np.load('/data/pycharm_project_9/lunwencode20190225/data/test_data.npz')
        self.title = np.array(train_data["title"])
        self.ingredients = np.array(train_data["ingredients"])
        self.instructions = np.array(train_data["instructions"])
        self.images = train_data["images"]

        self.len = self.title.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # return img,title,ingredient,instruction
        return self.images[index], self.title[index], self.ingredients[index], self.instructions[index]

class myloss(nn.Module):

    def __init__(self, delta):
        super(myloss, self).__init__()
        self.delta = delta

    def forward(self, x, y):
        # x,y both shape are (batch_size,feature_size)

        # l2 normalization
        x = x / torch.sqrt(torch.sum(torch.mul(x, x), dim=1, keepdim=True))
        y = y / torch.sqrt(torch.sum(torch.mul(y, y), dim=1, keepdim=True))
        # print('x',x[0])
        # print('y',y[0])

        similarity_m = torch.mm(x, y.t())  # similarity matrix
        # print(similarity_m.shape,77)#128*128
        cor_similarity = torch.mul(torch.ones(x.size(0), x.size(
            0)).to(device), torch.diag(similarity_m)).t()  # correct similarity
        # print('similiarty matirx',similarity_m)
        # print('correct matrix',cor_similarity)
        # print((similarity_m - cor_similarity + self.delta),82)
        zero_m = torch.zeros(x.size(0), x.size(0)).to(device)
        loss = torch.max(zero_m,
                         similarity_m - cor_similarity + self.delta) \
               + torch.max(zero_m,similarity_m.t() - cor_similarity + self.delta)

        for i in range(loss.size(0)):
            loss[i][i] = 0
        # print('loss',loss)
        loss = torch.sum(loss) / x.size(0)

        with torch.no_grad():
            recall=[]
            score = similarity_m - cor_similarity
            _, pred = score.topk(10, 1, True, True)
            pred = pred.t()  # shape:(10,N)

            target = torch.from_numpy(
                np.arange(0, x.size(0))).long().to(device).expand_as(pred)
            correct = pred.eq(target)
            for k in [1,5,10]:
                recall.append(torch.sum(correct[:k]))
            ndcg=0
            for i in range(x.size(0)):
                for j in range(10):
                    if(correct[j][i]==True):
                        ndcg+=math.log(2)/math.log(2+j)

            # print('pred',pred)
            # print('correct',target)


        return loss, recall,ndcg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=30167,
                        help='number of words in dictionary')
    parser.add_argument('--alpha', type=float, default=0.000004,
                        help='regularization strength')
    parser.add_argument('--delta', type=float, default=0.4,
                        help='margin for loss')
    parser.add_argument('--init_scale', type=float, default=0.0001,
                        help='scale for random uniform initializer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for a minibatch')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='dropout keep probability')
    parser.add_argument('--img_fea_dim', type=int, default=4096,
                        help='the dimension of image feature')
    parser.add_argument('--valDataDir', type=str, default='/data/pycharm_project_9/lunwencode20190225/data/test_data.npz',
                        help='test data directory')
    parser.add_argument('--trDataDir', type=str, default='/data/pycharm_project_9/lunwencode20190225/data/train_data.npz',
                        help='training data dir')
    parser.add_argument('--num_epochs', type=str, default=30,
                        help='number of ephochs')
    parser.add_argument('--word2vec_path', type=str, default='/data/pycharm_project_9/lunwencode20190225/data/vocab.bin',
                        help='word2vec path')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for learning')
    parser.add_argument('--joint_embedding_dim', type=int, default=1024,
                        help='dimension of joint embedding space')
    parser.add_argument('--embedding_size', type=int, default=300,
                        help='embedding size for word to vec')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='dimension of hidden vector')
    parser.add_argument('--max_sentence_length', type=int, default=20,
                        help='maximun length of sentence')
    parser.add_argument('--max_sentence_num', type=int, default=20,
                        help='maximun number of sentence')
    parser.add_argument('--grad_clip', type=int, default=5,
                        help='grad clip to prevent gradient explode')
    parser.add_argument('--evaluate_every', type=int, default=500,
                        help='evaluation frequency')
    parser.add_argument('--save_dir', type=str, default='model_supervised_co.pth',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=6000,
                        help='save_frequency')
    parser.add_argument('--resume',type=str,default=None,
                        help='directory to load checkpointed models')
    parser.add_argument('--alpha_a',type=int,default=0.2,help='concat coattention and supervised')
    args = parser.parse_args()
    if os.path.exists('record') == False:
        os.mkdir('record')

    net=Supervised_co(args).to(device)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint=torch.load(args.resume)
            net.load_state_dict(checkpoint)
            print(("=> loaded checkpoint {}").format(args.resume))
        else:
            print("no checkpoint found at {}".format(args.resume))
    print(net)
    param_num = 0
    for para in net.parameters():
        shape = para.shape
        num = 1
        for i in range(len(shape)):
            num*=shape[i]
        param_num+=num

    print('total parameters:',param_num)

    train_dataset = mydataset()
    test_dataset = mydataset_test()

    dataloaders_dict = {
        'train': data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        ),
        'val': data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
    }

    net = net.to(device)
    # 默认的是sgd
    optimizer=torch.optim.SGD(net.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=0.001)
    criterion=myloss(args.delta)
    net, best_recall = train_model(
        net, dataloaders_dict, criterion, optimizer,args.num_epochs)
    torch.save(net.state_dict(), args.save_dir)



# def savehis(val_hist, train_hist, path, best_recall, num_epochs):
#     plt.cla()
#     plt.title("Recall vs. Number of Training Epochs")
#     plt.xlabel('Training Epochs')
#     plt.ylabel('Recall')
#     plt.plot(range(1, num_epochs + 1), val_hist,0.0010.001criterion
#              label='val %.4f' % best_recall)
#     plt.plot(range(1, num_epochs + 1), train_hist, label='training')
#     plt.ylim(0, 1.0)
#     plt.xticks(np.arange(0, num_epochs + 1, 20))
#     plt.legend()
#     plt.savefig(path)

def evaluate(model,dataloaders,criterion):
    model.eval()
    running_loss = 0.0
    running_recall1 = 0.0
    running_recall5 = 0.0
    running_recall10 = 0.0
    running_ndcg = 0.0
    running_sim=0.0
    with torch.no_grad():
        for img, title, ingredient, instruction in dataloaders:
            # print(img.shape, 220)
            # img = torch.mean(img, dim=1, keepdim=True).squeeze().to(device)
            # print(img.shape, 221)
            img = img.to(device)
            title = title.to(device)
            ingredient = ingredient.to(device)
            instruction = instruction.to(device)
            inputs = (img, title, ingredient, instruction)
            img_feature, recipe_feature = model(inputs)
            loss, recall, ndcg = criterion(recipe_feature, img_feature)

            x = img_feature.detach().cpu()
            y = recipe_feature.detach().cpu()
            x = x / torch.sqrt(torch.sum(torch.mul(x, x), dim=1, keepdim=True))
            y = y / torch.sqrt(torch.sum(torch.mul(y, y), dim=1, keepdim=True))

            running_sim += torch.mul(x, y).sum().item()
            running_loss += loss.item() * img_feature.size(0)
            running_recall1 += recall[0].item()
            running_recall5 += recall[1].item()
            running_recall10 += recall[2].item()
            running_ndcg += ndcg

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_recall1 = 1.0 * running_recall1 / len(dataloaders.dataset)
    epoch_recall5 = 1.0 * running_recall5 / len(dataloaders.dataset)
    epoch_recall10 = 1.0 * running_recall10 / len(dataloaders.dataset)
    epoch_ndcg = 1.0 * running_ndcg / len(dataloaders.dataset)
    epoch_sim = 1.0 * running_sim / len(dataloaders.dataset)

    print('Loss: {:.4f} recall@1: {:.4f} recall@5: {:.4f} recall@10: {:.4f} NDCG@10: {:.4f} similarity: {:.4f}'.format(
        epoch_loss, epoch_recall1, epoch_recall5, epoch_recall10, epoch_ndcg, epoch_sim))

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_recall1_history = []
    train_recall1_history = []
    val_recall5_history = []
    train_recall5_history = []
    val_recall10_history = []
    train_recall10_history = []
    val_ndcg_history = []
    train_ndcg_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall10 = 0.0
    best_recall5 = 0.0
    best_recall1 = 0.0
    best_ndcg = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        if epoch == 50:
            optimizer.param_groups[0]['lr'] /= 10

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_recall1 = 0.0
            running_recall5 = 0.0
            running_recall10 = 0.0
            running_ndcg = 0.0

            for img, title, ingredient, instruction in dataloaders[phase]:
                # print(img.shape,334) #([128,1,4096])
                # print(img.shape, 312)
                img = img.squeeze().to(device)
                title = title.to(device)
                ingredient = ingredient.to(device)
                instruction = instruction.to(device)
                inputs = (img, title, ingredient, instruction)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    img_feature, recipe_feature = model(inputs)
                    # print(img_feature.shape,350)[128, 1024]
                    # print(recipe_feature.shape,351)[128, 1024]
                    loss, recall, ndcg = criterion(recipe_feature, img_feature)
                    # print('loss: {}  recall: {}'.format(loss,recall/img.size(0)))
                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * img_feature.size(0)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_recall1 = 1.0 * running_recall1 / len(dataloaders[phase].dataset)
            epoch_recall5 = 1.0 * running_recall5 / len(dataloaders[phase].dataset)
            epoch_recall10 = 1.0 * running_recall10 / len(dataloaders[phase].dataset)
            epoch_ndcg = 1.0 * running_ndcg / len(dataloaders[phase].dataset)

            info = {'Epoch': [epoch + 1],
                    'Loss': [epoch_loss],
                    'epoch_recall@1': [epoch_recall1],
                    'epoch_recall@5': [epoch_recall5],
                    'epoch_recall@10': [epoch_recall10],
                    'epoch_NDCG': [epoch_ndcg],
                    }
            record_info(info, 'record/Super_co' + phase + '.csv')
            # print('{} Loss: {:.4f} recall@1: {:.4f} recall@5: {:.4f} recall@10: {:.4f} NDCG: {:.4f}'.format(
            #    phase, epoch_loss, epoch_recall1, epoch_recall5, epoch_recall10, epoch_ndcg))

            if phase == 'val' and epoch_recall10 > best_recall10:
                best_recall10 = epoch_recall10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_recall10_history.append(epoch_recall10)
                val_recall5_history.append(epoch_recall5)
                val_recall1_history.append(epoch_recall1)
                val_ndcg_history.append(epoch_ndcg)
                best_recall5 = max(best_recall5, epoch_recall5)
                best_recall1 = max(best_recall1, epoch_recall1)
                best_ndcg = max(best_ndcg, epoch_ndcg)
            else:
                train_recall10_history.append(epoch_recall10)
                train_recall5_history.append(epoch_recall5)
                train_recall1_history.append(epoch_recall1)
                train_ndcg_history.append(epoch_ndcg)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val recall@10 : {:4f} recall@5 : {:4f} recall@1 : {:4f} ndcg : {:4f}'. \
          format(best_recall10, best_recall5, best_recall1, best_ndcg))

    model.load_state_dict(best_model_wts)
    return model, best_recall10

if __name__ == '__main__':
    main()
