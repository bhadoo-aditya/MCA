import os
import time
import argparse

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
print(device)


class mydataset(data.Dataset):

    def __init__(self):
        super(mydataset, self).__init__()
        train_data = np.load('/data/pycharm_project_9/lunwencode20190225/data/train_data.npz')
        self.title = np.array(train_data["title"])
        self.ingredients = np.array(train_data["ingredients"])
        self.instructions = np.array(train_data["instructions"])
        self.images = train_data["images"]
        self.len = self.title.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # return img,title,ingredient,instruction
        print(self.images.shape,36)
        img = np.mean(self.images[index, :], axis=0).squeeze(0)
        return img, self.title[index], self.ingredients[index], self.instructions[index]


# class mydataset(data.Dataset):
#
#     def __init__(self):
#         super(mydataset, self).__init__()
#         raw_data = np.load('/data/data/rf_data/raw_data.npz')
#         new_data = np.load('/data/project/crr/data_process/rf_data/new_data.npz')
#
#         titles = np.concatenate((raw_data['title'], new_data['title']), 0)
#
#         ingredients = np.concatenate((raw_data['ingredients'], new_data['ingredients']), 0)
#         instructions = np.concatenate((raw_data['instructions'], new_data['instructions']), 0)
#         images = np.concatenate((raw_data['images'], new_data['images']), 0)
#         start_end = np.concatenate((raw_data['start_end'], new_data['start_end']), 0)
#
#         self.title = titles
#         self.ingredients = ingredients
#         self.instructions = instructions
#         self.images = images
#         self.start_end = start_end
#         self.len = self.title.shape[0]
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         # return img,title,ingredient,instruction
#         start = int(self.start_end[index][0])
#         end = int(self.start_end[index][1])
#         # print(type(self.images),self.images.shape,start,end)
#         img = np.mean(self.images[index, start:end+1], axis=0)
#         return img, self.title[index], self.ingredients[index], self.instructions[index]


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
        # img = np.mean(self.images[index, :], axis=0).squeeze(0)
        # print(img.shape)
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

        similarity_m = torch.mm(x, y.t())  # similarity matrix
        cor_similarity = torch.mul(torch.ones(x.size(0), x.size(
            0)).to(device), torch.diag(similarity_m)).t()  # correct similarity

        zero_m = torch.zeros(x.size(0), x.size(0)).to(device)
        loss = torch.max(zero_m, similarity_m - cor_similarity + self.delta) + \
               torch.max(zero_m, similarity_m.t() - cor_similarity + self.delta)

        for i in range(loss.size(0)):
            loss[i][i] = 0

        loss = torch.sum(loss) / x.size(0)

        with torch.no_grad():
            recall = []
            score = similarity_m - cor_similarity
            _, pred = score.topk(10, 1, True, True)
            pred = pred.t()  # shape:(10,N)

            target = torch.from_numpy(
                np.arange(0, x.size(0))).long().to(device).expand_as(pred)
            correct = pred.eq(target)
            for k in [1, 5, 10]:
                recall.append(torch.sum(correct[:k]))

            ndcg=[]
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            for i in range(x.size(0)):
                for j in range(1):
                    if (correct[j][i] == True):
                        ndcg1 += math.log(2) / math.log(2 + j)
            ndcg.append(ndcg1)

            for i in range(x.size(0)):
                for j in range(5):
                    if (correct[j][i] == True):
                        ndcg5 += math.log(2) / math.log(2 + j)
            ndcg.append(ndcg5)

            for i in range(x.size(0)):
                for j in range(10):
                    if (correct[j][i] == True):
                        ndcg10 += math.log(2) / math.log(2 + j)
            ndcg.append(ndcg10)

        return loss, recall, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=30167,
                        help='number of words in dictionary')
    parser.add_argument('--alpha', type=float, default=0.000004,
                        help='regularization strength')
    parser.add_argument('--delta', type=float, default=0.3,
                        help='margin for loss')
    parser.add_argument('--init_scale', type=float, default=0.001,
                        help='scale for random uniform initializer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for a minibatch')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='dropout keep probability')
    parser.add_argument('--img_fea_dim', type=int, default=4096,
                        help='the dimension of image feature')
    parser.add_argument('--num_epochs', type=str, default=30,
                        help='number of ephochs')
    parser.add_argument('--word2vec_path', type=str, default='/data/pycharm_project_9/lunwencode20190225/data/vocab.bin',
                        help='word2vec path')
    parser.add_argument('--learning_rate', type=float, default=0.01,
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
    parser.add_argument('--evaluate_every', type=int, default=50,
                        help='evaluation frequency')
    parser.add_argument('--save_dir', type=str, default='model_dietw.pth',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=600,
                        help='save_frequency')
    parser.add_argument('--resume', type=str, default='/data/pycharm_project_9/lunwencode20190225/super_co_lunwen/model_supervised_co.pth',
                        help='directory to load checkpointed models')
    parser.add_argument('--evaluate', default=True, action='store_true')
    args = parser.parse_args()
    if os.path.exists('record') == False:
        os.mkdir('record')

    net = Supervised_co(args).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            net.load_state_dict(checkpoint)
            print(("=> loaded checkpoint {} "). \
                  format(args.resume))
        else:
            print("=> no checkpoint found at {}".format(args.resume))

    print(net)
    param_num = 0
    for name, para in net.named_parameters():
        shape = para.shape
        num = 1
        for i in range(len(shape)):
            num *= shape[i]
        param_num += num
    print('total prarameters:', param_num)

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

    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.001)
    criterion = myloss(args.delta)

    if args.evaluate:
        evaluate(net, dataloaders_dict['val'], criterion)
        return

    # net, best_recall = train_model(
    #     net, dataloaders_dict, criterion, optimizer, args.num_epochs)
    torch.save(net.state_dict(), args.save_dir)


# def savehis(val_hist, train_hist, path, best_recall, num_epochs):
#     plt.cla()
#     plt.title("Accuracy vs. Number of Training Epochs")
#     plt.xlabel('Training Epochs')
#     plt.ylabel('Accuracy')
#     plt.plot(range(1, num_epochs + 1), val_hist,
#              label='val %.4f' % best_recall)
#     plt.plot(range(1, num_epochs + 1), train_hist, label='training')
#     plt.ylim(0, 1.0)
#     plt.xticks(np.arange(0, num_epochs + 1, 20))
#     plt.legend()
#     plt.savefig(path)


def evaluate(model,dataloaders,criterion):
    model.eval()
    running_loss = 0.0
    img2text_running_recall1 = 0.0
    img2text_running_recall5 = 0.0
    img2text_running_recall10 = 0.0

    img2text_running_ndcg1 = 0.0
    img2text_running_ndcg5 = 0.0
    img2text_running_ndcg10 = 0.0

    text2img_running_recall1 = 0.0
    text2img_running_recall5 = 0.0
    text2img_running_recall10 = 0.0

    text2img_running_ndcg1 = 0.0
    text2img_running_ndcg5 = 0.0
    text2img_running_ndcg10 = 0.0
    running_sim=0.0
    with torch.no_grad():
        for img, title, ingredient, instruction in dataloaders:

            img = img.to(device)
            title = title.to(device)
            ingredient = ingredient.to(device)
            instruction = instruction.to(device)
            inputs = (img, title, ingredient, instruction)
            img_feature, recipe_feature = model(inputs)
            loss, recall, ndcg = criterion(img_feature, recipe_feature)
            img2text_running_recall1 += recall[0].item()
            img2text_running_recall5 += recall[1].item()
            img2text_running_recall10 += recall[2].item()

            img2text_running_ndcg1 += ndcg[0]
            img2text_running_ndcg5 += ndcg[1]
            img2text_running_ndcg10 += ndcg[2]

            loss, recall, ndcg1 = criterion(recipe_feature, img_feature)
            text2img_running_recall1 += recall[0].item()
            text2img_running_recall5 += recall[1].item()
            text2img_running_recall10 += recall[2].item()
            text2img_running_ndcg1 += ndcg1[0]
            text2img_running_ndcg5 += ndcg1[1]
            text2img_running_ndcg10 += ndcg1[2]

            x = img_feature.detach().cpu()
            y = recipe_feature.detach().cpu()
            x = x / torch.sqrt(torch.sum(torch.mul(x, x), dim=1, keepdim=True))
            y = y / torch.sqrt(torch.sum(torch.mul(y, y), dim=1, keepdim=True))

            running_sim += torch.mul(x, y).sum().item()
            running_loss += loss.item() * img_feature.size(0)


    epoch_loss = running_loss / len(dataloaders.dataset)
    img2text_epoch_recall1 = 1.0 * img2text_running_recall1 / len(dataloaders.dataset)
    img2text_epoch_recall5 = 1.0 * img2text_running_recall5 / len(dataloaders.dataset)
    img2text_epoch_recall10 = 1.0 * img2text_running_recall10 / len(dataloaders.dataset)
    img2text_epoch_ndcg1 = 1.0 * img2text_running_ndcg1 / len(dataloaders.dataset)
    img2text_epoch_ndcg5 = 1.0 * img2text_running_ndcg5 / len(dataloaders.dataset)
    img2text_epoch_ndcg10 = 1.0 * img2text_running_ndcg10 / len(dataloaders.dataset)

    text2img_epoch_recall1 = 1.0 * text2img_running_recall1 / len(dataloaders.dataset)
    text2img_epoch_recall5 = 1.0 * text2img_running_recall5 / len(dataloaders.dataset)
    text2img_epoch_recall10 = 1.0 * text2img_running_recall10 / len(dataloaders.dataset)
    text2img_epoch_ndcg1 = 1.0 * text2img_running_ndcg1 / len(dataloaders.dataset)
    text2img_epoch_ndcg5 = 1.0 * text2img_running_ndcg5 / len(dataloaders.dataset)
    text2img_epoch_ndcg10 = 1.0 * text2img_running_ndcg10 / len(dataloaders.dataset)
    epoch_sim = 1.0 * running_sim / len(dataloaders.dataset)

    print('Loss: {:.4f} \n \
           img2text_recall@1: {:.4f} img2text_recall@5: {:.4f} img2text_recall@10: {:.4f} img2text_NDCG1:{:.4f},img2text_NDCG5:{:.4f},img2text_NDCG10:{:.4f}, \n \
           text2img_recall@1: {:.4f} text2img_recall@5: {:.4f} text2img_recall@10: {:.4f} text2img_NDCG1:{:.4f},text2img_NDCG5:{:.4f},text2img_NDCG10:{:.4f},\n \
           similarity: {:.4f}'.format\
              (epoch_loss, \
               img2text_epoch_recall1, img2text_epoch_recall5, img2text_epoch_recall10, img2text_epoch_ndcg1, img2text_epoch_ndcg5, img2text_epoch_ndcg10, \
               text2img_epoch_recall1, text2img_epoch_recall5, text2img_epoch_recall10, text2img_epoch_ndcg1, text2img_epoch_ndcg5, text2img_epoch_ndcg10, \
               epoch_sim))


# def train_model(model, dataloaders, criterion, optimizer, num_epochs):
#     since = time.time()
#     val_recall1_history = []
#     train_recall1_history = []
#     val_recall5_history = []
#     train_recall5_history = []
#     val_recall10_history = []
#     train_recall10_history = []
#     val_ndcg_history = []
#     train_ndcg_history = []
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_recall10 = 0.0
#     best_recall5 = 0.0
#     best_recall1 = 0.0
#     best_ndcg = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
#         print('-' * 10)
#
#         if epoch == 50:
#             optimizer.param_groups[0]['lr'] /= 10
#
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_recall1 = 0.0
#             running_recall5 = 0.0
#             running_recall10 = 0.0
#             running_ndcg = 0.0
#
#             for img, title, ingredient, instruction in dataloaders[phase]:
#
#                 img = img.to(device)
#                 title = title.to(device)
#                 ingredient = ingredient.to(device)
#                 instruction = instruction.to(device)
#                 inputs = (img, title, ingredient, instruction)
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#
#                     img_feature, recipe_feature = model(inputs)
#
#                     loss, recall, ndcg = criterion(recipe_feature, img_feature)
#                     # print('loss: {}  recall: {}'.format(loss,recall/img.size(0)))
#                     if (phase == 'train'):
#                         loss.backward()
#                         optimizer.step()
#                 running_loss += loss.item() * img_feature.size(0)
#                 running_recall1 += recall[0].item()
#                 running_recall5 += recall[1].item()
#                 running_recall10 += recall[2].item()
#                 running_ndcg += ndcg
#
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_recall1 = 1.0 * running_recall1 / len(dataloaders[phase].dataset)
#             epoch_recall5 = 1.0 * running_recall5 / len(dataloaders[phase].dataset)
#             epoch_recall10 = 1.0 * running_recall10 / len(dataloaders[phase].dataset)
#             epoch_ndcg = 1.0 * running_ndcg / len(dataloaders[phase].dataset)
#
#             info = {'Epoch': [epoch + 1],
#                     'Loss': [epoch_loss],
#                     'epoch_recall@1': [epoch_recall1],
#                     'epoch_recall@5': [epoch_recall5],
#                     'epoch_recall@10': [epoch_recall10],
#                     'epoch_NDCG': [epoch_ndcg],
#                     }
#             record_info(info, 'record/NoAttenstion' + phase + '.csv')
#             #print('{} Loss: {:.4f} recall@1: {:.4f} recall@5: {:.4f} recall@10: {:.4f} NDCG: {:.4f}'.format(
#             #    phase, epoch_loss, epoch_recall1, epoch_recall5, epoch_recall10, epoch_ndcg))
#
#             if phase == 'val' and epoch_recall10 > best_recall10:
#                 best_recall10 = epoch_recall10
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'val':
#                 val_recall10_history.append(epoch_recall10)
#                 val_recall5_history.append(epoch_recall5)
#                 val_recall1_history.append(epoch_recall1)
#                 val_ndcg_history.append(epoch_ndcg)
#                 best_recall5 = max(best_recall5, epoch_recall5)
#                 best_recall1 = max(best_recall1, epoch_recall1)
#                 best_ndcg = max(best_ndcg, epoch_ndcg)
#             else:
#                 train_recall10_history.append(epoch_recall10)
#                 train_recall5_history.append(epoch_recall5)
#                 train_recall1_history.append(epoch_recall1)
#                 train_ndcg_history.append(epoch_ndcg)
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val recall@10 : {:4f} recall@5 : {:4f} recall@1 : {:4f} ndcg : {:4f}'.\
#           format(best_recall10, best_recall5, best_recall1, best_ndcg))
#
#     model.load_state_dict(best_model_wts)
#     # savehis(val_recall10_history, train_recall10_history, 'record/recall_10.png',
#     #         best_recall10, num_epochs=num_epochs)
#     # savehis(val_recall5_history, train_recall5_history, 'record/recall_5.png',
#     #         best_recall5, num_epochs=num_epochs)
#     # savehis(val_recall1_history, train_recall1_history, 'record/recall_1.png',
#     #         best_recall1, num_epochs=num_epochs)
#     # savehis(val_ndcg_history, train_ndcg_history, 'record/ndcg.png',
#     #         best_ndcg, num_epochs=num_epochs)
#     return model, best_recall10


if __name__ == '__main__':
    main()
