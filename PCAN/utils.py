
import pandas as pd
import os


def accuracy(outputs, targets, topk=(1,)):
    # compute the topk accuracy
    maxk = max(topk)
    batch_size = targets.size(0)

    # return the topk scores in every input
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()  # shape:(maxk,N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def record_info(info, filename):
    # result = (
    #     'Epoch {Epoch} '
    #     'Loss {Loss} '
    #     'epoch_recall@1 {epoch_recall1} '
    #     'epoch_recall@5 {epoch_recall5} '
    #     'epoch_recall@10 {epoch_recall10} '
    #     'epoch_NDCG {epoch_NDCG}'.format(Epoch=info['Epoch'], Loss=info['Loss'],
    #                                      epoch_recall1=info['epoch_recall1'],
    #                                      epoch_recall5=info['epoch_recall5'],
    #                                      epoch_recall10=info['epoch_recall10'],
    #                                      epoch_NDCG=info['epoch_NDCG']))
    print(info)

    df = pd.DataFrame.from_dict(info)
    column_names = ['Epoch', 'Loss', 'epoch_recall@1', 'epoch_recall@5', 'epoch_recall@10', 'epoch_NDCG']
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False,
                  index=False, columns=column_names)


