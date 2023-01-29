import torch


def get_recall(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = torch.nonzero(targets == indices)
    
    n_hits = len(hits)
    
    recall = float(n_hits)
    
    return recall


def get_ndcg(indices, targets):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = torch.nonzero(targets == indices)
    
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    ndcg = torch.sum((torch.log2(ranks + 1)).reciprocal()).data

    return ndcg.item()



def evaluate(outputs, target, k=[10, 20]):
    recalls = []
    ndcgs = []

    for k_ in k:
        _, indices = torch.topk(outputs, k_, dim=-1, largest=True)
        
        recall = get_recall(indices, target)
        ndcg = get_ndcg(indices, target)

        recalls.append(recall)
        ndcgs.append(ndcg)
    return recalls, ndcgs

