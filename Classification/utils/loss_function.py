# loss function using torch
import torch.nn as nn

# we have 5 labels
def loss_fn(outputs, targets):
    o1, o2, o3, o4 = outputs
    t1, t2, t3, t4= targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)

    return (l1 + l2 + l3 + l4) / 4