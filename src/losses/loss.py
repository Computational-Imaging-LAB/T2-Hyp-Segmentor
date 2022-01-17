

import torch



def dice_loss(output, seg,n_classes=1,class_weights=[1]):
    smooth = 1.
    loss = 0.
    seg[seg==255]=1
    #output=(output-torch.min(output))/(torch.max(output)-torch.min(output))
    class_weights = torch.FloatTensor(class_weights)
    for c in range(n_classes):
           iflat = torch.flatten(output[:, c ])
           tflat = torch.flatten(seg[:, c])
           intersection = (iflat * tflat).sum()
           TP=intersection
           FP=iflat.sum()-intersection
           FN=tflat.sum()-intersection
           w = class_weights[c]
           loss += w*(1 - ((2. *TP) /
                             (2*(FP)+2*TP + FN + smooth)))
    return loss