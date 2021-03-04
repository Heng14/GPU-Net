import torch
import numpy as np

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    #GT = GT == torch.max(GT)
    GT = GT > torch.max(GT)/2

    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    #print (SR.size(0), SR.size(1), SR.size(2), SR.size(3))
    #print (GT.size(0), GT.size(1), GT.size(2), GT.size(3))

    #corr = torch.sum(GT==GT)
    #print (corr)
    #raise
    acc = float(corr)/float(tensor_size)
    #print ('acc: ', acc)
    #raise
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).type(torch.uint8)
    #GT = GT == torch.max(GT)
    GT = (GT > torch.max(GT)/2).type(torch.uint8)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).type(torch.uint8)+(GT==1).type(torch.uint8))==2
    FN = ((SR==0).type(torch.uint8)+(GT==1).type(torch.uint8))==2
    #TP = (SR==1)&(GT==1)
    #FN = (SR==0)&(GT==1)

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    #print ('float(torch.sum(TP)): ', float(torch.sum(TP)))
    #raise
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = (SR > threshold).type(torch.uint8)
    #GT = GT == torch.max(GT)
    GT = (GT > torch.max(GT)/2).type(torch.uint8)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).type(torch.uint8)+(GT==0).type(torch.uint8))==2
    FP = ((SR==1).type(torch.uint8)+(GT==0).type(torch.uint8))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = (SR > threshold).type(torch.uint8)
    #GT = GT == torch.max(GT)
    GT = (GT > torch.max(GT)/2).type(torch.uint8)

    #SR = SR.data.cpu().numpy()[0,0,...]
    #GT = GT.data.cpu().numpy()[0,0,...]
    #print (np.sum(SR==1))
    #print (np.sum(((SR==1).astype(int)+(GT==1).astype(int))==2))
    #raise

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).type(torch.uint8)+(GT==1).type(torch.uint8))==2
    FP = ((SR==1).type(torch.uint8)+(GT==0).type(torch.uint8))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    #print ('PC: ', PC)
    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).type(torch.uint8)
    #GT = GT == torch.max(GT)
    GT = (GT > torch.max(GT)/2).type(torch.uint8)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).type(torch.uint8)
    #GT = GT == torch.max(GT)
    GT = (GT > torch.max(GT)/2).type(torch.uint8)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



