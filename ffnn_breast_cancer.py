"""Reproducible FFNN breast-cancer assignment. Run: python ffnn_breast_cancer.py"""
from pathlib import Path
import copy, csv, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=42; OUT=Path(__file__).resolve().parent; EPOCHS=180; PATIENCE=10
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
# UCI WDBC dataset, bundled unchanged in scikit-learn. Its raw labels are 0=malignant, 1=benign.
data=load_breast_cancer()
X=data.data[:,:20].astype('float32') # documented first 20 clinical measurements
Y=(data.target==0).astype('float32') # positive class = malignant
indices=np.arange(len(Y))
tr_idx,other_idx=train_test_split(indices,test_size=.4,stratify=Y,random_state=SEED)
val_idx,te_idx=train_test_split(other_idx,test_size=.5,stratify=Y[other_idx],random_state=SEED)
scaler=StandardScaler().fit(X[tr_idx]); X=scaler.transform(X).astype('float32')
xt=torch.tensor(X); yt=torch.tensor(Y).unsqueeze(1)
train_ds=TensorDataset(xt[tr_idx],yt[tr_idx]); train_loader=DataLoader(train_ds,batch_size=32,shuffle=True)

class FFNN(nn.Module):
    def __init__(self,batch_norm=False,dropout=0.):
        super().__init__(); blocks=[]; width=20
        for _ in range(5):
            blocks.append(nn.Linear(width,10))
            if batch_norm: blocks.append(nn.BatchNorm1d(10))
            blocks.append(nn.ReLU())
            if dropout: blocks.append(nn.Dropout(dropout))
            width=10
        blocks.extend([nn.Linear(10,1),nn.Sigmoid()]); self.net=nn.Sequential(*blocks)
    def forward(self,x): return self.net(x)

def evaluate(model,idx,criterion):
    model.eval()
    with torch.no_grad():
        pred=model(xt[idx]); loss=criterion(pred,yt[idx]).item()
        acc=((pred>=.5)==yt[idx]).float().mean().item()
    return loss,acc

def fit(name,optimizer_name='Adam',loss_name='BCE',batch_norm=False,dropout=0.,early_stop=True):
    # Same initialization across experiments for a controlled comparison.
    torch.manual_seed(SEED); model=FFNN(batch_norm,dropout)
    criterion=nn.BCELoss() if loss_name=='BCE' else nn.MSELoss()
    optimizer=(torch.optim.Adam(model.parameters(),lr=.001) if optimizer_name=='Adam'
               else torch.optim.SGD(model.parameters(),lr=.02,momentum=.9))
    hist={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best=float('inf'); best_epoch=0; best_weights=None; waiting=0
    for epoch in range(1,EPOCHS+1):
        model.train()
        for xb,yb in train_loader:
            optimizer.zero_grad(); loss=criterion(model(xb),yb); loss.backward(); optimizer.step()
        tl,ta=evaluate(model,tr_idx,criterion); vl,va=evaluate(model,val_idx,criterion)
        for key,value in zip(hist,(tl,vl,ta,va)): hist[key].append(value)
        if vl<best-1e-4:
            best=vl; best_epoch=epoch; best_weights=copy.deepcopy(model.state_dict()); waiting=0
        else: waiting+=1
        if early_stop and waiting>=PATIENCE: break
    if early_stop: model.load_state_dict(best_weights)
    train_loss,train_acc=evaluate(model,tr_idx,criterion)
    val_loss,val_acc=evaluate(model,val_idx,criterion)
    test_loss,test_acc=evaluate(model,te_idx,criterion)
    print(f'{name:22s} epochs={len(hist["val_loss"]):3d} best_epoch={best_epoch:3d} train={train_loss:.4f} val={val_loss:.4f} testacc={test_acc:.4f}',flush=True)
    return {'name':name,'model':model,'hist':hist,'epochs':len(hist['val_loss']),'best_epoch':best_epoch,
            'train_loss':train_loss,'val_loss':val_loss,'test_accuracy':test_acc,'train_accuracy':train_acc,'val_accuracy':val_acc}

baseline=fit('Baseline (no reg)',early_stop=False)
bn=fit('BatchNorm',batch_norm=True,early_stop=False)
do=fit('Dropout p=0.2',dropout=.2,early_stop=False)
full=fit('BN+Dropout+ES',batch_norm=True,dropout=.2,early_stop=True)
comparisons=[fit(f'{o} + {l}',o,l,batch_norm=True,dropout=.2,early_stop=True)
             for o,l in [('SGD','MSE'),('SGD','BCE'),('Adam','MSE'),('Adam','BCE')]]
# Choose based on validation accuracy, then malignant recall on validation, then lower validation BCE.
def val_recall(r):
    r['model'].eval()
    with torch.no_grad(): p=r['model'](xt[val_idx]).numpy().ravel()
    return recall_score(Y[val_idx],p>=.5,zero_division=0)
best=max(comparisons,key=lambda r:(r['val_accuracy'],val_recall(r),-r['val_loss']))
best['model'].eval()
with torch.no_grad(): prob=best['model'](xt[te_idx]).numpy().ravel()
pred=(prob>=.5).astype(int); truth=Y[te_idx].astype(int)
tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
metrics={'model':best['name'],'accuracy':accuracy_score(truth,pred),'precision':precision_score(truth,pred,zero_division=0),
'recall_sensitivity':recall_score(truth,pred,zero_division=0),'specificity':tn/(tn+fp),
'f1':f1_score(truth,pred,zero_division=0),'auc':roc_auc_score(truth,prob),
'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
with open(OUT/'comparison_results.csv','w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['Configuration','Epochs to converge','Best epoch','Final training loss','Final validation loss','Final test accuracy'])
    for r in comparisons: writer.writerow([r['name'],r['epochs'],r['best_epoch'],round(r['train_loss'],6),round(r['val_loss'],6),round(r['test_accuracy'],6)])

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
for r in [baseline,bn,do,full]: axes[0,0].plot(r['hist']['val_loss'],label=r['name'])
axes[0,0].set(title='Regularization comparison: validation loss',xlabel='Epoch',ylabel='Validation loss');axes[0,0].legend(fontsize=8)
for key,label in [('train_loss','Training'),('val_loss','Validation')]: axes[0,1].plot(best['hist'][key],label=label)
axes[0,1].set(title=f'Best model loss: {best["name"]}',xlabel='Epoch',ylabel='Loss');axes[0,1].legend()
for key,label in [('train_acc','Training'),('val_acc','Validation')]: axes[1,0].plot(best['hist'][key],label=label)
axes[1,0].set(title='Training vs validation accuracy',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.03));axes[1,0].legend()
for r in comparisons: axes[1,1].plot(r['hist']['val_loss'],label=r['name'])
axes[1,1].set(title='Optimizer and loss: validation loss (not cross-loss comparable)',xlabel='Epoch',ylabel='Loss');axes[1,1].legend(fontsize=8)
fig.savefig(OUT/'training_dashboard.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
cm=np.array([[tn,fp],[fn,tp]]);axes[0].imshow(cm,cmap='Blues')
for (i,j),v in np.ndenumerate(cm): axes[0].text(j,i,str(v),ha='center',va='center',fontsize=18,color='white' if v>cm.max()/2 else 'black')
axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['Benign','Malignant'],yticklabels=['Benign','Malignant'],xlabel='Predicted',ylabel='Actual',title=f'Confusion matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp})')
fpr,tpr,_=roc_curve(truth,prob);axes[1].plot(fpr,tpr,label=f'AUC = {metrics["auc"]:.4f}',lw=2);axes[1].plot([0,1],[0,1],'--',color='gray')
axes[1].set(title='ROC curve (Malignant = positive)',xlabel='False positive rate (1 - specificity)',ylabel='True positive rate (sensitivity)',xlim=(0,1),ylim=(0,1.02));axes[1].legend()
fig.savefig(OUT/'clinical_dashboard.png',dpi=170);plt.close(fig)

with open(OUT/'analysis_report.md','w') as f:
    f.write(f'''# FFNN Breast Cancer Classification — Experimental Report

## Objective and dataset
Predict benign versus malignant diagnosis from 20 selected clinical measurements. Source: UCI Breast Cancer Wisconsin (Diagnostic), dataset 17; 569 samples, 30 original features. This experiment selects the **first 20 features in the dataset's published feature order** (10 mean and 10 standard-error features). scikit-learn ships a copy of the WDBC dataset; no synthetic observations are used. **Malignant = 1**, Benign = 0. Stratified 60%/20%/20% train/validation/test split; StandardScaler is fit to training data only.

## Architecture and mathematical mapping
20 inputs → 10 ReLU → 10 ReLU → 10 ReLU → 10 ReLU → 10 ReLU → 1 sigmoid output. PyTorch weights are stored (out_features, in_features): (10,20), (10,10) × 4, (1,10); biases (10,) × 5 and (1,). Baseline trainable parameters = (20×10+10) + 4×(10×10+10) + (10×1+1) = **661**. BatchNorm adds 20 learnable parameters per hidden layer (scale and shift), so BN variants have **761** learnable parameters (running statistics are buffers). Sigmoid outputs P(malignant | features). Binary cross-entropy is appropriate for probabilistic binary classification; MSE is a comparative alternative.

## Training protocol
Five hidden layers, 10 units each; BN is placed between linear and ReLU, dropout p=0.2 after ReLU. Adam lr=0.001, SGD lr=0.02 momentum=0.9; batch size 32; up to {EPOCHS} epochs; early stopping monitors validation loss with patience={PATIENCE}, improvement tolerance 0.0001, and restores best weights. Fixed random seed {SEED}. The test split is held out until model selection by validation performance is finished.

## Optimizer/loss comparison
| Configuration | Epochs run | Best epoch | Final training loss | Final validation loss | Test accuracy |
|---|---:|---:|---:|---:|---:|
''')
    for r in comparisons: f.write(f'| {r["name"]} | {r["epochs"]} | {r["best_epoch"]} | {r["train_loss"]:.4f} | {r["val_loss"]:.4f} | {r["test_accuracy"]:.2%} |\n')
    f.write(f'''
**Selection:** {best['name']} had the strongest validation accuracy (with validation malignant recall and validation loss as tie-breakers). The different raw BCE and MSE losses should not be compared numerically as if on the same scale. Epochs run means the stopping epoch, not necessarily the best epoch.

## Held-out test evaluation — {best['name']}
- Accuracy: **{metrics['accuracy']:.2%}**
- Precision (malignant): **{metrics['precision']:.2%}** = TP/(TP+FP)
- Recall / sensitivity (malignant): **{metrics['recall_sensitivity']:.2%}** = TP/(TP+FN)
- Specificity (benign): **{metrics['specificity']:.2%}** = TN/(TN+FP)
- F1: **{metrics['f1']:.4f}** = harmonic mean of precision and recall
- ROC-AUC: **{metrics['auc']:.4f}**
- Confusion matrix: **TN={tn}, FP={fp}, FN={fn}, TP={tp}**

## Interpretation
Batch normalization can smooth optimization by normalizing hidden activations, though results are dataset-dependent. Dropout introduces training-time noise to reduce dependence on individual neurons, so its training loss can be higher than validation loss. Early stopping limits training after validation loss stalls and restores the best checkpoint, reducing the risk of keeping overfitted final weights. Examine `training_dashboard.png` for observed effects; regularization is not guaranteed to improve every run. Adam often converges efficiently with adaptive learning rates; SGD with momentum may require different tuning. BCE is better aligned with binary probability estimation than MSE, though neither is guaranteed to win on every finite dataset.

False negatives are malignant patients predicted benign; these are especially concerning because they may delay follow-up. Sensitivity should be prioritized alongside specificity and precision. Accuracy alone can conceal failures on the less frequent malignant class. ROC-AUC summarizes ranking quality across thresholds, while a single accuracy score uses one cutoff. However, even high ROC-AUC does **not** guarantee clinical safety: threshold-specific sensitivity, calibration, external validation, and clinical evaluation are still required. This model is educational, **not** a diagnostic device.

## Figures and reproducibility
- `training_dashboard.png`: regularization, loss, and accuracy comparisons.
- `clinical_dashboard.png`: confusion matrix and ROC curve.
- `comparison_results.csv`: measured experiment results.
- `ffnn_breast_cancer.py`: executable source code.

## References
1. UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. scikit-learn, `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. PyTorch documentation: https://pytorch.org/docs/stable/index.html
''')
print('BEST:',metrics)
