import os
from collections import OrderedDict
from PIL import Image
import torch
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from model import *
import pretrainedmodels
import model.resnet_cbam as resnet
from data_loader.ImageNet_datasets import ImageNetData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
#DATA_ROOT = '/nas/home/adas/GCT_data_val/1'
#RESULT_FILE = 'result.csv'

def test_and_generate_result_round2(epoch_num, model_name='resnet101', img_size=320, is_multi_gpu=False):
    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.53744068, 0.51462684, 0.52646497], [0.06178288, 0.05989952, 0.0618901])])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()

    if  'resnet152' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152(model_ft)
        del model_ft
    elif 'resnet152-r2' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2(model_ft)
        del model_ft
    elif 'resnet152-r2-2o' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = resnet152.MyResNet152_Round2_2out(model_ft)
        del model_ft
    elif 'resnet152-r2-2o-gmp' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = models.resnet152.MyResNet152_Round2_2out_GMP(model_ft)
        del model_ft
    elif 'resnet152-r2-hm-r1' == model_name.split('_')[0]:
        model_ft = models.resnet152(pretrained=True)
        my_model = models.resnet152.MyResNet152_Round2_HM_round1(model_ft)
        del model_ft
    elif 'resnet50' == model_name.split('_')[0]:
        model_ft = models.resnet50(pretrained=True)
        my_model = resnet50.MyResNet50(model_ft)
        del model_ft
    #elif 'resnet101' == model_name.split('_')[0]:
        #del model_ft
    elif 'densenet121' == model_name.split('_')[0]:
        model_ft = models.densenet121(pretrained=True)
        my_model = densenet121.MyDenseNet121(model_ft)
        del model_ft
    elif 'densenet169' == model_name.split('_')[0]:
        model_ft = models.densenet169(pretrained=True)
        my_model = densenet169.MyDenseNet169(model_ft)
        del model_ft
    elif 'densenet201' == model_name.split('_')[0]:
        model_ft = models.densenet201(pretrained=True)
        my_model = densenet201.MyDenseNet201(model_ft)
        del model_ft
    elif 'densenet161' == model_name.split('_')[0]:
        model_ft = models.densenet161(pretrained=True)
        my_model = densenet161.MyDenseNet161(model_ft)
        del model_ft
    elif 'ranet' == model_name.split('_')[0]:
        my_model = ranet.ResidualAttentionModel_92()
    elif 'senet154' == model_name.split('_')[0]:
        model_ft = pretrainedmodels.models.senet154(num_classes=1000, pretrained='imagenet')
        my_model = MySENet154(model_ft)
        del model_ft
    #else:
        #raise ModuleNotFoundError

    test_datasets   = datasets.ImageFolder('/nas/home/adas/GCT_data_test/',data_transform)
    test_dataloaders   = torch.utils.data.DataLoader(test_datasets, batch_size=16, shuffle=False, num_workers=8)
    #self.test_dataloader = test_dataloaders
    predictions = []
    actuals=[]
    probabilities=[]
    for i, (inputs, labels) in enumerate(test_dataloaders):            # Notice
            if is_use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.squeeze()
                #print(labels)
                if is_use_cuda:
                  my_model = resnet.resnet101()
                  my_model=torch.load('./checkpoint/resnet101/Models_epoch_5.ckpt')
                  my_model = my_model.cuda()
                  my_model.eval()
                  test_files_list = inputs
                  output = my_model(inputs)
                  output = F.softmax(output, dim=1).data.cpu()
                  prediction = output.argmax(dim=1, keepdim=True)
                  #predictio= torch.Tensor.cpu(prediction).detach().numpy()
                  predictions.extend(prediction)
                  actuals.extend(labels.view_as(prediction))
                  #actual= torch.Tensor.cpu(actuals).detach().numpy()[:,-1]
                  probabilities.extend(np.exp(output))
                  #print(probabilities)
                else:
                  labels = labels.squeeze()
    #actuals= torch.Tensor.cpu(actuals).detach().numpy()[:,-1]
    #print(probabilities)
    actual=[]
    for i in actuals:
     actual.append(i.item())
    prediction=[]
    for i in predictions:
     prediction.append(i.item())
    probabilitie=[]
    #for i in probabilities:
     #probabilitie=np.vstack(i.item())
    #print(probabilitie)
    ##print(actual)
    print('Confusion matrix:')
    print(confusion_matrix(actual, prediction))
    print('F1 score: %f' % f1_score(actual, prediction, average='micro'))
    print('Accuracy score: %f' % accuracy_score(actual, prediction))
    n_classes = max(actual)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
     #y_test_i = set(map(lambda x: 1 if x == i else 0, actual))
     #y_test_i=(actual ==  i)*1
     #print(y_test_i)
     #all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
     all_y_predict_proba = np.concatenate([all_y_predict_proba,probabilities [:, i]])
     fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
     roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])
    #fpr, tpr, _ = roc_curve(actual, probabilities)
    #roc_auc = auc(fpr, tpr)
    #print('ROC AuC: %f'% roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for digit=%d class' % which_class)
    plt.legend(loc="lower right")
    plt.show()
if __name__ == '__main__':
    test_and_generate_result_round2('9','resnet101',416,False)
