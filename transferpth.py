import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.frcnn import FasterRCNN
from trainer import FasterRCNNTrainer
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import LossHistory, weights_init
prepth = 'model_data/voc_weights_resnet.pth'
ofpnpth = 'xxx.pth'

print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_dict = torch.load(prepth, map_location=device)
ofpn_dict=torch.load(ofpnpth, map_location=device)
# num=0
# a=[]
# x=torch.zeros([64, 3, 7, 7])
# for k,v in ofpn_dict.items():
#     if num==0:
#         print('ofpn_pth',v)
#         a[0]=v 
#         num=num+1
# print(num)
# for k,v in pretrained_dict.items():
#     print('pretrained_dict',v.size()) 
#     num=num+1
# print(num)
# data = dict.fromkeys(np.arange(10))
# print(data)

# params = list(model.named_parameters())
# for k, v in params:
#     print('模型的key',k)
# for k,v in pretrained_dict.items():
#     if k=='extractor.0.conv1.weight':
#     # if k=='extractor.0.weight':
#         print(v)
# for k in ofpn_dict.items():
#     print('模型的key',k)

# new_list = list (ofpn_dict.keys() )
# trained_list = list (pretrained_dict.keys()  )
# # new_v = list (ofpn_dict.keys() )
# # trained_v = list (pretrained_dict.parameters()  )
# print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_list),len(trained_list)) )
# print("New state_dict first 10th parameters names")
# print(new_list[:1])
# print("trained state_dict first 10th parameters names")
# print(trained_list[:1])
# print(ofpn_dict[ new_list[0] ])

# # print(pretrained_dict[ trained_list[0] ])
# print(type(new_list))
# print(type(pretrained_dict))
# for i in range(258):
#     ofpn_dict[ new_list[i] ] = pretrained_dict[ trained_list[i] ]
# for i in range(258):
#     ofpn_dict[ new_list[i] ] = pretrained_dict[ trained_list[i] ]
# torch.save(model.state_dict(),'xxx.pth')



# net.load_state_dict(dict_new)
s='extractor.0.layer4.0.conv1.weight'
q='extractor.0.layer3.5.bn3.num_batches_tracked'
if not s.startswith('extractor.0.layer4.'):
        print (s)

