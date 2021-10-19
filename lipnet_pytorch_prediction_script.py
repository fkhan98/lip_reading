
def predict(sample_video_path, weight_path):

  """
    function that takes video and weight location, and returns lip-reading-recognized text from them
  
    Parameters
    ----------
    sample_video_path : str
        The file location of the video
    weight_path : str
        The file location of the weight

    Returns
    -------
    text
        a list of strings that is generated from lip-reading recognition
  
  """

  ## for lipnet model ##
  import torch 
  import torch.nn as nn
  import torch.nn.init as init
  import torch.nn.functional as F
  import math
  import numpy as np
  ## for lipnet model ##

  from torch.utils.data import DataLoader
  from torch.utils.data import Dataset
  import torch.optim as optim
  import os
  import sys
  import time
  import re
  import json
  import tempfile
  import shutil
  import cv2
  import face_alignment
  from skimage import io
  from google.colab.patches import cv2_imshow
  import glob
  import copy
  import editdistance
  import random


  ## cvtransforms.py ##

  def HorizontalFlip(batch_img, p=0.5):
      # (T, H, W, C)
      if random.random() > p:
          batch_img = batch_img[:,:,::-1,...]
      return batch_img

  def ColorNormalize(batch_img):
      batch_img = batch_img / 255.0
      return batch_img


  ## dataset.py ##

  class MyDataset(Dataset):
      letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

      def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
          self.anno_path = anno_path
          self.vid_pad = vid_pad
          self.txt_pad = txt_pad
          self.phase = phase
          
          with open(file_list, 'r') as f:
              self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
              
          self.data = []
          for vid in self.videos:
              items = vid.split(os.path.sep)            
              self.data.append((vid, items[-4], items[-1]))
          
                  
      def __getitem__(self, idx):
          (vid, spk, name) = self.data[idx]
          vid = self._load_vid(vid)
          anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))

          if(self.phase == 'train'):
              vid = HorizontalFlip(vid)
            
          vid = ColorNormalize(vid)                   
          
          vid_len = vid.shape[0]
          anno_len = anno.shape[0]
          vid = self._padding(vid, self.vid_pad)
          anno = self._padding(anno, self.txt_pad)
          
          return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
              'txt': torch.LongTensor(anno),
              'txt_len': anno_len,
              'vid_len': vid_len}
              
      def __len__(self):
          return len(self.data)
          
      def _load_vid(self, p): 
          files = os.listdir(p)
          files = list(filter(lambda file: file.find('.jpg') != -1, files))
          files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
          array = [cv2.imread(os.path.join(p, file)) for file in files]
          array = list(filter(lambda im: not im is None, array))
          array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
          array = np.stack(array, axis=0).astype(np.float32)
          return array
      
      def _load_anno(self, name):
          with open(name, 'r') as f:
              lines = [line.strip().split(' ') for line in f.readlines()]
              txt = [line[2] for line in lines]
              txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
          return MyDataset.txt2arr(' '.join(txt).upper(), 1)
      
      def _padding(self, array, length):
          array = [array[_] for _ in range(array.shape[0])]
          size = array[0].shape
          for i in range(length - len(array)):
              array.append(np.zeros(size))
          return np.stack(array, axis=0)
      
      @staticmethod
      def txt2arr(txt, start):
          arr = []
          for c in list(txt):
              arr.append(MyDataset.letters.index(c) + start)
          return np.array(arr)
          
      @staticmethod
      def arr2txt(arr, start):
          txt = []
          for n in arr:
              if(n >= start):
                  txt.append(MyDataset.letters[n - start])     
          return ''.join(txt).strip()
      
      @staticmethod
      def ctc_arr2txt(arr, start):
          pre = -1
          txt = []
          for n in arr:
              if(pre != n and n >= start):                
                  if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                      pass
                  else:
                      txt.append(MyDataset.letters[n - start])                
              pre = n
          return ''.join(txt).strip()
              
      @staticmethod
      def wer(predict, truth):        
          word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
          wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
          return wer
          
      @staticmethod
      def cer(predict, truth):        
          cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
          return cer


  ## Lipnet architecture ##

  class LipNet(torch.nn.Module):
      def __init__(self, dropout_p=0.5):
          super(LipNet, self).__init__()
          self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
          self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
          
          self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
          self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
          
          self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
          self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
          
          self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
          self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
          
          self.FC    = nn.Linear(512, 27+1)
          self.dropout_p  = dropout_p

          self.relu = nn.ReLU(inplace=True)
          self.dropout = nn.Dropout(self.dropout_p)        
          self.dropout3d = nn.Dropout3d(self.dropout_p)  
          self._init()
      
      def _init(self):
          
          init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
          init.constant_(self.conv1.bias, 0)
          
          init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
          init.constant_(self.conv2.bias, 0)
          
          init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
          init.constant_(self.conv3.bias, 0)        
          
          init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
          init.constant_(self.FC.bias, 0)
          
          for m in (self.gru1, self.gru2):
              stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
              for i in range(0, 256 * 3, 256):
                  init.uniform_(m.weight_ih_l0[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                  init.orthogonal_(m.weight_hh_l0[i: i + 256])
                  init.constant_(m.bias_ih_l0[i: i + 256], 0)
                  init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                  init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                  init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
          
          
      def forward(self, x):
          
          x = self.conv1(x)
          x = self.relu(x)
          x = self.dropout3d(x)
          x = self.pool1(x)
          
          x = self.conv2(x)
          x = self.relu(x)
          x = self.dropout3d(x)        
          x = self.pool2(x)
          
          x = self.conv3(x)
          x = self.relu(x)
          x = self.dropout3d(x)        
          x = self.pool3(x)
          
          # (B, C, T, H, W)->(T, B, C, H, W)
          x = x.permute(2, 0, 1, 3, 4).contiguous()
          # (B, C, T, H, W)->(T, B, C*H*W)
          x = x.view(x.size(0), x.size(1), -1)
          
          self.gru1.flatten_parameters()
          self.gru2.flatten_parameters()
          
          x, h = self.gru1(x)        
          x = self.dropout(x)
          x, h = self.gru2(x)   
          x = self.dropout(x)
                  
          x = self.FC(x)
          x = x.permute(1, 0, 2).contiguous()
          return x
          

  ## Helper Functions ##

  def get_position(size, padding=0.25):
      
      x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                      0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                      0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                      0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                      0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                      0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                      0.553364, 0.490127, 0.42689]
      
      y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                      0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                      0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                      0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                      0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                      0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                      0.784792, 0.824182, 0.831803, 0.824182]
      
      x, y = np.array(x), np.array(y)
      
      x = (x + padding) / (2 * padding + 1)
      y = (y + padding) / (2 * padding + 1)
      x = x * size
      y = y * size
      return np.array(list(zip(x, y)))
  def cal_area(anno):
      return (anno[:,0].max() - anno[:,0].min()) * (anno[:,1].max() - anno[:,1].min()) 

  def output_video(p, txt, dst):
      files = os.listdir(p)
      files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

      font = cv2.FONT_HERSHEY_SIMPLEX
      
      for file, line in zip(files, txt):
          img = cv2.imread(os.path.join(p, file))
          h, w, _ = img.shape
          img = cv2.putText(img, line, (w//8, 11*h//12), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
          img = cv2.putText(img, line, (w//8, 11*h//12), font, 1.2, (255, 255, 255), 0, cv2.LINE_AA)  
          h = h // 2
          w = w // 2
          img = cv2.resize(img, (w, h))     
          cv2.imwrite(os.path.join(p, file), img)
      
      cmd = "ffmpeg -y -i {}/%d.jpg -r 25 \'{}\'".format(p, dst)
      os.system(cmd)

  def transformation_from_points(points1, points2):
      points1 = points1.astype(np.float64)
      points2 = points2.astype(np.float64)
  
      c1 = np.mean(points1, axis=0)
      c2 = np.mean(points2, axis=0)
      points1 -= c1
      points2 -= c2
      s1 = np.std(points1)
      s2 = np.std(points2)
      points1 /= s1
      points2 /= s2
  
      U, S, Vt = np.linalg.svd(points1.T * points2)
      R = (U * Vt).T
      return np.vstack([np.hstack(((s2 / s1) * R,
                                        c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])

  def load_video(file):
      p = tempfile.mkdtemp()
      cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, p)
      os.system(cmd)
      
      files = os.listdir(p)
      files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
          
      array = [cv2.imread(os.path.join(p, file)) for file in files]
      
      
      array = list(filter(lambda im: not im is None, array))
      #array = [cv2.resize(im, (100, 50), interpolation=cv2.INTER_LANCZOS4) for im in array]
      
    #   fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
      fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
      points = [fa.get_landmarks(I) for I in array]
      
      front256 = get_position(256)
      video = []
      for point, scene in zip(points, array):
          if(point is not None):
              shape = np.array(point[0])
              shape = shape[17:]
              M = transformation_from_points(np.matrix(shape), np.matrix(front256))
            
              img = cv2.warpAffine(scene, M[:2], (256, 256))
              (x, y) = front256[-20:].mean(0).astype(np.int32)
              w = 160//2
              img = img[y-w//2:y+w//2,x-w:x+w,...]
              img = cv2.resize(img, (128, 64))
              video.append(img)
      
      
      video = np.stack(video, axis=0).astype(np.float32)
      video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0

      return video, p
  def ctc_decode(y):
      y = y.argmax(-1)
      t = y.size(0)
      result = []
      for i in range(t+1):
          result.append(MyDataset.ctc_arr2txt(y[:i], start=1))
      return result



  ## Prediction ##

  model = LipNet()
  # model = model.cuda()
  # pretrained_dict = torch.load(weight_path,map_location=torch.device('cuda'))
  model = model.to('cpu')
  pretrained_dict = torch.load(weight_path,map_location=torch.device('cpu'))


  model_dict = model.state_dict()

  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
  missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

  print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
  print('miss matched params:{}'.format(missed_params))

  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

  video, img_p = load_video(sample_video_path)
  # y = model(video[None,...].to('cuda'))
  y = model(video[None,...].to('cpu'))
  txt = ctc_decode(y[0])

  output_video(img_p, txt, "/content/")

  return txt

