import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


m = 10 # minimum frame nums

class VideoDataset(Dataset):
    def __init__(self, videos, transform, d3=False):      
        self.videos = videos
        self.transform = transform
        self.d3 = d3
    def __len__(self):
        return len(self.videos)
    def __getitem__(self, idx):
        
#         video=torch.stack([self.transform(image=cv2.imread(x,cv2.COLOR_BGR2RGB))['image'] for x in self.videos[idx]][:m])
#         video=torch.stack([self.transform(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)) for x in self.videos[idx]][:m])
        l = len(self.videos[idx])
        k = l//m
        video=torch.stack([self.transform(Image.open(self.videos[idx][x])) for x in range(0,k*10,k)])
        s = video.shape
        if self.d3:
            video = video.view(s[1],s[0],s[2],s[3])
        label = self.videos[idx][0].split('/')[-3]
        label = int(label.split('-')[0]) - 1
        
        return video, label