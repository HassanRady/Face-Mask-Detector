import torch
from torch import cuda
from torchvision import transforms
from PIL import Image

class initialize(object):
    classes = ['Mask ON', 'Mask OFF']
    
    def __init__(self):
        self.model = torch.load('model.pth')
        self.device = torch.device("cuda:0" if cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def predict(self, img, pil=False):
        
        if not pil:
            self.im = Image.fromarray(img)
        
        self.trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.im = self.trans(self.im)
        self.im = self.im.unsqueeze(0)
        self.im = self.im.to(self.device)
        self.model.eval()
        self.output = self.model(self.im)
        self.pred = self.output >= 0.5
        self.c = initialize.classes[self.pred.item()]
        
        return self.c