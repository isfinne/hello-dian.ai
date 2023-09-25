import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
from modules import *
from data_avat import AvatorSet
import matplotlib.pyplot as plt
import os

# 迭代数量: 74.8w
# target: 100w

batch_size = 64
noiseDim = 100
lr = 1e-4
n_epoch = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

slices = (35000, 70000)
#slices = (0, 35000)
#slices = (60000, 70000)
#slices = (4000, 5000)
slices = (59000, 60000)

#dataMode = 'single'
dataMode = 'all'
load = True
start_epoch = 5


#记得每次更新最新的模型
params = ('.\lab4\modelSave\dcgan_g_current.pth', '.\lab4\modelSave\dcgan_d_current.pth')

def load_model_gd(param_G, param_D):
    dict_G = torch.load(param_G)
    dict_D = torch.load(param_D)
    return dict_G, dict_D

def train_epoch(models, optimizers, trainloader, criterion, epoch, device):
    model_G, model_D = models
    opt_G, opt_D = optimizers
    model_G.to(device)
    model_D.to(device)
    model_G.train()
    model_D.train()
    pbar = tqdm(trainloader)
    pbar.set_description(f'epoch {epoch : 2}')    
    for data in pbar:
        images = data
        images = images.to(device)

        bs = images.size(0)

        """ Train D """
        noise = Variable(torch.randn(bs, noiseDim)).to(device)
        images_real = Variable(images).to(device)
        images_fake = model_G(noise)

        # label        
        label_real = torch.ones((bs)).to(device)
        label_fake = torch.zeros((bs)).to(device)

        # dis
        logit_real = model_D(images_real.detach())
        logit_fake = model_D(images_fake.detach())
        
        # compute loss
        loss_real = criterion(logit_real, label_real)
        loss_fake = criterion(logit_fake, label_fake)
        loss_D = (loss_real + loss_fake) / 2

        # update model
        model_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        """ train G """
        # leaf
        noise = Variable(torch.randn(bs, noiseDim)).to(device)
        images_fake = model_G(noise)

        # dis
        logit_fake = model_D(images_fake)
        
        # compute loss
        loss_G = criterion(logit_fake, label_real)

        # update model
        model_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        pbar.set_postfix_str(f' Loss_D: {loss_D.item():.4f}'
                             f' Loss_G: {loss_G.item():.2f}')
        #print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    pass

def test_eopch(generator, epoch, device):
    model_G = generator
    model_G.to(device)
    model_G.eval()
    
    noise = Variable(torch.randn(batch_size, noiseDim)).to(device)
    f_imgs_sample = (model_G(noise).data + 1) / 2.0
    filename = os.path.join('lab4\imageSave', f'Epoch_{start_epoch+epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    # show generated image
    grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    
    pass

def main():
    trainloader = data.DataLoader(AvatorSet(slices= slices, dataMode= dataMode),
                                  batch_size= batch_size, shuffle=True, num_workers= 4)
    
    # load trained model
    
    model_G = Generator(noise_size= noiseDim).to(device)
    model_D = Discriminator(3).to(device)
    if load :
        dict_G, dict_D = load_model_gd( *params )
        #print(dict_G)
        
        #model_G.load_state_dict(dict_G, strict= False)
        #model_D.load_state_dict(dict_D, strict= False)
        model_G.load_state_dict(dict_G)
        model_D.load_state_dict(dict_D)
        
        #start_epoch = dict_G['epoch'] + 1
        pass
    models = (model_G, model_D)
    
    opt_G = optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizers = (opt_G, opt_D)
    
    criterion = nn.BCELoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    

    for epoch in range(n_epoch):
        train_epoch(models, optimizers, trainloader, criterion, 
                    epoch, device)
        test_eopch(model_G, epoch, device)
        #if (start_epoch + epoch + 1) % 2 == 0:
            #with torch.no_grad():
        #torch.save(model_G.state_dict(), os.path.join('lab4\modelSave\\', f'dcgan_g_{start_epoch +epoch+1}.pth'))
        #torch.save(model_D.state_dict(), os.path.join('lab4\modelSave\\', f'dcgan_d_{start_epoch +epoch+1}.pth'))
        torch.save(model_G.state_dict(), os.path.join('lab4\modelSave\\', f'dcgan_g_current.pth'))
        torch.save(model_D.state_dict(), os.path.join('lab4\modelSave\\', f'dcgan_d_current.pth'))
    pass


if __name__ == '__main__':
    main()
    
