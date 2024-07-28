# contains code for the 30% extended question
import torch
import torch.nn as nn
import argparse
from torchvision import datasets
import os
os.environ['TORCH_USE_CUDA_DSA']="1"
import subprocess
from torch.optim.optimizer import Optimizer, required
import re
import torch
import numpy as np
from torchvision.transforms import v2

from utils.Models import ProjectionHead, ResNet
from utils.Utilities import *
from torch.utils.data import DataLoader


from typing import Iterable
from os import listdir

def plot_features_video_sameimg(model, num_feats, all_labels, dataloader, epoch, val_df = None):
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    with torch.no_grad():
        for labelled_frames in dataloader:
            (x1, _), (x2, _), _ = labelled_frames
            x1 = x1.squeeze().to(device = next(model.parameters()).device, dtype = torch.float)
            out = model.avgpool(model.resnet(x1)).squeeze()
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
            
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    
    for i in range(10):
        plt.scatter(x_feats[all_labels==i,1][:128],x_feats[all_labels==i,0][:128])
    
    plt.legend([str(i) for i in range(10)])
    plt.savefig(f'/HOME/simclr_pytorch/figures/VideoSimCLR_training_sameimg_{epoch}.png') 
    plt.close()

def plot_features_video_next_frame(model, num_feats, all_labels, dataloader, epoch, val_df = None):
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    with torch.no_grad():
        for labelled_frames in dataloader:
            (x1, _), _, (x2, _) = labelled_frames
            x1 = x1.squeeze().to(device = next(model.parameters()).device, dtype = torch.float)
            out = model.avgpool(model.resnet(x1)).squeeze()
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
            
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    
    for i in range(10):
        plt.scatter(x_feats[all_labels==i,1][:128],x_feats[all_labels==i,0][:128])
    
    plt.legend([str(i) for i in range(10)])
    plt.savefig(f'/HOME/simclr_pytorch/figures/VideoSimCLR_training_diffimg_{epoch}.png') 
    plt.close()

######################################## DATASET #########################################

home = "/HOME"

if __name__ == "__main__":
    # getting args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base model', default="resnet34")
    parser.add_argument('--epochs', type = int ,help = 'no of epochs',default=200)
    parser.add_argument('--continue_pretraining', type=str, help='Resume 200 epochs more', default="False")
    parser.add_argument('--save_name', type=str, help='Change saving files names', default="videoSimCLR_adam")
    parser.add_argument('--lambdas', type=tuple, help='change lambdas to use', default=(0.5, 0.5))

    # extracting args
    args = parser.parse_args()
    resume = eval(args.continue_pretraining)
    model_name=args.model
    epoch = args.epochs
    print("#"*5,f"PRETRAINING {model_name} (CONTINUATION={resume})","#"*5)
    dataset_name = 'animalia'
    name = args.save_name
    lambdas_ = args.lambdas
    offsets_ = [*range(len(lambdas_))]
    offsets_.pop(0)
    offsets_ = tuple(offsets_)

    # checkpoint/save path
    save_path = f'{home}//FINAL_RESULTS//final_models//pretrained_checkpoints//{model_name}_video_{dataset_name}.pt'

    # creating transformations
    #Keep in mind:
    #Gaussian Noise --> Not implemented by Pytorch
    #Cutout --> Not implemented by Pytorch
    resize_size = (64, 64) #224,224
    transform_normal = v2.Compose([
        v2.Resize(size=resize_size),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    strength = 1.0
    transform_augmented = v2.Compose([
        v2.Resize(size=resize_size),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomResizedCrop(resize_size[0],(0.08,1.0)),
        v2.Compose([v2.RandomApply(
                                [v2.ColorJitter(brightness=0.8*strength,
                                                        contrast=0.8*strength,
                                                        saturation=0.8*strength,
                                                        hue=0.2*strength)], p = 0.8),
                            v2.RandomGrayscale(p=0.2)
                            ]),
        # v2.RandomApply([v2.GaussianBlur(kernel_size=(resize_size[0]*10//100, resize_size[1]*10//100), sigma=(0.1, 2.0))], p = 0.5), #kernel=(22,22) needs to be odd
        v2.RandomApply([v2.GaussianBlur(kernel_size=(resize_size[0]*10//100+1, resize_size[1]*10//100+1), sigma=(0.1, 2.0))], p = 0.5),
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # creating dataset
    batch_size = 200
    root_ = f"{home}//simclr_pytorch//datasets//ImagesAnimalia"
    dataset_train = VideoFrameDataset(root_,transform=transform_augmented, offsets=offsets_)

    print("Length of dataset",len(dataset_train))

    trainloader_augmented = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    ######################################## TRAINING #########################################

    epoch_number = epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device",device)
    resnet = ResNet(model_name,change_conv=True).to(device)
    if resume:
        print("RESUMING...")
        # checkpoint_path = f"{home}/FINAL_RESULTS/final_models/pretrained_checkpoints/resnet34_animalia_videosimclr.pt"
        resnet.load_state_dict(torch.load(save_path))

    # xent_loss = SimCLR_Loss(batch_size = batch_size, temperature=0.5)
    #xent_loss = NTXentLoss()
    vidloss = VideoSimCLRLoss(lambdas=lambdas_, batch_size = batch_size, temperature=0.5)

    # optim = LARS(
    #             [params for params in resnet.parameters() if params.requires_grad],
    #             lr=0.2, #Try 0.5 Get lr from 0.3*batch_size/256
    #             weight_decay=1e-6,
    #             exclude_from_weight_decay=["batch_normalization", "bias"],
    #         )
    learning_rate = 0.2
    optim = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch : (epoch+1)/10.0, verbose = True)
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, eta_min=0.05, last_epoch=-1, verbose = True)
    running_loss = []
    for epoch in range(epoch_number):
        loss_epoch = 0
        for labelled_frames in trainloader_augmented:
        
            resnet.train()
            optim.zero_grad()

            zs = []
            for (frame, _) in labelled_frames:
                frame = frame.to(device).float()
                zi = resnet(frame)
                zs.append(zi)

            loss = vidloss(*zs)
            # breakpoint()
            loss.backward()
            loss_epoch += loss.item()
            optim.step()

        if epoch < 10:
            warmupscheduler.step()
        if epoch >= 10:
            mainscheduler.step()

        running_loss.append(loss_epoch)

        if (epoch+1) % 5 == 0:
            torch.save(resnet.state_dict(), save_path)
            os.system(f"chmod -Rf 770 {home}/FINAL_RESULTS/final_models/")

        # if (epoch) % 50 == 0:
        #     plot_features_video_sameimg(resnet,512,range(10),trainloader_augmented,epoch)
        #     plot_features_video_next_frame(resnet,512,range(10),trainloader_augmented,epoch)

        print(f"[Epoch {epoch}] LOSS:",loss_epoch)

    # save_path = f'/HOME/simclr_pytorch/models/{model_name}_{dataset_name}_{name}.pt'
    torch.save(resnet.state_dict(), save_path)

    #Plot LOSS
    plot_loss(
        list = running_loss,
        path = f"{home}//FINAL_RESULTS//final_plots",
        name_of_file = f"baseline_{model_name}_{epoch_number}epochs_{dataset_name}_accuracy_curve_{name}"
    )


    #CHMOD
    os.system(f"chmod -Rf 770 {home}/FINAL_RESULTS/final_models/")