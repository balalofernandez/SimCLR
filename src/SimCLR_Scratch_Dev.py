import torch
import torch.nn as nn
import argparse
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
import os
os.environ['TORCH_USE_CUDA_DSA']="1"
import subprocess
from torch.optim.optimizer import Optimizer, required
import re
import torch
import numpy as np

from utils.Models import ProjectionHead, ResNet
from utils.Utilities import *
from torch.utils.data import DataLoader

######################################## DATASET #########################################

home = "/HOME"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base model', default="resnet34")
    parser.add_argument('--continue_pretraining', type=str, help='Resume 150 epochs more', default="False")
    parser.add_argument('--dataset', type=str, help='dataset for pretraining', default="inat64")
    args = parser.parse_args()
    resume = eval(args.continue_pretraining)
    model_name=args.model
    dataset_name=args.dataset

    print("#"*5,f"PRETRAINING {model_name} ON DATASET={dataset_name} (CONTINUATION={resume})","#"*5)

    if dataset_name == 'cifar10':
        #Keep in mind:
        #Gaussian Noise --> Not implemented by Pytorch
        #Cutout --> Not implemented by Pytorch
        transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        strength = 0.5 # Set to 1.0 if using Imagenet

        size=(32,32)#(224,224)
        transform_augmented = transforms.Compose([
            # transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(32,(0.8,1.0)),
            transforms.Compose([transforms.RandomApply(
                                    [transforms.ColorJitter(brightness=0.8*strength,
                                                            contrast=0.8*strength,
                                                            saturation=0.8*strength,
                                                            hue=0.2*strength)], p = 0.8),
                                transforms.RandomGrayscale(p=0.2)
                                ]),
            #If ImageNet add gaussian Blur
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(size[0]*10//100, size[1]*10//100), sigma=(0.1, 1.5))], p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 512 #Fourth best performing in the paper
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        trainset_normal = datasets.CIFAR10(root="./datasets", train=True, download=True, transform=transform_normal)
        trainloader_normal = DataLoader(trainset_normal, batch_size=batch_size, shuffle=False, num_workers=2)
        all_labels = np.array([label for _,label in trainset_normal])

        cifar_dataset_train = MixedCIFAR10(root="./datasets", train=True, transform=transform_augmented)
        trainloader_augmented = DataLoader(cifar_dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

        cifar_dataset_test = MixedCIFAR10(root="./datasets", train=False, transform=transform_augmented)
        # cifar_dataset_test = MixedINat(root="./datasets", transform=transform_augmented)
        testloader_augmented = DataLoader(cifar_dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)
    
    elif dataset_name == 'inat':
        resize_size = (128, 128) #224,224

        strength = 1.0
        transform_augmented = transforms.Compose([
            # transforms.Resize(size=resize_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(resize_size[0],(0.08,1.0)),
            transforms.Compose([transforms.RandomApply(
                                    [transforms.ColorJitter(brightness=0.8*strength,
                                                            contrast=0.8*strength,
                                                            saturation=0.8*strength,
                                                            hue=0.2*strength)], p = 0.8),
                                transforms.RandomGrayscale(p=0.2)
                                ]),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(resize_size[0]*10//100, resize_size[1]*10//100), sigma=(0.1, 2.0))], p = 0.5), #kernel=(22,22) needs to be odd
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(resize_size[0]*10//100 + 1, resize_size[1]*10//100 + 1), sigma=(0.1, 2.0))], p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        batch_size = 750#256
        inat_dataset_train = MixedINat128(
            '/cs/student/projects3/COMP0087/grp1/simclr_pytorch_new/datasets/iNat128/lanczos',
            transform=transform_augmented,
        )

        trainloader_augmented = DataLoader(
            inat_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
        )
    
    elif dataset_name == 'inat64':
        resize_size = (64, 64) #224,224

        strength = 1.0
        transform_augmented = transforms.Compose([
            # transforms.Resize(size=resize_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(resize_size[0],(0.08,1.0)),
            transforms.Compose([transforms.RandomApply(
                                    [transforms.ColorJitter(brightness=0.8*strength,
                                                            contrast=0.8*strength,
                                                            saturation=0.8*strength,
                                                            hue=0.2*strength)], p = 0.8),
                                transforms.RandomGrayscale(p=0.2)
                                ]),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(resize_size[0]*10//100, resize_size[1]*10//100), sigma=(0.1, 2.0))], p = 0.5), #kernel=(22,22) needs to be odd
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(resize_size[0]*10//100 + 1, resize_size[1]*10//100 + 1), sigma=(0.1, 2.0))], p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        batch_size = 350
        inat_dataset_train = MixedINat128(
            '/cs/student/projects3/COMP0087/grp1/simclr_pytorch_new/datasets/iNat64/lanczos',
            transform=transform_augmented,
        )

        trainloader_augmented = DataLoader(
            inat_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
        )

    elif dataset_name == "animalia":

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

        batch_size = 300    
        dataset_train = MixedAnimalia("/HOME/datasets/ImagesAnimalia",transform=transform_augmented)

        print("Length of dataset",len(dataset_train))

        trainloader_augmented = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )


    ######################################## TRAINING #########################################

    epoch_number = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet(model_name, True).to(device)
    if resume:
        checkpoint_path = f"{home}/FINAL_RESULTS/final_models/checkpoints/resnet34_inat64.pt"
        resnet.load_state_dict(torch.load(checkpoint_path))

    xent_loss = NTXentLoss(temperature= 0.5)

    # optim = LARS(
    #             [params for params in resnet.parameters() if params.requires_grad],
    #             lr=0.2, #Try 0.5 Get lr from 0.3*batch_size/256
    #             weight_decay=1e-6,
    #             exclude_from_weight_decay=["batch_normalization", "bias"],
    #         )
    learning_rate = 0.2
    optim = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
    #optim = torch.optim.SGD(resnet.parameters(), lr=0.06)

    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch : (epoch+1)/10.0, verbose = True)
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, eta_min=0.05, last_epoch=-1, verbose = True)
    running_loss = []

    for epoch in range(epoch_number):
        loss_epoch = 0
        for (batch_1, _), (batch_2, _) in trainloader_augmented:
            
            resnet.train()
            optim.zero_grad()
            batch_1,batch_2 = batch_1.to(device).float(),batch_2.to(device).float()
            
            z_1 = resnet(batch_1)
            z_2 = resnet(batch_2)
            
            loss = xent_loss(z_1, z_2)

            loss.backward()
            loss_epoch += loss.item()
            optim.step()

        if epoch < 10:
            warmupscheduler.step()
        if epoch >= 10:
            mainscheduler.step()

        running_loss.append(loss_epoch)

        
        if (epoch+1) % 5 ==0:
            save_path = f'{home}/FINAL_RESULTS/final_models/pretrained_checkpoints/{model_name}_{dataset_name}3.pt'
            torch.save(resnet.state_dict(), save_path)
            os.system(f"chmod -Rf 770 {home}/FINAL_RESULTS/final_models/")
        if (epoch) % 50 == 0:
            #Not all dataset have classes: This method won't work for the video SimCLR approach
            plot_features(resnet,512,trainloader_augmented,epoch)
            
        print(f"[Epoch {epoch}] LOSS:",loss_epoch)

    #Plot LOSS
    plot_loss(
        list = running_loss,
        path = f"{home}/FINAL_RESULTS/final_plots",
        name_of_file = f"pretrain_{model_name}_{epoch_number}epochs_{dataset_name}_accuracy_curve"
    )

    #CHMOD
    os.system(f"chmod -Rf 770 {home}/FINAL_RESULTS/final_models/")