import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Subset, DataLoader
import numpy as np
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from typing import Sequence, Union
import os
import torch
from torch import distributed as torch_dist
from torch import nn
import json
from utils.Models import ProjectionHead, ResNet, FullNet
from utils.Utilities import plot_loss, save_results, split_development_set, accuracy, segmentation_plots, plot_accuracy, PET_dataset
import pickle

home = "/HOME"
num_epochs = 100
accuracy_max_list = []
method_list = ["baseline", "simclr_inat_adam", "simclr_animalia_adam","simclr_adam_cifar10","videosimclr_adam"]
split_list = [1,10,100]



def save_results(result_txt,value,model):
    file = f"{home}/FINAL_RESULTS/results.json"
    if not(os.path.exists(file)):
        data = {}
    else:
        with open(file, 'r') as f:
            data = json.load(f)
    if data.get(model) == None:
        data[model] = {}
    data[model][result_txt] = value
    with open(file, 'w') as file:
        json.dump(data, file, indent=4)

for method in method_list:
      for split in split_list:
            filename = f'{home}/FINAL_RESULTS/lists/{method}_resnet34_{num_epochs}epochs_accuracy_curve_{split}.pkl'

            with open(filename, 'rb') as file:
                  accuracy_list = pickle.load(file)
                  accuracy_max_list.append(max(accuracy_list))


accuracy_baseline_1 = accuracy_max_list[0] #Feed it the accuracy list for every finetuning epoch
save_results("baseline_1",accuracy_baseline_1,f"{method_list[0]}_resnet34_{num_epochs}")
accuracy_baseline_10 = accuracy_max_list[1]
save_results("baseline_10",accuracy_baseline_10,f"{method_list[0]}_resnet34_{num_epochs}")
accuracy_baseline_100 = accuracy_max_list[2]
save_results("baseline_100",accuracy_baseline_100,f"{method_list[0]}_resnet34_{num_epochs}")

accuracy_inat_1 = accuracy_max_list[3]
save_results("iNat_1",accuracy_inat_1,f"{method_list[1]}_resnet34_{num_epochs}")
accuracy_inat_10 = accuracy_max_list[4]
save_results("iNat_10",accuracy_inat_10,f"{method_list[1]}_resnet34_{num_epochs}")
accuracy_inat_100 = accuracy_max_list[5]
save_results("iNat_100",accuracy_inat_100,f"{method_list[1]}_resnet34_{num_epochs}")

accuracy_video_1 = accuracy_max_list[6]
save_results("video_1",accuracy_video_1,f"{method_list[2]}_resnet34_{num_epochs}")
accuracy_video_10 = accuracy_max_list[7]
save_results("video_10",accuracy_video_10,f"{method_list[2]}_resnet34_{num_epochs}")
accuracy_video_100 = accuracy_max_list[8]
save_results("video_100",accuracy_video_100,f"{method_list[2]}_resnet34_{num_epochs}")

accuracy_cifar_1 = accuracy_max_list[9]
save_results("cifar_1",accuracy_cifar_1,f"{method_list[3]}_resnet34_{num_epochs}")
accuracy_cifar_10 = accuracy_max_list[10]
save_results("cifar_10",accuracy_cifar_10,f"{method_list[3]}_resnet34_{num_epochs}")
accuracy_cifar_100 = accuracy_max_list[11]
save_results("cifar_100",accuracy_cifar_100,f"{method_list[3]}_resnet34_{num_epochs}")

accuracy_video_t_1 = accuracy_max_list[12]
save_results("videot_1",accuracy_video_t_1,f"{method_list[4]}_resnet34_{num_epochs}")
accuracy_video_t_10 = accuracy_max_list[13]
save_results("videot_10",accuracy_video_t_10,f"{method_list[4]}_resnet34_{num_epochs}")
accuracy_video_t_100 = accuracy_max_list[14]
save_results("videot_100",accuracy_video_t_100,f"{method_list[4]}_resnet34_{num_epochs}")



# Data
labels = ['1', '10', '100']

baseline = [accuracy_baseline_1, accuracy_baseline_10, accuracy_baseline_100]
inat = [accuracy_inat_1, accuracy_inat_10, accuracy_inat_100]
videos = [accuracy_video_1, accuracy_video_10, accuracy_video_100]
cifar = [accuracy_cifar_1,accuracy_cifar_10,accuracy_cifar_100]
video_t = [accuracy_video_t_1,accuracy_video_t_10,accuracy_video_t_100]

# Number of groups
num_labels = len(labels)

# Set the positions of the bars
index = np.arange(num_labels)
bar_width = 0.1

# Create bars
fig, ax = plt.subplots()
cmap = plt.get_cmap('tab20')

bar1 = ax.bar(index, baseline, bar_width, label='Baseline', color=cmap(0))
bar2 = ax.bar(index + bar_width, inat, bar_width, label='SimCLR - iNat', color=cmap(4))
bar3 = ax.bar(index + bar_width * 2, videos, bar_width, label='SimCLR - Videos', color=cmap(2))
bar4 = ax.bar(index + bar_width * 3, cifar, bar_width, label='SimCLR - CIFAR10', color=cmap(3))
bar5 = ax.bar(index + bar_width * 4, video_t, bar_width, label='SimCLR - VideoT', color=cmap(5))

# Add labels, title and axes ticks
ax.set_xlabel(r'Percentage of PET development set (%)', fontsize=16)
ax.set_ylabel(r'Accuracy (%)', fontsize=16)
ax.set_title(r'Models Accuracy vs Percentage of PET', fontsize=18)
ax.set_xticks(index + (3/2)*(bar_width))
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

path = f"{home}/FINAL_RESULTS/final_plots"
name_of_file = f"PET_accuracy_splits_all_models"
plt.savefig(f'{path}/{name_of_file}.png')

# Show the plot
plt.close()


##### SEGMENTATIONS PLOT #####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

method_list = ["baseline", "simclr_inat_adam", "simclr_animalia_adam",'simclr_adam_cifar10','videosimclr_adam']

size = (128,128)
batch_size=100
_, _, loader_test_PET, _ = PET_dataset(size, batch_size)

#Create plots
num_rows = 2
fig,axs = plt.subplots(num_rows,6,figsize=(6,2))
axs[0,0].set_title(r"G-T")
axs[0,1].set_title(r"Baseline")
axs[0,2].set_title(r"iNat")  
axs[0,3].set_title(r"Animalia")
axs[0,4].set_title(r"CIFAR10")
axs[0,5].set_title(r"VideoT")

#Ground truth first
images, segmentations = next(iter(loader_test_PET))
images = images.to(device).float()
segmentations = torch.argmax(segmentations, dim=1,keepdim=True).detach().cpu()
for i in range(num_rows):
      axs[i,0].imshow(segmentations[i].squeeze().numpy())
      axs[i,0].set_axis_off()

for j in range(len(method_list)):
      #Load model
      resnet = ResNet(model="resnet34").to(device)
      checkpoint_path = f"{home}/FINAL_RESULTS/final_models/{method_list[j]}_resnet34_{num_epochs}epochs_model_100.pt"
      model = FullNet(resnet, size).to(device)
      model.load_state_dict(torch.load(checkpoint_path))

      #Do inference
      model_out = model(images.to(device)).cpu().detach()
      labels = torch.argmax(model_out.squeeze(), dim=1,keepdim=True).detach().cpu()
      for i in range(num_rows):
            axs[i,j+1].imshow(labels[i].squeeze().numpy())
            axs[i,j+1].set_axis_off()

path = f"{home}/FINAL_RESULTS/final_plots"
name_of_file = f"PET_segmentations_all_models"
plt.savefig(f'{path}/{name_of_file}.png')

plt.close(fig)




