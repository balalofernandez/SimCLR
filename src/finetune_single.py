import torch
import torch.nn as nn

import torch

from utils.Models import ProjectionHead, ResNet, FullNet
from utils.Utilities import plot_loss, accuracy, segmentation_plots, plot_accuracy, PET_dataset
import argparse
import pickle


home = "/HOME"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, help='Encoder model', default="resnet34")
    parser.add_argument('--percentage_PET', type=str, help='Amount of PET', default="1")
    parser.add_argument('--method', type=str, help='Baseline or not', default="videosimclr_adam")
    args = parser.parse_args()
    model_name=args.backbone
    percentage_PET = args.percentage_PET
    method = args.method

    print("#"*5,f"{method} run with encoder {model_name}","#"*5)

    ##### DATA #####

    size=(128,128)
    batch_size=100
    loader_train_PET, loader_validation_PET, loader_test_PET, development_list_PET = PET_dataset(size, batch_size)
    
    ##### LOAD MODEL #####

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = ResNet(model_name).to(device)
    if method != "baseline":
        checkpoint_path = f"{home}/FINAL_RESULTS/final_models/pretrained_checkpoints/resnet34_inat642.pt"
        resnet.load_state_dict(torch.load(checkpoint_path))

    model = FullNet(resnet, size).to(device)

    # resnet = ResNet("resnet34").to(device)

    # #Load model
    # checkpoint_path = f"/HOME/resnet_weights_DCL_300.pt"
    # resnet.resnet.load_state_dict(torch.load(checkpoint_path))
    # model = FullNet(resnet, size).to(device)

    ##### SUPERVISED LEARNING LOOP #####

    epoch_number = 20

    cross_entropy = nn.CrossEntropyLoss()

    learning_rate = 0.01
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    accuracy_list = []

    for epoch in range(1,epoch_number+1):
        print(f"Start training epoch #{epoch}")
        running_loss = 0.0
        loss_per_epoch = 0.0

        if percentage_PET == "100":
            index = -1
        elif percentage_PET == "10":
            index = 1
        else:
            index = 0

        for i, data in enumerate(development_list_PET[index], 1):
            images, segmentations = data

            model.train()

            images = images.to(device).float()
            segmentations = segmentations.to(device) #Does it need the float?
            #segmentations.shape --> (batch_size, 1, 32, 32)
            #cross entropy loss requires shape --> (batch_size, 32, 32) Thus, I squeeze that dimension off
            segmentations = segmentations.squeeze(1)

            optim.zero_grad()

            decoded_images = model(images).squeeze()

            loss = cross_entropy(decoded_images, segmentations)

            loss.backward()

            optim.step()

            running_loss += loss.item()
            
            loss_per_epoch += running_loss

            if i % 10 == 0:
                print(f"[Epoch {epoch}], average batch loss = {running_loss/10}")
                running_loss = 0.0
        
        
        acc = accuracy(dataloader = loader_test_PET, model = model.eval(), device = device)
        accuracy_list.append(acc)
        print(f"[Epoch {epoch}], test set accuracy = {acc}")

        loss_list.append(loss_per_epoch)
        loss_per_epoch = 0

    max_accuracy = max(accuracy_list)

    ##### RESULTS HOUSEKEEPING #####

    #Save ground_truth segmentation vs generated_segmentation
    segmentation_plots(
        dataloader_batch = next(iter(loader_test_PET)),
        model = model.eval(),
        device = device,
        saving_path = f"{home}/FINAL_RESULTS/final_plots",
        name_of_file = f"{method}_{model_name}_{epoch_number}epochs_segmentations_{percentage_PET}"
    )

    #Save model
    out_filename = f"{method}_{model_name}_{epoch_number}epochs_model_{percentage_PET}"
    torch.save(model.state_dict(), f"{home}/FINAL_RESULTS/final_models/{out_filename}.pt")

    #Plot and save loss curve
    plot_loss(
        list = loss_list,
        path = f"{home}/FINAL_RESULTS/final_plots/",
        name_of_file = f"{method}_{model_name}_{epoch_number}epochs_loss_curve_{percentage_PET}"
    )

    #Plot and save accuracy curve
    plot_accuracy(
        list = accuracy_list,
        path = f"{home}/FINAL_RESULTS/final_plots/",
        name_of_file = f"{method}_{model_name}_{epoch_number}epochs_accuracy_curve_{percentage_PET}"
    )

    #Save loss list and accuracy list for future plotting purposes, I use pickle, we can use whatever
    file_name_loss_list = f"{method}_{model_name}_{epoch_number}epochs_loss_list_{percentage_PET}"
    with open(f"{home}/FINAL_RESULTS/lists/{file_name_loss_list}.pkl", 'wb') as f:
        pickle.dump(loss_list, f)
    
    file_name_acc_list = f"{method}_{model_name}_{epoch_number}epochs_accuracy_curve_{percentage_PET}"
    with open(f"{home}/FINAL_RESULTS/lists/{file_name_acc_list}.pkl", 'wb') as f:
        pickle.dump(accuracy_list, f)


