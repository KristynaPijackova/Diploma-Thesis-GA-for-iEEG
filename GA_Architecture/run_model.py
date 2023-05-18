import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import *
from statistics import *
from tqdm import tqdm
from inception_ask_blocks import *

from matplotlib import pyplot as plt
import seaborn as sns
from datetime import date

def run_model(arch_enc, outputs, inc_channels):
    nn_epochs = 10
    n_classes = 4
    inp = 8
    # GET MODEL INFORMATION
    # print(f'parameters values:\n\n{PARAMETERS.values()}')
    # arch_enc, outputs, inc_channels = PARAMETERS.values()
    print(f'\nArchitecture:\n{arch_enc}\nOutput channels:\n{outputs}\nInception reduction:\n{inc_channels}\n')

    # DATA INITIALIZATION
    dataset_train = Dataset('/home/pijackova/projects/DATASET_FNUSA', 'train_small_1_fnusa.csv')
    dataset_test = Dataset('/home/pijackova/projects/DATASET_FNUSA', 'valid_fnusa.csv')
    kern = dataset_train.__getitem__(0)[0].shape

    BATCH = 4
    LR = 0.001
    L2 = 0
    NWORKERS = 1
    NUM_ACC_STEPS = 32  #128/4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TRAIN = DataLoader(dataset=dataset_train,
                       batch_size=BATCH,
                       shuffle=True,
                       drop_last=False,
                       num_workers=NWORKERS)

    TEST = DataLoader(dataset=dataset_test,
                      batch_size=BATCH,
                      shuffle=True,
                      drop_last=False,
                      num_workers=NWORKERS)

    # MODEL INITIALIZATION
    model = Architecture_Encoder(in_channels=inp,
                                 architecture=arch_enc,
                                 output_channels=outputs,
                                 inception_output_channels=inc_channels,
                                 n_classes=n_classes)
    model = model.to(device)
    summary(model, input_size=(BATCH, 1, kern[1], kern[2]), depth=10)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    loss = nn.CrossEntropyLoss()
    statistics = Statistics()

    writer = SummaryWriter()
    monitor_kappa = []
    monitor_f1 = []
    monitor_auroc = []
    monitor_auprc = []
    monitor_loss = []

    monitor_f1_all = []
    monitor_auroc_all = []
    monitor_auprc_all = []

    start_time = datetime.now()
    for epoch in range(nn_epochs):
        torch.cuda.empty_cache()
        current_loss = 0
        epoch_loss = 0
        epoch_loss_valid = 0

        model.train()
        for i, (x, t) in enumerate(tqdm(TRAIN)):
            x = x.to(device).float()
            t = t.to(device).long()
            y = model(x)
            J = loss(input=y, target=t)
            J = J/NUM_ACC_STEPS
            J.backward()
            if ((i+1) % NUM_ACC_STEPS == 0) or (i+1 == len(TRAIN)):
                optimizer.zero_grad()
                optimizer.step()

        # validate
        model.eval()  # evaluate results for validation test
        for i, (x, t) in enumerate(tqdm(TEST)):
            x = x.to(device).float()
            t = t.to(device).long()
            y = model(x)


            J = loss(input=y, target=t)
            J = J/NUM_ACC_STEPS
            if ((i + 1) % NUM_ACC_STEPS == 0) or (i + 1 == len(TEST)):
                epoch_loss_valid += J.item()
            statistics.append(target=t.detach().cpu(), logits=y.detach().cpu())
        kappa, F1, conf, auroc, auprc, auroc_all, auprc_all = statistics.evaluate()
        end_time = datetime.now()

        # monitor kappa and f1 scores for each epoch to choose the best later
        monitor_kappa.append(kappa)
        monitor_f1.append(np.mean(F1))
        monitor_auroc.append(auroc)
        monitor_auprc.append(auprc)

        monitor_loss.append(epoch_loss_valid / TEST.__len__())


        monitor_f1_all.append(F1)
        monitor_auroc_all.append(auroc_all)
        monitor_auprc_all.append(auprc_all)

        # VALIDATION LOSS
        writer.add_scalar('Loss per Epoch/valid', epoch_loss_valid / TEST.__len__(), epoch)
        # KAPPA, mean F1
        writer.add_scalar('Score per Epoch/Kappa', kappa, epoch)
        writer.add_scalar('Score per Epoch/F1', np.mean(F1), epoch)

        # TEXT INFO ABOUT TRAINING AND VALIDATION TIMES
        time_elapsed = (end_time - start_time)
        text_time = ('Time spent on training and validation for {} epochs: {}'.format(EPOCHS, time_elapsed))
        writer.add_text('Model Time:', text_time)

        # CONFUSION MATRIX FOR TENSORBOARD
        classes = ['powerline', 'noise', 'pathology', 'physiology']
        cm = conf
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 7))
        sns.set(font_scale=1.3)
        conf_fig = sns.heatmap(cmn, cmap='Blues', annot=True, fmt='.2f', xticklabels=classes,
                               yticklabels=classes).get_figure()
        writer.add_figure("Confusion matrix", conf_fig, epoch)

        # F1 SCORE FOR TENSORBOARD
        writer.add_scalar('F1 Score/class 0 - powerline', F1[0], epoch)
        writer.add_scalar('F1 Score/class 1 - noise', F1[1], epoch)
        writer.add_scalar('F1 Score/class 2 - pathology', F1[2], epoch)
        writer.add_scalar('F1 Score/class 3 - physiology', F1[3], epoch)
        writer.add_scalar('AUROC', auroc, epoch)
        writer.add_scalar('AUPRC', auprc, epoch)

        # if i >= 3 and max(monitor_kappa) <= 76:
        #     writer.add_text('Training Interruption:', "Stopped training after epoch #3 due to low kappa score.")
        #     break

    # BEST KAPPA
    best_kappa = max(monitor_kappa)
    best_epoch = monitor_kappa.index(best_kappa)
    best_f1 = monitor_f1[best_epoch]
    best_loss = monitor_loss[best_epoch]

    best_auroc = monitor_auroc[best_epoch]
    best_auprc = monitor_auprc[best_epoch]

    best_f1_all = monitor_f1_all[best_epoch]
    best_auroc_all = monitor_auroc_all[best_epoch]
    best_auprc_all = monitor_auprc_all[best_epoch]


    # RESULTS AND HYPERPARAMETERS FOR TENSORBOARD
    writer.add_hparams(
        {"Loss": best_loss,
         "Kappa": best_kappa,
         "F1": best_f1,
         "AUROC": best_auroc,
         "AUPRC": best_auprc,

         "Best Epoch": best_epoch})

    writer.flush()
    writer.close()
    return (best_kappa, best_f1, best_epoch, str(time_elapsed), best_auroc, best_auprc,
            best_auroc_all, best_auprc_all, best_f1_all)
