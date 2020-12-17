import torch
import numpy as np
import math
import argparse
from augerino import datasets, models, losses
from torch.utils.data import DataLoader, Subset
from oil.utils.mytqdm import tqdm
from smallnet import smallnet
import pandas as pd
def main(args):
    net = smallnet(in_channels=1,num_targets=10)
    augerino = models.UniformAug()
    model = models.AugAveragedModel(net, augerino,ncopies=args.ncopies)
    
    start_widths = torch.ones(6) * -5.
    start_widths[2] = 1.
    model.aug.set_width(start_widths)
    
    softplus = torch.nn.Softplus()
    
    dataset = datasets.RotMNIST("~/datasets/", train=True)
    trainset, valset = Subset(dataset, range(50000)), Subset(dataset, range(50000, 60000))
    trainloader = DataLoader(trainset, batch_size=args.batch_size)
    valloader = DataLoader(valset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam([{'name': 'model', 'params': model.model.parameters(), "weight_decay": args.wd}, 
                                  {'name': 'aug', 'params': model.aug.parameters(), "weight_decay": 0.}], 
                                 lr=args.lr)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Using Cuda")

    ## save init model ##
    fname = "/model" + str(args.aug_reg) + "_init.pt"
    torch.save(model.state_dict(), args.dir + fname)

    criterion = losses.safe_unif_aug_loss
    logger = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        batches = 0
        trainbar = tqdm(trainloader, desc=f"Training epoch {epoch}")
        for i, data in enumerate(trainbar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels, model, reg=args.aug_reg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            batches += 1
            # print(epoch, loss.item(), softplus(model.aug.width).detach().data)
            log = model.aug.width.tolist()
            log += model.aug.width.grad.data.tolist()
            log += [loss.item()]
            logger.append(log)        
            if not i % 10:
                train_acc = (outputs.argmax(-1) == labels).float().mean().cpu().item()
                trainbar.set_postfix(train_acc=train_acc)
        with torch.no_grad():
            model.eval()
            val_accs = []
            for i, data in enumerate(valloader):
                inputs, labels = data
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                acc = (outputs.argmax(-1) == labels).float().mean().cpu().item()
                val_accs.append(acc)
            print(f"Epoch {epoch} val accuracy: {np.mean(val_accs)}")
            model.train()

    fname = "/model" + str(args.aug_reg) + ".pt"
    torch.save(model.state_dict(), args.dir + fname)
    df = pd.DataFrame(logger)
    df.to_pickle(args.dir + "/auglog_" + str(args.aug_reg) + ".pkl")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="olivetti augerino")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--aug_reg",
        type=float,
        default=0.1,
        help="augmentation regularization weight",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        metavar="weight_decay",
        help="weight decay",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    parser.add_argument(
        "--ncopies",
        type=int,
        default=4,
        metavar="N",
        help="number of augmentations in network (defualt: 4)"
    )
    args = parser.parse_args()

    main(args)
