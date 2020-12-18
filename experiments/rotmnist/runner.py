import torch
import numpy as np
import math
import argparse
from augerino import datasets, models, losses
from torch.utils.data import DataLoader, Subset
from oil.utils.mytqdm import tqdm
import pandas as pd
import torchsummary
from smallnet import smallnet


def main(args):
    net = smallnet(in_channels=1,num_targets=55)
    if args.disable_aug:
        model = net
    else:
        augerino = models.UniformAug()
        model = models.AugAveragedModel(net, augerino,ncopies=args.ncopies)
        start_widths = torch.ones(6) * -5.
        start_widths[2] = 1.
        model.aug.set_width(start_widths)
    
    softplus = torch.nn.Softplus()
    
    dataset = datasets.LocalRotMNIST("~/datasets/", train=True)
    train_dset, val_dset = Subset(dataset, range(50000)), Subset(dataset, range(50000, 60000))
    trainloader = DataLoader(train_dset, batch_size=args.batch_size)
    valloader = DataLoader(val_dset, batch_size=args.batch_size)

    if args.disable_aug:
        param_groups = model.parameters()
    else:
        param_groups = [{'name': 'model', 'params': model.model.parameters(), "weight_decay": args.wd}]
        param_groups.append({'name': 'aug', 'params': model.aug.parameters(), "weight_decay": 0.})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Using Cuda")
    torchsummary.summary(model, (1, 50, 50))

    ## save init model ##
    fname = "/model" + str(args.aug_reg) + "_init.pt"
    torch.save(model.state_dict(), args.dir + fname)

    criterion = losses.no_aug_loss if args.disable_aug else losses.safe_unif_aug_loss
    torch.nn.CrossEntropyLoss()
    logger = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        batches = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch} training")
        for i, data in enumerate(pbar, 0):
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
            log = []
            if not args.disable_aug:
                log += model.aug.width.tolist()
                log += model.aug.width.grad.data.tolist()
            log += [loss.item()]
            logger.append(log)        
            if not i % 10:
                train_acc = (outputs.argmax(-1) == labels).float().mean().cpu().item()
                pbar.set_postfix(train_acc=train_acc)
        train_acc = (outputs.detach().argmax(-1) == labels).float().mean().cpu().item()
        if not args.disable_aug:
            print(f"Train epoch {epoch}: Loss {loss.item()}, Acc: {train_acc}, Aug widths: {softplus(model.aug.width).detach().data}")
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
            print(f"Val accuracy: {np.mean(val_accs)}")
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
        default=0.1e-2,
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
    parser.add_argument("--disable_aug", action="store_true")
    args = parser.parse_args()

    main(args)
