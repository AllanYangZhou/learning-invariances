import torch
import numpy as np
import math
import argparse
from augerino import datasets, models, losses
from torch.utils.data import DataLoader, random_split
from oil.utils.mytqdm import tqdm
from smallnet import smallnet
import pandas as pd
from tqdm import tqdm


def main(args):
    net = smallnet(in_channels=1,num_targets=10)
    if args.disable_aug:
        model = net
    else:
        augerino = models.UniformAug()
        model = models.AugAveragedModel(net, augerino,ncopies=args.ncopies)
        start_widths = torch.ones(6) * -5.
        start_widths[2] = 1.
        model.aug.set_width(start_widths)
    
    softplus = torch.nn.Softplus()
    
    dataset = datasets.RotMNIST("~/datasets/", train=True)
    train_dset, val_dset = random_split(dataset, (50000, 10000))
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

    ## save init model ##
    fname = "/model" + str(args.aug_reg) + "_init.pt"
    torch.save(model.state_dict(), args.dir + fname)

    criterion = losses.no_aug_loss if args.disable_aug else losses.safe_unif_aug_loss
    torch.nn.CrossEntropyLoss()
    logger = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        batches = 0
        for i, data in enumerate(tqdm(trainloader), 0):
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
        train_acc = (outputs.detach().argmax(-1) == labels).float().mean().cpu().item()
        # print(f"Train epoch {epoch}: Loss {loss.item()}, Acc: {train_acc}, Aug widths: {softplus(model.aug.width).detach().data}")
        print(f"Train epoch {epoch}: Loss {loss.item()}, Acc: {train_acc}.")
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
    parser.add_argument("--disable_aug", action="store_true")
    args = parser.parse_args()

    main(args)
