from utils import classes_dict
from datasets import PartDataset
from models import PointNetCls

import torch 
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTIONS][DATA_FOLDER]...",
        description="Training script for PointNet model.",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "-p", "--num_points", type=int, default=2500
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10
    )
    parser.add_argument(
        "-m", "--model_dir", default="cls"
    )
    parser.add_argument(
        "-l", "--log_dir", default="logs"
    )
    parser.add_argument("DATA_FOLDER", type=str)
    return parser

def train_step(model, device, optimizer, points, target):
    ''' Execute one training pass.
    '''
    
    points = points.to(device) 
    target = target.to(device) 
    model.train()

    optimizer.zero_grad()
    pred, _ = model(points)
    loss = F.nll_loss(pred, target)
    loss.backward()
    optimizer.step()
    
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()

    return loss.item(), correct.item() / points.cpu().size()[0]

def test_step(model, device, points, target):
    ''' Execute one testing pass.
    '''

    points = points.to(device) 
    target = target.to(device) 
    model.eval()
    
    with torch.no_grad():
        pred, _ = model(points)
        loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()

    return loss.item(), correct.item() / points.cpu().size()[0]

def main():

    parser = init_argparse()
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_points = args.num_points
    workers = args.workers
    n_epochs = args.epochs
    model_dir = args.model_dir # TODO: mkdir when not present, ex:'cls/cls_model_0.pth'
    log_dir = args.log_dir 

    # DATASET & DATA LOADER
    DATA_FOLDER = args.DATA_FOLDER
    
    # Train Dataset & DataLoader
    dataset = PartDataset(root=DATA_FOLDER, npoints=num_points, classification=True, train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Test Dataset & DataLoader
    test_dataset = PartDataset(root=DATA_FOLDER, npoints=num_points, classification=True, train=False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


    # CALL THE MODEL
    num_classes = len(classes_dict.items())
    classifier = PointNetCls(num_points=num_points, k=num_classes)
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # VISUALIZATION
    writer = SummaryWriter(log_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    classifier.to(device)
 
    blue = lambda x:'\033[94m' + x + '\033[0m'
    for epoch in range(1, n_epochs + 1):
        
        num_batch = len(dataset) / batch_size
        train_loss = 0
        train_acc = 0 #TODO: accuracy should be implemented as a streaming metric
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target= Variable(points), Variable(target[:,0]) # need Variable?
            points = points.transpose(2, 1)
            loss, acc = train_step(classifier, device, optimizer, points, target)
            train_loss += loss
            train_acc += acc
            sys.stdout.write('\r[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss, acc))
        print("")
        train_loss = train_loss / num_batch
        train_acc = train_acc / num_batch

        num_batch = len(test_dataset) / batch_size    
        test_loss = 0
        test_acc = 0
        for j, data in enumerate(testdataloader): 
            points, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2, 1)
            loss, acc = test_step(classifier, device, points, target)
            test_loss += loss
            test_acc += acc
            # TODO: use \r to refresh display
            sys.stdout.write('\r[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, j, num_batch, blue('test'), loss, acc))
        print("")
        test_loss = test_loss / num_batch
        test_acc = test_acc / num_batch

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (model_dir, epoch))
        # writer (.add_scalars with train/test loss/accuracy)
        writer.add_scalars('Loss', {'Train' : train_loss,
                                    'Test' : test_loss,}, epoch)
        writer.add_scalars('Accuracy', {'Train' : train_acc,
                                    'Test' : test_acc,}, epoch)

        #TODO: visualize point global feat emb. each freq, images=
    writer.close()

if __name__ == "__main__":
    main()
