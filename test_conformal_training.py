import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.utils import fix_randomness



class MNIST_Net(nn.Module):
    def __init__(self, num_classes):
        super(MNIST_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        #print("Input shape before reshaping:", x.shape)
        x = x.view(-1, 28 * 28)
        #print("Input shape after reshaping:", x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x

class Another_Net(nn.Module):
    def __init__(self, num_classes):
        super(Another_Net, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        #print("Input shape before reshaping:", x.shape)
        x = x.view(-1, 3 * 224 * 224)
        #print("Input shape after reshaping:", x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x
            
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        #print(f"Batch {batch_idx + 1}: Output shape - {output.shape}, Target shape - {target.shape}")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
            

def test_training(name, num_classes, train_data_loader, cal_data_loader, test_data_loader):
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    result = {}
    loss = "CE"
    criterion = nn.CrossEntropyLoss()
    
    predictors = [SplitPredictor, ClassWisePredictor, ClusterPredictor]
    scores = ["THR", "APS", "RAPS", "SAPS"]
    
    ##################################
    # Training a pytorch model
    ##################################
    if name == "MNIST":
        model = MNIST_Net(num_classes)
    else:
        model = Another_Net(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(5):
        train(model, train_data_loader, criterion, optimizer, epoch)

    for score in scores:
        if score == "THR":
            score_function = THR()
        elif score == "APS":
            score_function = APS()
        elif score == "RAPS":
            score_function = RAPS(1, 0)
        elif score == "SAPS":
            score_function = SAPS(weight=0.2)
        if score not in result:
            result[score] = {}
        for class_predictor in predictors:
            predictor = class_predictor(score_function, model)
            result[score][predictor.__class__.__name__]={}
            for alpha in alphas:
                result[score][predictor.__class__.__name__][alpha]={}
                predictor.calibrate(cal_data_loader, alpha)                
                tmp_res = predictor.evaluate(test_data_loader)
                result[score][predictor.__class__.__name__][alpha]['Coverage_rate'] = tmp_res['Coverage_rate']
                result[score][predictor.__class__.__name__][alpha]['Average_size'] = tmp_res['Average_size']
                print(f"Score: {score}. Predictor : {predictor.__class__.__name__}. Alpha : {alpha}. Result is {result[score][predictor.__class__.__name__][alpha]}")

