import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import pickle
import csv

def predicting(model, device, loader, drugs):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    index = -1
    count = 0
    total_pred_dict = {}
    total_label_dict = {}
    with torch.no_grad():
        for data in loader:
            index = index + 1
            # #print("index: {}, drug_name: {}, drug[{}]: {} ".format(index, drug_name, index, drugs[index]))
            # #print(data)
            # if drug_name != drugs[index]:
            #     continue
            # count = count + 1
            print(total_preds.shape[0])
            drug = drugs[total_preds.shape[0]]
            data = data.to(device)
            if drug not in total_pred_dict:
                total_pred_dict[drug] = torch.Tensor()
                total_label_dict[drug] = torch.Tensor()
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_pred_dict[drug] = torch.cat((total_pred_dict[drug], output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_label_dict[drug] = torch.cat((total_label_dict[drug], data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(), total_pred_dict, total_label_dict

def caculator():
    dataset = 'GDSC'
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = GINConvNet().to(device)
    model.load_state_dict(torch.load('./model_GINConvNet_GDSC.model'))
    model.eval()
    with open('list_drug_mix_test', 'rb') as fp:
        drugs = pickle.load(fp)
    drug_names = {}
    for name in drugs:
        if name in drug_names:
            drug_names[name] += 1
        else:
            drug_names[name] = 1
    for name in drug_names:
        print("{}: {}".format(name, drug_names[name]))        
    

    G_test, P_test, total_pred_dict, total_label_dict = predicting(model, device, test_loader, drugs)
    print("G_test length: {}; P_test length: {}".format(len(G_test), len(P_test)))

    with open('result.csv', mode='w') as result:
        writer = csv.writer(result, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['file name', 'G_test', 'P_test'])
        for index in range(len(G_test)):
            writer.writerow([drugs[index], G_test[index], P_test[index] ])
    
    ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
    with open('employee_file.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['file name', 'G_test length', 'P_test', 'rmse', 'mse', 'pearson', 'spearman' ])
        print(ret_test)
        employee_writer.writerow(["all", len(G_test), len(P_test), ret_test[0], ret_test[1], ret_test[2], ret_test[3] ])
        for drug in total_pred_dict:
            G_test = total_pred_dict[drug].numpy().flatten()
            P_test = total_label_dict[drug].numpy().flatten()
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
            employee_writer.writerow([drug, len(G_test), len(P_test), ret_test[0], ret_test[1], ret_test[2], ret_test[3] ])

   
def load_drug_data():
    with open('list_drug_mix_test', 'rb') as fp:
        drug = pickle.load(fp)
    return drug
    
def load_cell_line_data():    
    with open('list_cell_mix_test', 'rb') as fp:
        cells = pickle.load(fp)
    cell_names = {}
    for name in cells:
        cell_names[name] = 1
    print(cell_names.keys())
    print(len(cell_names.keys()))
    
caculator()
# dataset = 'GDSC'
# test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix')
# print(test_data)
