# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import os
import sys
import pickle
import itertools
import copy
import re

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from matplotlib import pyplot as plt
from ga_model import Model, evaluate_model, uncompress_model


def sample_basis(origin):
    basis = []
    for m in origin.modules():
        if isinstance(m, nn.Conv2d):
            dev = torch.normal(m.weight-m.weight, 1.0)
            dev = dev / dev.norm(1) * m.weight.norm(1)
            basis.append(dev.data)
            
        #elif isinstance(m,nn.Linear):
            #dev = torch.normal(m.weight-m.weight, 1.0)
            #dev = dev / dev.norm(1)
            #basis.append(dev.data)
        else:
            basis.append(None)
    return basis


def basis_norm(b1,b2):
    norm = 0.0
    for m1,m2 in zip(b1, b2):
        if m1 is not None:
            m1, m2 = m1.view(-1), m2.view(-1)
            norm += torch.dot(m1,m2)
            
    return norm
            
 
def mix_point(origin, basis, coord):
    model = copy.deepcopy(origin)
    for mo, m, b1, b2 in zip(origin.modules(), model.modules(),basis[0],basis[1]):
        if isinstance(m, nn.Conv2d):
            m.weight = Parameter( mo.weight.data + b1 * coord[0] + b2 * coord[1])
        #elif isinstance(m,nn.Linear):
        #    m.weight = Parameter(mo.weight.data + b1 * coord[0] + b2 * coord[1])
    return model
        
   
def main(model_file, env_name, contour_layer=20, save_fig = 'test.png', resolution=3, rep=1):
    origin = torch.load(model_file)
    b1 = sample_basis(origin)
    while True:
        b2 = sample_basis(origin)
        basis = (b1, b2)
        if abs(basis_norm(basis[0],basis[1]))<1e-3:
            break
    
    xs = np.linspace(-1.0, 1.0, resolution)
    ys = np.linspace(-1.0, 1.0, resolution)


    result = np.zeros((resolution, resolution))

    for i,j in itertools.product(range(resolution), repeat=2):
        x,y =xs[i],ys[j]
        model =  mix_point(origin, basis, (x,y))
        r = []
        for k in range(rep):
            r.append(evaluate_model(env_name,model)[0])
        result[i,j]=np.mean(r)
        print(x,y,result[i,j])




    cs = plt.contourf(xs,ys,result,contour_layer)
    cbar = plt.colorbar(cs)
    plt.savefig(save_fig)
    with open(save_fig[:-4]+'.pkl','wb') as w:
        pickle.dump(result,w)



if __name__ == '__main__':
    model_file = '/home/leesy714/source/neuroevo/trained_models/4/FrostbiteNoFrameskip-v4.pt'
    if len(sys.argv)>=2:
        model_file = sys.argv[1]
    env_name = model_file[model_file.rfind('/')+1:-3]
    save_fig=env_name+'.png'
    main(model_file=model_file, env_name=env_name,save_fig=save_fig)
