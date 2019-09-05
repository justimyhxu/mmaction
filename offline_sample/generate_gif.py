import imageio
import os

path = '../data/epic-kitchen/train.txt'
path_list = [x.spilit(',')[1:]  for x in open(path)]
