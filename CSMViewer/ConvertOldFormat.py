#Programmer: Chris Tralie
#Purpose: To convert json files from the old format to the new format
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import scipy.misc
import time
import matplotlib.pyplot as plt
import json
import argparse
import base64
import fleep

def getAudioExtension(base64str):
    b = base64.b64decode(base64str)
    fout = open("tempaudio", "wb")
    fout.write(b)
    fout.close()
    with open("tempaudio", "rb") as file:
        info = fleep.get(file.read(128))
    return info.extension


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True, help="Path to original JSON file")
    parser.add_argument('--new', type=str, required=True, help="Path to new JSON file")
    opt = parser.parse_args()
    with open(opt.original) as f:
        data = json.load(f)
    print(data.keys())
    ext1 = getAudioExtension(data['file1'])
    data['file1'] = "data:audio/%s;base64, "%ext1 + data['file1']
    ext2 = getAudioExtension(data['file2'])
    data['file2'] = "data:audio/%s;base64, "%ext2 + data['file2']
    for feature in data['FeatureCSMs']:
        for imgtype in data['FeatureCSMs'][feature]:
            s = data['FeatureCSMs'][feature][imgtype]
            if type(s) == float:
                continue
            data['FeatureCSMs'][feature][imgtype] = "data:image/png;base64, " + data['FeatureCSMs'][feature][imgtype]
    
    fout = open(opt.new, "w")
    fout.write(json.dumps(data))
    fout.close()