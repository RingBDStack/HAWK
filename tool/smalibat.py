#encoding : utf-8
import os
import sys
import time

def decompilation(filename,outdir):
    command = "apktool.jar d {0} -o {1}".format(filename, outdir)
    os.system(command)

def docompile(input_apk_path, output_apk_path):
    apklist = os.listdir(input_apk_path)
    time_start = time.time()
    index = 0
    for apk in apklist:
        outdir = output_apk_path + apk
        filename = path + apk
        print(filename)
        print(index)
        decompilation(filename, outdir)
        index += 1
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

#decompile
if __name__ == '__main__':
    index=0
    path = "VirusShare2018\\Virsus2019\\"
    apklist = os.listdir(path)
    time_start = time.time()

    for apk in apklist:
        outdir="out-of-sample-2019\\"+apk
        filename=path+apk
        print(filename)
        print(index)
        decompilation(filename,outdir)
        index+=1
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
