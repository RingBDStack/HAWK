#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import re
from collections import defaultdict
import operator
import numpy as np
import scipy.io as scio

listres=[]
permission_frequency=defaultdict(int)
Dic = {}

# apk所在目录  恶意： G:\\VirusShare2018\\VirusShare_Android_APK_2018  良性：G:\\Benign_app\\Benign\\finish2020
app_path = "G:\\VirusShare2018\\Virsus2019"
# app_path ="G:\\CICInvesAndMal2017\\Virsus"

# appnum=len(os.listdir(app_path))
permissionnum=0
permission_list=[]

def process_permission(benign_apk_dir, mal_apk_dir, per_name_filename):
    dir_list=[benign_apk_dir, mal_apk_dir]
    for dir in dir_list:
        apklist = os.listdir(app_path)
        index = 0
        for apk in apklist:
            filename = app_path + "/" + apk
            # print(filename)
            print(apk)
            print(index)
            getAppBaseInfo(filename, index)
            print("\n")
            index += 1
    for per in permission_frequency.keys():
        if(permission_frequency[per]>1):
            listres.append(per)
    text_save(per_name_filename, listres)

def text_save(per_name_filename, data):
    file = open(per_name_filename,'a')
    for i in range(len(data)):
        s= str(data[i])+"\n"
        file.write(s)
    file.close()
    print("保存文件成功")

def getAppBaseInfo(apkpath):
    try:
        output = os.popen("aapt d badging %s" % apkpath).read()
    except FileNotFoundError:
        print("FileNotFoundError")
        print(apkpath)
        os.remove(apkpath)

    except UnicodeDecodeError:
        print("UnicodeDecodeError")
        print(apkpath)
        os.remove(apkpath)
    else:
        listres = list()

        outList = output.split('\n')
        listtem=[]
        for line in outList:
            if line.startswith('uses-permission:') and ("android.permission") in line:
                s_permission = line.split(':')[1].split('\'')[1].split('.')[2].replace(" ", "").upper()
                if(s_permission not in listtem):
                    listtem.append(s_permission)

        for s_permission in listtem :
            if (s_permission not in permission_frequency):
                permission_frequency[s_permission] = 1
            else:
                permission_frequency[s_permission] = permission_frequency[s_permission] + 1


if __name__ == "__main__":
    index=0
    apklist = os.listdir(app_path)
    for apk in apklist:
        filename = app_path + "/" + apk
        # print(filename)
        print(apk)
        print(index)
        getAppBaseInfo(filename,index)
        print("\n")
        index+=1
