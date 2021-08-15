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
permissionnum=0
permission_list=[]
Blist=["SEND_SMS", "READ_SMS", "RECEIVE_SMS", "WRITE_SMS", "READ_SETTINGS", "INTERACT_ACROSS_USERS_FULL", "UPDATE_APP_OPS_STATS", "RECEIVE_MMS", "WRITE_INTERNAL_STORAGE", "READ_INTERNAL_STORAGE", "WRITE_APN_SETTINGS", "ACCESS_CACHE_FILESYSTEM", "WRITE_SECURE_SETTINGS", "ACCESS_MTK_MMHW", "READ_USER_DICTIONARY", "MEDIA_CONTENT_CONTROL", "BROADCAST_SMS", "FOREGROUND_SERVICE"]

def permission_Matrix(permission_name_file, apk_path, permission_mat, Benign):
    appnum = len(os.listdir(apk_path))
    with open(permission_name_file, "r") as f:
        for line in f:
            permission_list = line.split("[")[1].split(']')[0].split(",")

    permission_list = [item.replace("'","").replace(" ","").replace("\"","") for item in permission_list]
    permissionnum = len(permission_list)
    M = np.zeros((appnum, permissionnum + 38))
    apklist = os.listdir(apk_path)
    index = 0
    for apk in apklist:
        filename = apk_path + "/" + apk
        M_res = getAppBaseInfo(filename, index, Benign, M)
        index += 1
    scio.savemat(permission_mat, {'permission': M_res})


def getAppBaseInfo(apkpath,index,Benign,M):
    try:
        output = os.popen("aapt d badging %s" % apkpath).read()
    except FileNotFoundError:
        print("FileNotFoundError")
        os.remove(apkpath)
    except UnicodeDecodeError:
        print("UnicodeDecodeError")
        os.remove(apkpath)
    else:

        flag_1=0
        outList = output.split('\n')
        if(Benign):
            f_num = 86
        else:
            f_num = 67
        for line in outList:
            if line.startswith('uses-permission:') and ("android.permission") in line:
                s_permission = line.split(':')[1].split('\'')[1].split('.')[2].replace(" ", "").upper()
                if(s_permission in permission_list):
                    flag_1 = 1
                    p_index = permission_list.index(s_permission)
                    M[index,p_index]=1
                if(s_permission in Blist):
                    flag_1 = 1
                    p_index = Blist.index(s_permission)
                    M[index, p_index+f_num] = 1
        if(flag_1==0):
            print(apkpath+"have no permission.")
            if(Benign):
                M[index, 66] = 1
            else:
                M[index, 85] = 1
    return M


