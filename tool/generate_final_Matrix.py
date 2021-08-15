import scipy.io as scio
import numpy as np

def get_Matrix(dataFileb, dataFilev, resMatrix):
    permissionb = scio.loadmat(dataFileb)['permission']
    # permissionb=permissionb[:,0:-2]

    permissionv = scio.loadmat(dataFilev)['permission']
    # permissionv=permissionv[:,0:-2]

    d = np.row_stack((np.array(permissionb),np.array(permissionv)))
    res=np.dot(d,d.T)

    scio.savemat(resMatrix, {'permission':res})

