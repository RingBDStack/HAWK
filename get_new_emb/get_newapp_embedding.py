import numpy as np
import heapq
import scipy.io as scio

#neighbours
num=6

def sumlist(ls):
    total = 0
    for ele in range(0, len(ls)):
        total = total + ls[ele]
    return total


def get_neibor_index(m):
    max_number = heapq.nlargest(num, m)
    max_index = []
    for t in max_number:
        index = m.index(t)
        max_index.append(index)
        m[index] = 0
    # print(max_number)
    # print(max_index)
    return  max_index,max_number


def aggregation(M,m,index):
    # for i in len(m):
    #     m[i]=2*m[i]/(M[index][index])
    nerlist,nervaluelist = get_neibor_index(m)
    # print(nerlist)
    # print(nervaluelist)
    # 权重列表 numpy
    powerlist = np.array(nervaluelist)/sumlist(nervaluelist)
    # print(powerlist)

    temlist=[]
    d = []
    res=np.zeros((1,128))
    j=0
    for i in nerlist:

        res += M[i]*powerlist[j]
        # res += M[i] * 0.2
        j+=1

    # print(res)
    return res


M=np.load('getembeding\\featts.npy')


A_extern = scio.loadmat('new_adj\\nebor_adj_CIC2019\\new_Res_extern_package.mat')['permission']
A_interface = scio.loadmat('new_adj\\nebor_adj_CIC2019\\new_Res_Interface.mat')['permission']
A_per = scio.loadmat('new_adj\\nebor_adj_CIC2019\\new_Res_permission.mat')['permission']
A_so = scio.loadmat('new_adj\\nebor_adj_CIC2019\\new_Res_so.mat')['permission']
A_api= scio.loadmat('new_adj\\nebor_adj_CIC2019\\new_Res_APICIC2019.mat')['permission']


res=np.zeros((len(A_per)-3, 128))

for i in range(0,len(A_per)-3):
    l_e = A_extern[i].tolist()
    l_n_e = aggregation(M, l_e,i)
    # print(l_n_e)
    # res[i]=l_n_e
    l_i = A_interface[i].tolist()
    l_n_i = aggregation(M, l_i, i)
    # res[i] = l_n_i
    l_p = A_per[i].tolist()
    l_n_p = aggregation(M, l_p,i)
    l_s = A_so[i].tolist()
    l_n_s= aggregation(M, l_s,i)
    l_a = A_api[i].tolist()
    l_n_a = aggregation(M, l_a, i)

    # res[i] = (l_n_e+l_n_i+l_n_p+l_n_a)/4
    res[i] = l_n_e*0.12+ l_n_i *0.1 + l_n_p *0.57 + 0.17*l_n_a + l_n_s *0.1
    # res[i] =l_n_a
print(res.shape)

#adjust
# listd=[]
#
# for i in range(249,251):
#     listd.append(i)
# res=np.delete(res,listd,axis=0)
#
# print(np.argwhere(np.isnan(res)))
# print(res)
# print(res.shape)


np.save('new_app_emb\\new_app_emb_CIC2019\\new_app_emb_all_6.npz',res)