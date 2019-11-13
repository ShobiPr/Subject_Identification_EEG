import scipy.io as sc
import numpy as np

len_sample = 1
full = 7000
len_a = full // len_sample  # 6144 class1
label0 = np.zeros(len_a)  # [0, 0, 0, .... , 0] shape: 7000
label1 = np.ones(len_a)
label2 = np.ones(len_a) * 2
label3 = np.ones(len_a) * 3
label4 = np.ones(len_a) * 4
label5 = np.ones(len_a) * 5
label6 = np.ones(len_a) * 6  # [6, 6, 6, .... , 6] shape: 7000
label7 = np.ones(len_a) * 7  # [7, 7, 7, .... , 7] shape: 7000
label = np.hstack((label0, label1, label2, label3, label4, label5, label6, label7))
label = np.transpose(label)  # (56000, 1)
label.shape = (len(label), 1)  # (56000, 1)

feature = sc.loadmat("rest/EID-M.mat")  # EID_M, with three trials, 21000 samples per sub
all = feature['eeg_close_ubicomp_8sub']  # (168000, 15)
n_fea = 14
all = all[0:21000 * 8, 0:n_fea]  # (168 000, 14) aka (8 sub * (3 trial * 7000 sample), 14 channels)
# all = all.transpose()
# all = all[0:full * 8, 0:n_fea]


test = []
for i in range(all.shape[0]):
    if 35000 <= i < 42000:
        test.append(all[i])

print("test: ", np.shape(test))
np.savetxt('RestData_S2_Sess2.csv', test, delimiter=',')



"""
for i in range(all.shape[0]):
    sampling_count = 0
    _S = 1
    Sess = 1
    sub_samples = []
        if count < 7000:
            sub_samples.append(all[i])
            count = count + 1
            if count == 7000:
                np.savetxt('2darray.csv', arr2D, delimiter=',', fmt='%d')
                np.savetxt("test.csv", sub_samples, delimiter=",")
                count = 0
                sub_samples = []

"""
# print("all.shape[1]", all.shape[1])
# print("i: ", i)
# x = all[:, i]
# print("x: ", np.shape(x))


"""
a1 = all[0:21000]  # select 21 000 samples from 168 000
for i in range(2, 9):
     b = all[168000 * (i-1) : 168000 * i]
     c = b[0 : 21000]
     # print(c.shape)
     a1 = np.vstack((a1, c))
     print(i, a1.shape)

all = a1
print(all.shape)
"""
