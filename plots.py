import matplotlib.pyplot as plt

EMD_ch7 = [0.7462, 0.7563, 0.7661, 0.7704]
EMD_ch56 = [0.9125, 0.9370, 0.9399, 0.9514]
EMD_ch7_nopre = [0.9260, 0.9408, 0.9326, 0.9407]
EMD_ch56_nopre = [0.9952, 0.9899, 0.9870, 0.9912]


x = [10, 20, 30, 40]
xi = list(range(len(x)))
plt.plot(xi, EMD_ch7, '-.', marker='o', color='C0',  label='7 channels w/ preprocessing')
plt.plot(xi, EMD_ch56, marker='o', color='C0', label='56 channels w/ preprocessing')
plt.plot(xi, EMD_ch7_nopre, '-.', marker='o', color='C3', label='7 channels, no preprocessing')
plt.plot(xi,EMD_ch56_nopre, marker='o', color='C3', label='56 channels, no preprocessing')
plt.xlabel('Instances')
plt.ylabel('Accuracy')
plt.xticks(xi, x)
plt.title('Accuracy for P300-speller [EMD]')
plt.legend()
plt.savefig('accuracy_EMD_P300.eps', format='eps')

plt.show()