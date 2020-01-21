import matplotlib.pyplot as plt

lin_SVM = [0.94, 0.99, 0.99]
RF = [0.79, 0.85, 0.90, ]
DT = [0.87, 0.92, 0.94]
k_NN = [0.92, 0.97, 0.97]
NB = [0.83, 0.79, 0.80]

x = [7, 32, 56]
xi = list(range(len(x)))
plt.plot(xi, lin_SVM, marker='o', color='C1',  label='linear SVM')
plt.plot(xi, RF, marker='o', color='C2', label='Random forest')
plt.plot(xi, k_NN, marker='o', color='C3', label='Decision tree')
plt.plot(xi, DT, marker='o', color='C4', label='k-NN')
plt.plot(xi, NB, marker='o', color='C5', label='naive Bayes')
plt.xlabel('Channels')
plt.ylabel('Accuracy')
plt.xticks(xi, x)
# plt.title('Accuracy for P300-speller [EMD]')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid()
plt.savefig('accuracy_classifiers.png', format='eps')

plt.show()