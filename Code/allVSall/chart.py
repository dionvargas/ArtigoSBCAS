import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Neurons
labels = [64, 128, 256, 512, 1024, 2048, 4096]

# Tests
elu_mean = [73, 78.8, 78.2, 84.2, 82.6, 82.4, 82]
relu_mean = [72.4, 77.2, 82.2, 80.4, 83.2, 83.2, 81.4]
selu_mean = [74.8, 77.6, 81, 81.8, 78.2, 84, 81.2]
sigmoid_mean = [63.6, 63, 66, 68.4, 71.2, 78, 74.8]
softplus_mean = [72.2, 74, 79.4, 76.6, 80.8, 80, 83.6]
tanh_mean = [60.2, 64, 63.8, 73.2, 69.4, 74, 75.4]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2.5*width, tanh_mean, width, color="#000000", label='Tangente Hiperbólica')
rects2 = ax.bar(x - 1.5*width, sigmoid_mean, width, color="#222222", label='Sigmóide')
rects3 = ax.bar(x - 0.5*width, softplus_mean, width, color="#444444", label='Softplus')
rects4 = ax.bar(x + 0.5*width, elu_mean, width, color="#666666", label='ELU')
rects5 = ax.bar(x + 1.5*width, selu_mean, width, color="#888888", label='SELU')
rects6 = ax.bar(x + 2.5*width, relu_mean, width, color="#AAAAAA", label='RELU')

ax.set_ylabel('Acurácia (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Neurônios na camada intermediária')
ax.legend(loc='lower right')

fig.tight_layout()

plt.show()
