import matplotlib.pyplot as plt
import numpy as np

Anchors = [16, 32, 64, 128, 256, 512]

RFs = [48, 108, 228, 340, 468, 724]

plt.plot(Anchors, RFs)
plt.plot(Anchors, np.sqrt(Anchors))

plt.show()