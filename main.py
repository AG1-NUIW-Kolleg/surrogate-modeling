from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from smt.surrogate_models import RBF

xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

sm = RBF(d0=5)
sm.set_training_values(xt, yt)
sm.train()

num = 100
x = np.linspace(0.0, 4.0, num)
y = sm.predict_values(x)

plt.plot(xt, yt, 'o')
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Training data', 'Prediction'])
plt.savefig('out')
