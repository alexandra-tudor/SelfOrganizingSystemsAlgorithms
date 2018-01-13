from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot2d(function, bounds):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(bounds[0], bounds[1], 0.1)
    y = np.arange(bounds[0], bounds[1], 0.1)

    X, Y = np.meshgrid(x, y)

    f = function([X, Y])
    # R = np.sqrt(X ** 2 + Y ** 2)
    # f = np.sin(R)
    surf = ax.plot_surface(X, Y, f, cmap=cm.rainbow, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
