"""van-der-pol.py: Plot Van der Pol Equation"""
__author__      = "Ryota Ishidu"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import cm

mu = 34.2
beta = -0.263
resolution = 30

def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 + 2 * beta * x[0] - 1.0) * x[1] - x[0]
    res = np.array([nx0, nx1])
    return res

def plot_phase_space(filename=None):
    t_max = 5000.0
    ts = np.linspace(0.0, t_max, int(t_max) * 1000)
    xs = odeint(van_der_pol_oscillator_deriv, [0,0.1], ts)
    xs_peak1 = np.amax([np.amax(xs[:,0]),np.fabs(np.amin(xs[:,0]))]) + 1
    xs_peak2= np.amax([np.amax(xs[:,1]),np.fabs(np.amin(xs[:,1]))]) + 1
    X, Y = np.meshgrid(np.linspace(-xs_peak1, xs_peak1, resolution), np.linspace(-xs_peak2, xs_peak2, resolution))
    U = Y
    V = -mu * (X ** 2.0 + 2 * beta * X - 1.0) * Y - X
    C = np.hypot(U, V)
    plt.figure()
    U_norm = U / np.sqrt(U**2 + V**2)
    V_norm = V / np.sqrt(U**2 + V**2)
    Q = plt.quiver(X, Y, U_norm, V_norm, C, units='xy',cmap=cm.gnuplot)
    plt.colorbar()
    plt.plot(xs[:,0], xs[:,1])
    xs = odeint(van_der_pol_oscillator_deriv, [-3.0, -30.0], ts)
    plt.plot(xs[:,0], xs[:,1])
    xs = odeint(van_der_pol_oscillator_deriv, [3.0, 40.0], ts)
    plt.plot(xs[:,0], xs[:,1])
    plt.xlim(-xs_peak1, xs_peak1)
    plt.ylim(-xs_peak2, xs_peak2)
    plt.xlabel('$q$')
    plt.ylabel('$\dot{q}$')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plot_vib(filename=None):
    t_max = 50.0
    ts = np.linspace(0.0, t_max, int(t_max) * 10)
    plt.figure()
    xs = odeint(van_der_pol_oscillator_deriv, [0,0.1], ts)
    plt.plot(ts, xs[:,0])
    plt.ylabel('$q$')
    plt.xlabel('$\tau$')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':
    plot_vib("vib.pdf")
    plot_phase_space("phase_space.pdf")