import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import scipy as sp

def plot_disp(ax, w, y1, y2, y3, y1_d, y2_d, y3_d, first_floor=True, second_floor=True, third_floor=True, damped=False):
    if damped:
        if first_floor:
            ax.plot(w, y1, label='1st floor')
            ax.plot(w, y1_d, label='1st floor damped')
        if second_floor:
            ax.plot(w, y2, label='2nd floor')
            ax.plot(w, y2_d, label='2nd floor damped')
        if third_floor:
            ax.plot(w, y3, label='3rd floor')
            ax.plot(w, y3_d, label='3rd floor damped')

    else:
        if first_floor:
            ax.plot(w, y1, label='1st floor')
        if second_floor:
            ax.plot(w, y2, label='2nd floor')
        if third_floor:
            ax.plot(w, y3, label='3rd floor')
    
    ax.legend()
    return ax

def mass_matrix(m, md, N, N_dampers):
    M = np.zeros((N + N_dampers, N + N_dampers), dtype=np.float64) # create the mass matrix
    # initialize the mass matrix
    for i in range(3):
        M[i, i] = m
    for i in range(3, N + N_dampers):
        M[i][i] = md

    return np.mat(M)

def stiffness_matrix(k, kd, N, N_dampers):
    K = np.zeros((N + N_dampers, N + N_dampers), dtype=np.float64) # create the mass matrix
    if N_dampers == 0:
        K[0][0] = 4 * k
    else:
        K[0][0] = 4 * k + np.sum(kd)
    K[0][1] = -2 * k
    K[1][0] = -2 * k
    K[1][1] = 4 * k
    K[1][2] = -2 * k
    K[2][1] = -2 * k
    K[2][2] = 2 * k

    for i in range(N_dampers):
        K[i + N][0] = -kd[i]
        K[0][i + N] = -kd[i]
        K[i + N][i + N] = kd[i]

    return np.mat(K)

def damping_matrix(l, ld, N, N_dampers):
    L = np.zeros((N + N_dampers, N + N_dampers), dtype=np.float64) # create the mass matrix
    if N_dampers == 0:
        L[0][0] = 3 * l
    else:
        L[0][0] = 3 * l + np.sum(ld)
    L[0][1] = -l
    L[1][0] = -l
    L[1][1] = 3 * l
    L[1][2] = - l
    L[2][1] = - l
    L[2][2] = 2 * l

    for i in range(N_dampers):
        L[i + N][0] = -ld[i]
        L[0][i + N] = -ld[i]
        L[i + N][i + N] = ld[i]

    return np.mat(L)


m = 1.83
L = 0.2 
N = 3
b = 0.08
E = 210E9
d = 0.001
I = b*d*d*d/12
k = (12*E*I)/(L*L*L)
L = 2e-3
l = 2

Q = 500 # number of points to plot the functions
N_dampers = 1000 # Number of dampers
W = 130 # Upper limit of frequency sweep

M = mass_matrix(m, 0, N, 0)
K = stiffness_matrix(k, 0, N, 0)
L = damping_matrix(l, 0, N, 0)

D, V = linalg.eig(K, M, right=True, left=False)

freqs = np.sqrt(np.abs(D))
hertz = freqs / (2 * np.pi)

damped_freqs = np.array([W * (i + 1) / (N_dampers + 1) for i in range(N_dampers)])
md = 0.15 / N_dampers * 10
ld = np.ones(N_dampers)
kd = md * damped_freqs * damped_freqs
# initialize the mass matrix
Md = mass_matrix(m, md, N, N_dampers)
Kd = stiffness_matrix(k, kd, N, N_dampers)
Ld = damping_matrix(l, ld, N, N_dampers)
F = np.zeros(N + N_dampers)
F[0] = 1
F = np.mat(F).T

w = np.linspace(0, W, Q)
disp = np.zeros((Q, N))
disp_damped = np.zeros((Q, N + N_dampers))

for i in range(Q):
    disp[i] = np.reshape((np.abs(np.linalg.solve(K - ((w[i] * w[i]) * M) + 1j * w[i] * L, F[:N]))), (N, ))
    disp_damped[i] = np.reshape(np.abs(np.linalg.solve(Kd - ((w[i] * w[i]) * Md) + 1j * w[i] * Ld, F)), (N + N_dampers, ))
    '''B = K - ((w[i] * w[i]) * M) - 
    Bd = Kd - ((w[i] * w[i]) * Md)

    disp[i] = np.reshape(abs((B.I).dot(F[:N])), (N, ))
    disp_damped[i] = np.reshape(abs((Bd.I).dot(F)), (N + N_dampers))
'''

y1 = disp[:, 2] # 1st floor
y2 = disp[:, 1] # 2nd floor
y3 = disp[:, 0] # 3rd floor

y1_d = disp_damped[:, 2] # 1st floor
y2_d = disp_damped[:, 1] # 2nd floor
y3_d = disp_damped[:, 0] # 3rd floor

fig1 = plt.figure()
ax1 = fig1.subplots()
#ax1.set_ylim((0, L))
ax1 = plot_disp(ax1, w, y1, y2, y3, y1_d, y2_d, y3_d, first_floor=True, second_floor=False, third_floor=False, damped=True)

fig2 = plt.figure()
ax2 = fig2.subplots()
#ax2.set_ylim((0, L))
ax2 = plot_disp(ax2, w, y1, y2, y3, y1_d, y2_d, y3_d, first_floor=False, second_floor=True, third_floor=False, damped=True)

fig3 = plt.figure()
ax3 = fig3.subplots()
#ax3.set_ylim((0, L))
ax3 = plot_disp(ax3, w, y1, y2, y3, y1_d, y2_d, y3_d, first_floor=False, second_floor=False, third_floor=True, damped=True)
plt.show()
