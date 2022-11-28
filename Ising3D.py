from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt


def initialize(N):
    """Create an N by N by N 3D-lattace with random +1 or -1 states """
    spin = np.random.choice([1, -1], size=(N, N, N))
    return spin


def magnetization(spin):
    """Calculate the Magnetization of the spin system"""
    mag = abs(np.sum(spin))
    return mag


def meanEnergy(spin, N):
    """Calculate the mean energy of the spin system"""
    energy = 0
    for i in range(len(spin)):
        for j in range(len(spin)):
            for k in range(len(spin)):
                s = spin[i, j, k]
                energy += -s * neighbors(spin, N, i, j, k)
    return energy / 8.


def neighbors(spin, N, x, y, z):
    """Find 6 neighboring lattices of the randomly picked lattice using modulus calculation"""
    left = spin[x, (y - 1) % N, z]
    right = spin[x, (y + 1) % N, z]
    top = spin[(x - 1) % N, y, z]
    bottom = spin[(x + 1) % N, y, z]
    front = spin[x, y, (z - 1) % N]
    back = spin[x, y, (z + 1) % N]

    totalSpin = left + right + top + bottom + front + back

    return totalSpin


def mcMove(spin_states, beta, N):
    """Monte Carlo move using Metropolis algorithm"""
    for x in range(len(spin_states)):
        for y in range(len(spin_states)):
            for z in range(len(spin_states)):
                s = spin_states[x, y, z]
                cost = 2 * s * neighbors(spin_states, N, x, y, z)
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                spin_states[x, y, z] = s
    return spin_states


nt = 80  # Temperature Points
N = 5  # Dimension of the lattice system
eqSteps = 2000       #  number of MC sweeps for equilibration
mcSteps = 1250       #  number of MC sweeps for calculation

spin = initialize(N)
m = magnetization(spin)
print ("m =", m)

T = np.linspace(1,7,nt)
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2 = 1.0/(mcSteps*N*N*N), 1.0/(mcSteps*mcSteps*N*N*N)

for tt in range(nt):
    E1 = M1 = E2 = M2 = 0
    spin = initialize(N)
    iT = 1.0 / T[tt]
    iT2 = iT * iT  # inverse temperature for beta

    for i in range(eqSteps):
        mcMove(spin,iT,N)

    for i in range(mcSteps):
        mcMove(spin,iT,N)
        Ene = meanEnergy(spin,N)
        Mag = magnetization(spin)

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag * Mag
        E2 = E2 + Ene * Ene

    E[tt] = n1 * E1
    M[tt] = n1 * M1
    C[tt] = (n1 * E2 - n2 * E1 * E1) * iT2
    X[tt] = (n1 * M2 - n2 * M1 * M1) * iT


print(M)

"""
plt.plot(T,M)
plt.xlabel('Temperature')
plt.ylabel('Magnetisation')
plt.title('3D Ising Model Magnetization')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(T,C,'+')
"""

fig, axs = plt.subplots(2,2)
fig.suptitle('3D Ising Model')
axs[0,0].plot(T,M)
axs[0,0].set_xlabel('Temperature')
axs[0,0].set_ylabel('Magnetization')
axs[0,0].grid(True)
#axs[0,0].set_title('3D Ising Model Magnetization')

axs[1,0].plot(T,E)
axs[1,0].set_xlabel('Temperature')
axs[1,0].set_ylabel('Mean Energy')
axs[1,0].grid(True)
#axs[1,0].set_title('3D Ising Model Mean Energy')

axs[1,1].plot(T,C)
axs[1,1].set_xlabel('Temperature')
axs[1,1].set_ylabel('Specific Heat Capacity')
axs[1,1].grid(True)
#axs[1,1].set_title('3D Ising Model Specific Heat Capacity')

axs[0,1].plot(T,X)
axs[0,1].set_xlabel('Temperature')
axs[0,1].set_ylabel('Susceptibility')
axs[0,1].grid(True)
#axs[0,1].set_title('3D Ising Model Susceptibility')

plt.tight_layout()
plt.show()

