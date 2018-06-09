'''
One dimensional random spin chain
'''
import numpy as np
from numpy import linalg
import time
import sys
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import expm, logm
from scipy.sparse import csr_matrix

sx = sparse.csr_matrix(np.array([[0, 1], [1, 0]]) * 0.5)
sy = sparse.csr_matrix(np.array([[0, (-1j)], [(1j), 0]]) * 0.5)
sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]]) * 0.5)

def fkron(a, b, c = 1, d = 1):
    return sparse.kron(sparse.kron(sparse.kron(a, b), c), d)

def m_op(n, op, gs):
    temp = csr_matrix((2**n, 2**n), dtype=complex)
    for p1 in range(n):
        temp += fkron(sparse.identity(2**p1), op, sparse.identity(2**(n-1-p1)))
    return np.conj(gs).dot(temp.dot(gs))/n

def spin_correlation(n, op, gs):
    spin_cor = np.zeros(n)
    temp = fkron(op.dot(op), sparse.identity(2 ** (n - 1)))
    spin_cor[0] = np.real(np.conj(gs).dot(temp.dot(gs)))
    for p1 in range(n-1):
        temp = fkron(op, sparse.identity(2 ** p1), op, sparse.identity(2**(n -2 -p1)))
        spin_cor[p1+1] = np.real(np.conj(gs).dot(temp.dot(gs)))
    return spin_cor

def Ham(n, J, hf, gamma, disorder):
    '''
    It is spin 1/2 chain with open boundary condition
    '''
    Jx, Jy, Jz = J
    hx, hy, hz = hf
    Jx = J[0] + disorder*(np.random.random(n-1)*2 -1 )
    Jy = J[1] + gamma + disorder*(np.random.random(n-1)*2 -1 )
    Jz = J[2] + disorder*(np.random.random(n-1)*2 -1 )
    hz = hf[2] + disorder*(np.random.random(n)*2 -1 )*0

    # np.random.seed(1234)

    print(n, J, hf, gamma, disorder)
    print(Jx)
    print(Jy)
    print(Jz)
    print(hx, hy, hz)

    h = csr_matrix((2**n, 2**n), dtype=complex)
    for p1 in range(n-1):
        h += Jx[p1] * fkron(sparse.identity(2 ** p1), sx, sx, sparse.identity(2**(n-2-p1)))
        h += Jy[p1] * fkron(sparse.identity(2 ** p1), sy, sy, sparse.identity(2 ** (n - 2 - p1)))
        h += Jz[p1] * fkron(sparse.identity(2 ** p1), sz, sz, sparse.identity(2 ** (n - 2 - p1)))
    for p1 in range(n):
        h += hx * fkron(sparse.identity(2**p1), sx, sparse.identity(2**(n-1-p1)))
        h += hy * fkron(sparse.identity(2 ** p1), sy, sparse.identity(2 ** (n - 1 - p1)))
        h += hz[p1] * fkron(sparse.identity(2 ** p1), sz, sparse.identity(2 ** (n - 1 - p1)))

    #v, d = sparse.linalg.eigsh(h, which="SA", k=6)
    h = h.todense()
    v, d = np.linalg.eigh(h)
    # print(v.shape, d[:, 0:10].shape)
    gs = d[:, 0]
    np.savetxt("1d_n_%s_J_%s_hf_%s_energy.csv"%(n,J,hf), (v[np.newaxis]-v[0])/n , delimiter=',')

    return v/n, gs

def cal_operator(n, gs):
    mx = m_op(n, sx, gs)
    my = m_op(n, sy, gs)
    mz = m_op(n, sz, gs)
    ssx = spin_correlation(n, sx, gs)
    ssy = spin_correlation(n, sy, gs)
    ssz = spin_correlation(n, sz, gs)
    return mx, my, mz, ssx, ssy, ssz

def m_hz(n):
    J = [-1, -1, 0]
    m = 100
    listh = np.linspace(0, 2, m)
    data = np.zeros((m, 4))
    for p1 in range(m):
        hf = [0, 0, listh[p1]]
        v, gs = Ham(n, J, hf)
        data[p1, :] = cal_operator(n, gs)
    print(data[:, 3])
    plt.plot(listh, data[:, 3], '-bo')
    plt.grid()
    plt.show()
    plt.close()

def ssplot(n, J, hf, gamma, disorder):
    v, gs = Ham(n, J, hf)
    data = cal_operator(n, gs)
    ssx, ssy, ssz = np.abs(data[3:6])
    f = open("1d_N_%s_J_%s_hf_%s_m_ss.csv" % (n, J, hf), 'wb')
    np.savetxt(f, data[0:3], delimiter=',')
    np.savetxt(f, ssx[np.newaxis], delimiter=',')
    np.savetxt(f, ssy[np.newaxis], delimiter=',')
    np.savetxt(f, ssz[np.newaxis], delimiter=',')
    f.close()
    plt.subplot(2,2,1)
    plt.plot(ssx, '-bo', label='ssx')
    plt.title("1d_N = %s, J = %s \n hf = %s"%(n, J, hf))
    plt.grid()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(ssy, '-bo', label='ssy')
    plt.grid()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(ssz, '-bo', label='ssz')
    plt.grid()
    plt.legend()
    plt.savefig("1d_N_%s_J_%s_hf_%s.png"%(n, J, hf))
    plt.close()
    print(ssx)
    print(ssy)
    print(ssz)

def cvplot(n): # to take all eigenvalues !!!
    J = [1, 1, 0]
    hf = [0, 0, 0]
    energy, gs = Ham(n, J, hf)
    sizeenergy = energy.shape
    m = 100
    listt = np.linspace(0.01, 0.2, m)
    dedt = np.zeros(m)
    print(energy.shape, sizeenergy[0])
    for p1 in range(m):
        for p2 in range(sizeenergy[0]):
            dedt[p1] += energy[p2]*(((energy[p2])/listt[p1]**2)*np.exp(energy[p2]/listt[p1])
            *(1/(1+np.exp(energy[p2]/listt[p1]))**2))
    #
    plt.subplot(1,2,1)
    plt.plot(listt, dedt, 'bo--', label='Cv')
    plt.legend()
    plt.grid()
    plt.title("N = %s, J = %s \n hf = %s" % (n, J, hf))
    plt.xlabel('T')
    plt.subplot(1,2,2)
    z = np.polyfit(np.log(listt[:20]), np.log(dedt[:20]),1)
    plt.loglog(listt[:20], dedt[:20], 'bo--', label='Cv log log %s'%z)
    plt.xlabel('ln(T)')
    plt.legend()
    plt.grid()
    plt.savefig("N_%s_J_%s_hf_%s_cv.png" % (n, J, hf))
    plt.close()

if __name__ == '__main__':
    t1 = time.time()
    n = 4
    J = [1, 1, 1]
    hf = [0, 0, 0]
    gamma = 0
    disorder = 0

    v, gs = Ham(n, J, hf, gamma=0, disorder=0)
    print(v[:10])
    sys.exit()

    t2 = time.time()
    print('time = ', t2-t1, ' s ')
    print('time = ', (t2-t1)/60, ' mins ')
    print(time.asctime(time.localtime(time.time())))
