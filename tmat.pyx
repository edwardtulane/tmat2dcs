# july 2018: this is the version that is burning
cimport cython
from cypolyscat cimport wig3j, wig3j0, clebg, coulcc

from sympy.physics.wigner import clebsch_gordan, wigner_3j
from scipy.special import loggamma

from numpy import arange, zeros, pi, zeros_like
import numpy as np
from pandas import Series, MultiIndex
from libc.math cimport sqrt, signbit#, exp#, conj

cdef extern from "complex.h" nogil:
    double complex cexp(double complex)

# cdef double[:] gam = loggamma(arange(400) + 1 ).real
cdef double[:] clg = zeros(200)

cdef:
    int i
    int lmin
    int l1, m1, l2, m2, L
    int m, mp
    int lmax = 4 
    int lmax2= 2 * lmax
    double w, w2, fac

# for i in range(lmax2):
#     print(gam[i])

@cython.boundscheck(False)
def AL(t_ser):
    cdef int i, L
    cdef int l1,l1p, l2, l2p, m1, m2
    cdef int lmin, lmax, lminmin
    cdef double eng, p
    cdef double e_ha = 27.211386
#     cdef double pi = pi
    cdef double complex a_ll, a_lamlam, A_incr
    cdef double[:] clg = zeros(200)
    cdef int[:] lev1 = zeros(len(t_ser), dtype=np.int32)
    cdef int[:] lev2 = zeros_like(lev1)
    cdef int[:] lev3 = zeros_like(lev1)
    cdef double complex[:] t_val = t_ser.values
    cdef int k, kp
    cdef int ldim = t_ser.index.get_level_values(1).max()
    cdef int err
    
    cdef double complex[:]  AL = zeros(2*ldim+1,
                                       dtype=np.complex128)
    cdef double complex[:] AL_out = zeros_like(AL)
    cdef double complex[:] dummy = np.zeros(ldim+1,
                                       dtype=np.complex128)
    cdef double complex[:] coul_phs = np.zeros_like(dummy)
    
    eng = t_ser.index[0][0]
    lev1 = t_ser.index.get_level_values(1).values.astype(np.int32)
    lev2 = t_ser.index.get_level_values(2).values.astype(np.int32)
    lev3 = t_ser.index.get_level_values(3).values.astype(np.int32)
#     print(eng)
    p = sqrt(2 * eng / e_ha)
    
    err = coulcc(1j, -1/p, 0, ldim+1, dummy, dummy, dummy, dummy,
                 coul_phs, 1, 0)
    
#     for i in range(ldim+1):
#         print(coul_phs[i], cexp(1j*coul_phs[i]))
    for k in range(len(t_ser)):
        l1, l1p, m1 = lev1[k], lev2[k], lev3[k]
        
        a_ll = sqrt(4 * pi * (l1p + l1p + 1)) * t_val[k] / p
        a_ll = a_ll * (1j)**(l1p-l1)
        a_ll = a_ll * cexp(1j*(coul_phs[l1]+coul_phs[l1p]))
        
        for k_p in range(len(t_ser)):
            l2, l2p, m2 = lev1[k_p], lev2[k_p], lev3[k_p]
            a_lamlam  = sqrt(4 * pi * (l2p + l2p+ 1)) * t_val[k_p] / p
            a_lamlam = a_lamlam * (1j)**(l2p-l2) 
            a_lamlam = a_lamlam * cexp(1j*(coul_phs[l2]+coul_phs[l2p]))
            
            A_incr = a_ll * a_lamlam.conjugate()
#             print(A_incr.imag)
            
            AL[:] = 1.
            lmax = min(l1+l2, l1p+l2p) +1
            lminmin = 0
            
            lmin = clebg(        l1, l2,
                         m2-m1,  m1, clg)
            lminmin = max(lmin, lminmin)
            for i in range(l1+l2+1 - lmin):
                L   = lmin + i
                AL[L] = AL[L] * clg[i]
            lmin = clebg(        l1, l2,
                             0,   0, clg) 
            lminmin = max(lmin, lminmin)
            for i in range(l1+l2+1 - lmin):
                L   = lmin + i
                AL[L] = AL[L] * clg[i]
            ###
            ###
            lmin = clebg(        l1p, l2p,
                         m2-m1,  m1, clg) 
            lminmin = max(lmin, lminmin)
            for i in range(l1p+l2p+1 - lmin):
                L   = lmin + i
                AL[L] = AL[L] * clg[i]
            lmin = clebg(        l1p, l2p,
                              0,   0, clg) 
            lminmin = max(lmin, lminmin)
            for i in range(l1p+l2p+1 - lmin):
                L   = lmin + i
                AL[L] = AL[L] * clg[i] 
                
#             print(lminmin, lmax)
            if lmax < len(AL): AL[lmax:] = 0.
            AL[:lminmin] = 0.
            for i in range(len(AL)):
                AL[i] = AL[i] * sqrt((l1+l1+1)/(l2+l2+1)) * (i+i+1) / (l2p+l2p+1)
                AL[i] = AL[i] * A_incr
                AL_out[i] = AL_out[i] + AL[i]
#             print(lmax, np.asarray(AL)[:4])
#             AL_out = AL_out + AL
    
    return np.asarray(AL_out)
            
#   
