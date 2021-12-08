import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector

cdef struct Debug_result:
    int total
    vector[int] path

cdef int get_neighbor_sum(cnp.int8_t[:,:] arr, int size_n, int I, int J, int pos_i, int pos_j) nogil:
    cdef int i, j, total
    total = 0
    for i in range(pos_i-size_n, pos_i+size_n+1):
        for j in range(pos_j-size_n, pos_j+size_n+1):
            total += arr[i%I, j%J]
    return total

cdef cnp.int8_t[:,:] _iterate(cnp.int8_t[:,:] arr, int min_survive, int max_survive, int min_born, int max_born, int size_n):
    cdef int i, j, I, J
    I = arr.shape[0]
    J = arr.shape[1]

    cdef cnp.int8_t[:,:] arr_c = np.zeros_like(arr)
    cdef cnp.int8_t neighbs = 0

    for i in range(I):
        for j in range(J):
            neighbs = get_neighbor_sum(arr, size_n, I, J, i, j)
            neighbs = neighbs - arr[i,j]
            if arr[i,j] == 1 and neighbs >= min_survive and neighbs <= max_survive:
                arr_c[i,j] = 1
            elif arr[i,j] == 0 and neighbs >= min_born and neighbs <= max_born:
                arr_c[i,j] = 1
            else:
                arr_c[i,j] = 0

    return arr_c


def iterate(cnp.int8_t[:,:] arr, int min_survive, int max_survive, int min_born, int max_born, int size_n):
    return _iterate(arr, min_survive, max_survive, min_born, max_born, size_n)