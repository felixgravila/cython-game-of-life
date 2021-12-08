import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector

cdef int get_neighbor_sum(cnp.int8_t[:,:] arr, int size_n, int I, int J, int pos_i, int pos_j) nogil:
    cdef int i, j, total
    total = 0
    for i in range(pos_i-size_n, pos_i+size_n+1):
        for j in range(pos_j-size_n, pos_j+size_n+1):
            total += arr[i%I, j%J]
    return total

def iterate(cnp.int8_t[:,:] arr, int min_survive, int max_survive, int min_born, int max_born, int size_n):
    cdef int i, j, I, J
    I = arr.shape[0]
    J = arr.shape[1]

    cdef cnp.int8_t[:,:] arr_c = np.zeros_like(arr)
    cdef cnp.int8_t neighbs = 0

    cdef int survived = 0
    cdef int died = 0
    cdef int born = 0

    for i in range(I):
        for j in range(J):
            neighbs = get_neighbor_sum(arr, size_n, I, J, i, j)
            neighbs = neighbs - arr[i,j]
            if arr[i,j] == 1:
                if neighbs >= min_survive and neighbs <= max_survive:
                    arr_c[i,j] = 1
                    survived += 1
                else:
                    arr_c[i,j] = 0
                    died += 1
            elif arr[i,j] == 0 and neighbs >= min_born and neighbs <= max_born:
                arr_c[i,j] = 1
                born += 1
            else:
                arr_c[i,j] = 0

    return {
        "result": arr_c,
        "survived": survived,
        "died": died,
        "born": born
    }
