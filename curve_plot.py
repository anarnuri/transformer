from numba import jit, types, prange, int32, int64, typed
import numpy as np
import math 
from typing import Callable, Dict
import scipy.spatial.distance as sciDist
import copy

@jit(nopython=True)
def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a

@jit(nopython=True)
def _fast_frechet_matrix(dist: np.ndarray,
                         diag: np.ndarray,
                         p: np.ndarray,
                         q: np.ndarray) -> np.ndarray:

    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist

@jit(nopython=True)
def _fast_distance_matrix(p, q, diag, dist_func):
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # Create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # Fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist

@jit(nopython=True)
def _bresenham_pairs(x0: int, y0: int,
                     x1: int, y1: int) -> np.ndarray:

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs

@jit(nopython=True)
def _fdfd_matrix(p: np.ndarray,
                 q: np.ndarray,
                 dist_func: Callable[[np.array, np.array], float]) -> float:
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca

class FastDiscreteFrechetMatrix(object):

    def __init__(self, dist_func):
        self.times = []
        self.dist_func = dist_func
        self.ca = np.zeros((1, 1))
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        ca = _fdfd_matrix(p, q, self.dist_func)
        self.ca = ca
        return ca[p.shape[0]-1, q.shape[0]-1]


@jit(nopython=True, fastmath=True)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))

@jit(nopython=True, fastmath=True)
def haversine(p: np.ndarray,
              q: np.ndarray) -> float:
    d = q - p
    a = math.sin(d[0]/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) \
        * math.sin(d[1]/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c

def is_simple(tpMat):
    # simple: i.e., this can be solved with chain tree and intersection of circles/arc sects.
    # find links and set up joint table.
    # actuator will be noted with negative value.
    fixParam = [1, 3]
    jT = {}
    fixJ = []
    kkcJ = []
    chain = {}

    # step 1, initialize, set all joints and links to unknown (0 in jointTable) and jointLinkTable.
    for i in range(tpMat.shape[0]):
        jT[i] = 0
        chain[i] = {'from': None, 'next': []}

    # step 2, set all ground joints to known (1 to be known)
    for i in range(tpMat.shape[0]):
        if tpMat[i, i] in fixParam:
            jT[i] = 1
            fixJ.append(i)
            kkcJ.append((i, 'fixed', i))
            chain[i]['from'] = i

    # step 3, set joints in the kinematic chain to known
    pivotJ = fixJ
    while True:
        prevCtr = len(kkcJ)
        newJ = []
        for i in pivotJ:
            for j in range(tpMat.shape[1]):
                if tpMat[i, j] < 0 and jT[j] == 0:
                    jT[j] = 1
                    newJ.append(j)
                    kkcJ.append((j, 'chain', i))
                    chain[i]['next'].append(j)
                    chain[j] = {'from': i, 'next': []}

        if len(kkcJ) == prevCtr:
            break
        else:
            pivotJ = newJ  # This is based on the idea of tree node expansion

    if len(kkcJ) == tpMat.shape[0]:
        print(jT)
        return kkcJ, chain, True

    # step 4, set joints that can be solved through the intersection of circles to known
    while True:
        foundNew = False
        for k in jT:
            if jT[k] == 0:
                for i, _, _ in kkcJ:
                    for j, _, _ in kkcJ:
                        if i < j and tpMat[i, k] * tpMat[j, k] != 0 and not foundNew:
                            foundNew = True
                            jT[k] = 1
                            kkcJ.append((k, 'arcSect', (i, j)))
        if not foundNew:
            break

    # return chain and isSimple (meaning you can solve this with direct chain)
    return kkcJ, chain, len(kkcJ) == tpMat.shape[0]


# Direct kinematics:
def compute_chain_by_step(step, rMat, pos_init, unitConvert=np.pi / 180):
    pos_new = copy.copy(pos_init)
    dest, _, root = step
    pos_new[dest, 2] = rMat[root, dest] * unitConvert + pos_new[root, 2]
    c = np.cos(pos_new[dest, 2])
    s = np.sin(pos_new[dest, 2])
    posVect = pos_init[dest, 0:2] - pos_init[root, 0:2]
    pos_new[dest, 0] = posVect[0] * c - posVect[1] * s + pos_new[root, 0]
    pos_new[dest, 1] = posVect[0] * s + posVect[1] * c + pos_new[root, 1]
    return pos_new


# Inverse kinematics:
def compute_arc_sect_by_step(step, posOld, distMat, Ppp=None, threshold=0.1, timefactor=0.1):
    global is_impossible

    threshold = np.max(distMat) * threshold
    posNew = copy.copy(posOld)
    ptSect, _, centers = step
    cntr1, cntr2 = centers
    r1s = distMat[cntr1, ptSect]
    r2s = distMat[cntr2, ptSect]
    if r1s < 10e-12:
        posNew[ptSect, 0:2] = posOld[cntr1, 0:2]
    elif r2s < 10e-12:
        posNew[ptSect, 0:2] = posOld[cntr2, 0:2]
    else:
        ptOld = posOld[ptSect, 0:2]
        ptCen1 = posOld[cntr1, 0:2]
        ptCen2 = posOld[cntr2, 0:2]
        d12 = np.linalg.norm(ptCen1 - ptCen2)
        if d12 > r1s + r2s or d12 < np.absolute(r1s - r2s):
            # print('impossible \n')
            return posOld, False
        elif d12 < 10e-12:  # incidence joint
            # print('illegal \n')
            return posOld, False
        else:
            # print('legal')
            # a means the LENGTH from cntr1 to the mid point between two intersection points.
            # h means the LENGTH from the mid point to either of the two intersection points.
            # v means the Vector from cntr1 to the mid point between two intersection points.
            # vT 90 deg rotation of v
            a = (r1s ** 2 - r2s ** 2 + d12 ** 2) / (d12 * 2)
            h = np.sqrt(r1s ** 2 - a ** 2)
            v = ptCen2 - ptCen1
            vT = np.array([-v[1], v[0]])
            r1 = a / d12
            r2 = h / d12
            ptMid = ptCen1 + v * r1
            sol1 = ptMid + vT * r2
            sol2 = ptMid - vT * r2
            # print(ptOld, sol1, np.linalg.norm(sol1 - ptOld), sol2, np.linalg.norm(sol2 - ptOld))
            # compute ref point
            refPoint = ptOld
            if type(Ppp) != type(None):
                refPoint += (Ppp[ptSect, 0:2] - ptOld) * timefactor
            if np.linalg.norm(sol1 - refPoint) > np.linalg.norm(sol2 - refPoint):
                posNew[ptSect, 0:2] = sol2
                # print('sol2 selected \n')
            else:
                posNew[ptSect, 0:2] = sol1
            # detect if there's an abrupt change:
            if np.max(np.linalg.norm(posNew - posOld, axis=1)) > threshold:
                # print('thresholded', posNew, posOld)
                return posOld, False

        return posNew, True


# Basic data for computing a mechanism.
def compute_dist_mat(tpMat, pos):
    cdist = sciDist.cdist
    tpMat = copy.copy(np.absolute(tpMat))
    tpMat[list(range(0, tpMat.shape[0])), list(range(0, tpMat.shape[1]))] = 0
    return np.multiply(cdist(pos[:, 0:2], pos[:, 0:2]), tpMat)


def compute_curve_simple(tpMat, pos_init, rMat, distMat=None, maxTicks=360, baseSpeed=1):
    # preps
    kkcJ, chain, isReallySimple = is_simple(tpMat)
    if distMat is None:
        distMat = compute_dist_mat(tpMat, pos_init)
    poses1 = np.zeros((pos_init.shape[0], maxTicks, 3))
    poses2 = np.zeros((pos_init.shape[0], maxTicks, 3))
    # Set first tick
    poses1[:, 0, 0:pos_init.shape[1]] = pos_init
    poses2[:, 0, 0:pos_init.shape[1]] = pos_init
    # Compute others by step.
    meetAnEnd = False
    meetTwoEnds = False
    tick = 0
    offset = 0
    while not meetTwoEnds:
        # get tick
        tick += 1
        if tick + offset >= maxTicks:
            poses = poses1  # never flips
            break
        # decide which direction to compute.
        if not meetAnEnd:
            time = 1 * baseSpeed
            pos = poses1[:, tick - 1, :]
            posp = poses1[:, tick - 1, :]
            if tick - 2 < 0:
                Ppp = None
            else:
                Ppp = poses1[:, tick - 2, :]
        else:
            time = 1 * baseSpeed * (-1)
            pos = poses2[:, tick - 1, :]
            posp = poses2[:, tick - 1, :]
            if tick - 2 < 0:
                Ppp = None
            else:
                Ppp = poses2[:, tick - 2, :]
        # step-wise switch solution
        for step in kkcJ:
            if step[1] == 'fixed':
                pos[step[0], 0:2] = pos_init[step[0], :]
                notMeetEnd = True
            elif step[1] == 'chain':
                pos = compute_chain_by_step(step, rMat * time, posp[:, :])
                notMeetEnd = True
            elif step[1] == 'arcSect':
                if not meetAnEnd:
                    pos[step[0], :] = poses1[step[0], tick - 1, :]
                else:
                    pos[step[0], :] = poses2[step[0], tick - 1, :]
                pos, notMeetEnd = compute_arc_sect_by_step(step, pos, distMat, Ppp)
                if notMeetEnd and not meetAnEnd:  # never met an end -> to poses1
                    poses1[:, tick, :] = pos
                elif not notMeetEnd and not meetAnEnd:  # meet end like right now. This tick is not a solution.
                    poses1 = poses1[:, 0:tick, :]
                    offset = tick - 1  # the number of valid ticks
                    meetAnEnd = True
                    tick = 0  # reset tick to zero for time pos
                    break
                elif notMeetEnd and meetAnEnd:  # met an end. -> to poses2
                    poses2[:, tick, :] = pos
                else:  # not notMeetEnd and meetAnEnd. met both ends right now, this tick is not a solution.
                    poses2 = poses2[:, 1:tick, :]  # poses2 (<-) poses1(->). First pose of poses2 is pos_init.
                    poses2 = np.flip(poses2, axis=1)  # make poses2 (->)
                    poses = np.concatenate([poses2, poses1], axis=1)
                    meetTwoEnds = True
                    break
            else:
                print('Unexpected step:, ' + step[1])
                break
    return poses, meetAnEnd, isReallySimple


def get_pca_inclination(qx, qy, ax=None, label=''):
    """ 
        Performs the PCA
        Return transformation matrix
    """
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)

    covar = np.array([[covar_xx, covar_xy], [covar_yx, covar_yy]])
    eig_val, eig_vec = np.linalg.eig(covar)

    # Inclination of major principal axis w.r.t. x axis
    if eig_val[0] > eig_val[1]:
        phi = np.arctan2(eig_vec[1, 0], eig_vec[0, 0])
    else:
        phi = np.arctan2(eig_vec[1, 1], eig_vec[0, 1])

    return phi


def rotate_curve(x, y, theta):
    cpx = x * np.cos(theta) - y * np.sin(theta)
    cpy = x * np.sin(theta) + y * np.cos(theta)
    return cpx, cpy


def normalize(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    x, y = np.subtract(x, mean_x), np.subtract(y, mean_y)
    denom = np.sqrt(np.var(x, axis=0, keepdims=True) + np.var(y, axis=0, keepdims=True))
    x, y = np.divide(x, denom), np.divide(y, denom)
    phi = -get_pca_inclination(x, y)
    
    return rotate_curve(x, y, phi)
