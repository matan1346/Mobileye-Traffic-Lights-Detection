import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    """

    :param prev_container:
    :param curr_container:
    :param focal:
    :param pp:
    :return:
    """
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_curr_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = \
            calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    """

    :param prev_container:
    :param curr_container:
    :param focal:
    :param pp:
    :return:
    """
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(curr_container.EM)
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    """

    :param norm_prev_pts:
    :param norm_curr_pts:
    :param R:
    :param foe:
    :param tZ:
    :return:
    """
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    """
    transform pixels into normalized pixels using the focal length and principle point
    :param pts: x ()
    :param focal: focal length
    :param pp: principle point
    :return:
    """
    normalized_pts = []
    for p in pts:
        normalized_pts.append([(p[0] - pp[0]) / focal, (p[1] - pp[1]) / focal, 1])
    return np.array(normalized_pts)


def unnormalize(pts, focal, pp):
    """
    transform normalized pixels into pixels using the focal length and principle point
    :param pts:
    :param focal:
    :param pp:
    :return:
    """
    return (focal * pts[:, :2]) + pp


def decompose(EM):
    """
    extract R, foe and tZ from the Ego Motion
    :param EM:
    :return:
    """
    R = np.array(EM[:3, :3])
    t = [EM[0][3]] + [EM[1][3]] + [EM[2][3]]
    foe = np.array([t[0] / t[2]] + [t[1] / t[2]])
    return R, foe, t[2]


def rotate(pts, R):
    """
    rotate the points - pts using R
    :param pts:
    :param R:
    :return:
    """
    r_pts = []
    for p in pts:
        r_pt = np.dot(R, p)
        r_pt = [r_pt[0] / r_pt[2], r_pt[1] / r_pt[2]]
        r_pts.append(r_pt)
    return np.array(r_pts)


def find_corresponding_points(p, norm_pts_rot, foe):
    """
    compute the epipolar line between p and foe
    run over all norm_pts_rot and find the one closest to the epipolar line
    return the closest point and its index
    :param p:
    :param norm_pts_rot:
    :param foe:
    :return:
    """
    eX, eY = foe[0], foe[1]
    xP, yP = p[0], p[1]
    m = (eY - yP) / (eX - xP)
    n = (yP * eX - eY * xP) / (eX - xP)
    d = lambda point: abs((m * point[0] + n - point[1]) / ((m ** 2 + 1) ** 0.5))
    closest_point = min(norm_pts_rot, key=d)
    return np.where(norm_pts_rot == closest_point), closest_point


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    zX = (tZ * (foe[0] - p_rot[0]))  #/ p_curr[0]
    # curr_rotate x
    crX = p_curr[0] - p_rot[0]

    if crX != 0:
        zX /= crX

    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    zY = (tZ * (foe[1] - p_rot[1])) #/ p_curr[1]
    # curr_rotate y
    crY = p_curr[1] - p_rot[1]
    if crY != 0:
        zY /= crY

    # calculate z distance using curr and rotate positions
    sumW = abs(crX) + abs(crY)

    if sumW == 0:
        return 0

    Z = np.array([(abs(crX) / sumW) * zX, (abs(crY) / sumW) * zY])
    return np.sum(Z)