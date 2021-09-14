import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import part3.SFM


def visualize(prev_container, curr_container, focal, pp):
    """
    :param prev_container:
    :param curr_container:
    :param focal:
    :param pp:
    :return:
    """
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = part3.SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = part3.SFM.rotate(norm_prev_pts, R)
    rot_pts = part3.SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(part3.SFM.unnormalize(np.array([norm_foe]), focal, pp))

    fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
    prev_sec.set_title('prev(' + str(prev_container.frame_id) + ')')
    prev_sec.imshow(prev_container.img)
    prev_p = prev_container.traffic_light
    prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

    curr_sec.set_title('curr(' + str(curr_container.frame_id) + ')')
    curr_sec.imshow(curr_container.img)
    curr_p = curr_container.traffic_light
    curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

    for i in range(len(curr_p)):
        curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if curr_container.valid[i]:
            curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                          r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
    curr_sec.plot(foe[0], foe[1], 'r+')
    curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
    plt.show()


class FrameContainer(object):
    def __init__(self, img_path):
        # fn = get_sample_data(img_path, asfileobj=True)
        self.img = np.array(Image.open(img_path))
        self.frame_id = 0
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
