import pickle
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from part1.main import test_find_tfl_lights
from part2.utilities import crop_not_fl_img, is_cropped_image_ok
from part3.SFM_standAlone import FrameContainer, visualize
import part3.SFM


class Controller:

    def __init__(self, pls_path):
        with open(pls_path, 'r') as pls_file:
            pls_data = pls_file.readlines()
            self.pkl_path = pls_data[0].strip()
            self.start_frame_id = pls_data[1].strip()
            self.frames = list(map(lambda x: x.strip(), pls_data[2:]))
        self.loaded_model = load_model("part2/model.h5")

    def run(self):
        with open(self.pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file, encoding='latin1')
            prev_tfl_manager = None
            for frame in self.frames:
                curr_id = int(frame.split('_')[2].strip())
                print(f'Frame ({curr_id}): ', frame)
                tfl_manager = TflManager(frame,
                                         self.loaded_model,
                                         curr_id,
                                         data['egomotion_' + str(curr_id - 1) + '-' + str(curr_id)],
                                         data['principle_point'],
                                         data['flx'])
                candidates, auxiliary = tfl_manager.part_1(frame)
                traffic_lights, aux = tfl_manager.part_2(frame, candidates, auxiliary)

                print('Part 1: num candidates: ', len(candidates))
                print('Part 2: num traffic lights: ', len(traffic_lights), "/", len(candidates))

                if prev_tfl_manager:
                    print(f'Part 3: calculating distance by frames {curr_id-1}-{curr_id}')
                    tfl_manager.part_3(traffic_lights, prev_tfl_manager)
                prev_tfl_manager = tfl_manager


class TflManager:

    def __init__(self, i, model, frame_id, em=None, pp=None, fl=None):
        self.em_matrix = em
        self.principle_point = pp
        self.focal_len = fl
        self.img = i
        self.frame_id = frame_id
        self.candidates = np.array([])
        self.traffic_lights = np.array([])
        self.loaded_model = model

    def part_1(self, image_path):
        """
            Input: Current frame - one png image
            Output:
            candidates - vector of N positions
            auxiliary – vector of N colors {green, orenge, red}
        """
        self.candidates = test_find_tfl_lights(image_path)
        auxiliary = []
        return self.candidates, auxiliary

    def part_2(self, image_path, candidates, auxiliary):
        """inputs:
            Current frame - one png image
            candidates  - output from Part 1
            auxiliary – output from Part 1

        outputs:
            TrafficLights - vector of K positions notice K <= N
            auxiliary – vector of K colors
        """
        current_image_array = np.array(Image.open(image_path), dtype='uint8')

        images_to_predict = []
        pixels_to_predict = []
        for pixel in candidates:
            cropped_img = crop_not_fl_img(current_image_array, pixel[::-1])
            if is_cropped_image_ok(cropped_img):
                images_to_predict.append(cropped_img)
                pixels_to_predict.append(pixel)
        predictions_data = self.loaded_model.predict(np.array(images_to_predict))
        predictions = predictions_data[:, 1]

        traffic_lights = []
        for i, pred in enumerate(predictions):
            if pred > 0.8:
                traffic_lights.append(pixels_to_predict[i])

        plt.figure().clf()
        plt.imshow(current_image_array)
        self.traffic_lights = np.array(traffic_lights)

        return self.traffic_lights, auxiliary

    def part_3(self, traffic_lights, prev_tfl_manager):
        """
        input:
        Previous and current frames - two png images
        Two vectors of TrafficLights positions – from Part 2
        Two auxiliary vectors – from Part 2
        EM matrix, Principal point, focal length
        output: Distance – vector of K lengthes of traffic lights
        :return:
        """
        prev_container = FrameContainer(prev_tfl_manager.img)
        curr_container = FrameContainer(self.img)
        prev_container.traffic_light = prev_tfl_manager.traffic_lights
        curr_container.traffic_light = traffic_lights
        curr_container.EM = self.em_matrix
        prev_container.frame_id = prev_tfl_manager.frame_id
        curr_container.frame_id = self.frame_id
        curr_container = part3.SFM.calc_TFL_dist(prev_container, curr_container, self.focal_len,
                                                 self.principle_point)
        visualize(prev_container, curr_container, self.focal_len, self.principle_point)


def main():
    inst = Controller(r"play_list.pls")
    inst.run()


if __name__ == '__main__':
    main()
