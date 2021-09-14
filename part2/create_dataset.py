import glob
import json
from os.path import join, isdir
from os import mkdir
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from part2 import settings
import utilities



def get_tfl(path: str, image: np.ndarray) -> List[List]:
    """
    getting the list of data and labels for the image with traffic lights
    :param path: json path file to load labels
    :param image: np.ndarray traffic light image
    :return: list of all data and labels of the traffic lights found
    """
    # open json file and load data
    with open(path) as j_file:
        j_obj = json.load(j_file)
        j_file.close()
    tfl = []
    for elem in j_obj['objects']:
        # collect all the data of traffic lights
        if elem['label'] == 'traffic light':
            # find best pixels to crop traffic light image
            y_max = max(elem['polygon'], key=lambda x: x[0])[0]
            y_min = min(elem['polygon'], key=lambda x: x[0])[0]
            x_max = max(elem['polygon'], key=lambda x: x[1])[1]
            x_min = min(elem['polygon'], key=lambda x: x[1])[1]
            # getting cropped image of traffic light
            current_traffic_light = utilities.crop_tfl_img(image, y_max, y_min, x_max, x_min)

            # is traffic light cropped image is withing the right shape -> append to list
            if utilities.is_cropped_image_ok(current_traffic_light):
                tfl.append([current_traffic_light, settings.TFL_SIGN])
    return tfl


def get_not_tfl(image: np.ndarray, num: int) -> List[List]:
    """
    getting the list of data and labels for the image without traffic lights
    :param image: np.ndarray traffic light image
    :param num: how many random positions we need to generate
    :return: list of num data and labels of the none traffic lights found
    """
    not_tfl = []
    for i in range(num):
        x = random.randint(0, image.shape[0])
        y = random.randint(0, image.shape[1])

        current_traffic_light = utilities.crop_not_fl_img(image, (x, y))

        if utilities.is_cropped_image_ok(current_traffic_light):
            not_tfl.append([current_traffic_light, settings.NOT_TFL_SIGN])
    return not_tfl


def get_list_tfl_not_fl_data(file_path: str, image: np.ndarray) -> List[List]:
    """
    getting list of data and labels with traffic light and without
    :param file_path: json file
    :param image: np.ndarray image
    :return: list of data and labels with traffic light and without
    """
    data = get_tfl(file_path, image)
    data.extend(get_not_tfl(image, len(data)))
    return data


def create_directories(root_path: str,dirs: List[str]) -> None:
    for dir_name in dirs:
        dir_path = join(root_path, dir_name)
        if not isdir(dir_path):
            mkdir(dir_path)
            print(f'-Created `{dir_path}` directory')



def save_binary_file(data: List[List]) -> None:
    """
    saving the list of data and labels into binary files
    :param data: list of data and labels of traffic lights and without
    :return: None
    """
    if not isdir(settings.DATA_DIR_PATH):
        mkdir(settings.DATA_DIR_PATH)
        print(f'-Created `{settings.DATA_DIR_PATH}` directory')

    create_directories(settings.DATA_DIR_PATH, [settings.TRAIN_DIR_NAME, settings.VAL_DIR_NAME, settings.TEST_DIR_NAME])

    current_path = join(settings.DATA_DIR_PATH, settings.TEST_DIR_NAME)

    # open data and labels files to write all the data
    with open(join(current_path, settings.DATA_BIN_FILE_NAME), 'ab') as data_file, \
            open(join(current_path, settings.LABELS_BIN_FILE_NAME), 'ab') as labels_file:
        for item in data:
            cropped_img = item[0]
            status = item[1]
            np.array(cropped_img).tofile(data_file)
            labels_file.write(np.array(status).tobytes())


def main():
    # get all list of files to scan
    files_to_read = glob.glob(settings.JSON_PATH)
    count = 1
    # labeled_list = []
    for file_path in files_to_read[settings.FILES_START_INDEX:]:
        if settings.FILES_USE_COUNT and settings.FILES_COUNT == count:
            break
        try:
            # after getting image array, we need to get the color path of this array
            # getting the directory name
            dic_with_file_path = '/'.join(file_path.split('\\')[-2:])
            file_change = dic_with_file_path.replace(settings.FILE_EXT_FROM, settings.FILE_EXT_TO)
            color_path = join(settings.LEFT_IMG_PATH_DIRECTORY, settings.TEST_DIR_NAME, file_change)
            print(f'{count}: Loading file:', color_path)
            image = np.array(Image.open(color_path), dtype='uint8')

            # getting labels data
            current_labels = get_list_tfl_not_fl_data(file_path, image)
            # save to binary
            save_binary_file(current_labels)
        except Exception as e:
            print('Error: ', e)
        count += 1
    print('-Done-')


if __name__ == '__main__':
    main()
