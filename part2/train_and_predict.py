import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sbn

import settings


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, settings.DATA_BIN_FILE_NAME),
                       mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, settings.LABELS_BIN_FILE_NAME), mode='r', dtype=np.uint8)
    labels = np.array(list(map(lambda x: x - 48, labels)))

    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name=settings.LABELS_TO_NAME):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05},
                      squeeze=False, sharex=True, sharey=True)[1]  # .flatten()
    index_shape = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(index_shape):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None:
            title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)

    plt.show()


# root = './'  #this is the root for your val and train datasets
datasets = {
    'val': load_tfl_data(join(settings.DATA_DIR_PATH, settings.VAL_DIR_NAME)),
    'train': load_tfl_data(join(settings.DATA_DIR_PATH, settings.TRAIN_DIR_NAME)),
}

for k, v in datasets.items():
    print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

viz_my_data(num=(6, 6), **datasets['val'])


# ####################### define the model used for training ###################


def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 32)
    spatial_layer(2, 64)
    spatial_layer(2, 96)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


m = tfl_model()
m.summary()

# ############################ train #####################

datasets = {
    'val': load_tfl_data(join(settings.DATA_DIR_PATH, settings.VAL_DIR_NAME)),
    'train': load_tfl_data(join(settings.DATA_DIR_PATH, settings.TRAIN_DIR_NAME)),
}

# prepare our model
m = tfl_model()
m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']

# train it, the model uses the 'train' dataset for learning.
# We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
history = m.fit(train['images'], train['labels'],
                validation_data=(val['images'], val['labels']), epochs=50)

# compare train vs val accuracy,
# why is val_accuracy not as good as train accuracy? are we over fitting?


# compare train vs val accuracy,
# why is val_accuracy not as good as train accuracy? are we over fitting?
epochs = history.history
epochs['train_acc'] = epochs['accuracy']
plt.figure(figsize=(10, 10))
for k in ['train_acc', 'val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k], label=k)

plt.legend()

# ##################### evaluate and predict ####################

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0])

predicted_label = np.argmax(predictions, axis=-1)
print('accuracy:', np.mean(predicted_label == val['labels']))

viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val)

# ############# save the model ##################

m.save("model.h5")
