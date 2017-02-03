import numpy as np

from keras.layers import Input, LeakyReLU
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.models import save_model, load_model

# Data Loaded
X_train = np.load('/Users/zhangguanghua/Desktop/Stampede/0.npy')
#X_train = 2.0*(X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1.
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_train = 1.0 - X_train
X_train = X_train.reshape((X_train.shape[0], 3, 64, 64))

print(np.min(X_train), np.max(X_train))
print('X_train shape --- ', X_train.shape)
print(X_train.shape[0], 'train samples')

# generative model

g_input = Input(shape=(100,))

generator = Sequential()
generator.add(Dense(1024 * 4 * 4, input_shape=(100,)))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))
generator.add(Reshape([1024, 4, 4]))

generator.add(UpSampling2D(size=(2, 2), dim_ordering='th'))
generator.add(Convolution2D(512, 5, 5, border_mode='same', dim_ordering='th'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))

generator.add(UpSampling2D(size=(2, 2), dim_ordering='th'))
generator.add(Convolution2D(256, 5, 5, border_mode='same', dim_ordering='th'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))

generator.add(UpSampling2D(size=(2, 2), dim_ordering='th'))
generator.add(Convolution2D(128, 5, 5, border_mode='same', dim_ordering='th'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))

generator.add(UpSampling2D(size=(2, 2), dim_ordering='th'))
generator.add(Convolution2D(64, 5, 5, border_mode='same', dim_ordering='th'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))

generator.add(Convolution2D(3, 5, 5, border_mode='same', dim_ordering='th'))
generator.add(Activation('sigmoid'))

generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.5))
generator.summary()

# discriminative model

discriminator = Sequential()

discriminator.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', dim_ordering='th', input_shape=X_train.shape[1:]))
discriminator.add(LeakyReLU(0.2))


discriminator.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same', dim_ordering='th'))
discriminator.add(LeakyReLU(0.2))


discriminator.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', dim_ordering='th'))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', dim_ordering='th'))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Flatten())

discriminator.add(Dense(1024))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.5))

discriminator.add(Dense(2, activation='softmax'))

discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.5))
discriminator.summary()

# GAN Model
gan_input = Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan_model = Model(gan_input, gan_output)

gan_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.5))
gan_model.summary()

# Load Model
#generator.load_weights('/Users/zhangguanghua/Desktop/G1_weights.h5')
#discriminator.load_weights('/Users/zhangguanghua/Desktop/D1_weights.h5')
#print("Model Loaded")

print("Pre-training generator...")
noise_gen = np.random.uniform(0, 1, size=(10000, 100))   # at (0,1) creates 10000 points
generated_images = generator.predict(noise_gen)

print('generated_images shape ----', generated_images.shape)

X = np.concatenate((X_train[:10000, :, :, :], generated_images))
y = np.zeros([20000, 2])
y[:10000, 1] = 1
y[10000:, 0] = 1

discriminator.fit(X, y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)

# set up loss storage vector
losses = {"d": [], "g": []}


def train_for_n(nb_epoch=20000, batch_size=128):
    for e in range(nb_epoch):

        # Make generative images
        train_idx = np.random.randint(0, X_train.shape[0], size=batch_size)  # 0 <= train_idx <= X_train.shape[0]
        mini_batch = X_train[train_idx]
        noise_gen = np.random.normal(0, 1, size=(batch_size, 100))
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((mini_batch, generated_images))
        y = np.zeros([2 * batch_size, 2])
        y[:batch_size, 1] = 1
        y[batch_size:, 0] = 1

        discriminator.trainable = True
        for layer in discriminator.layers:
            layer.trainable = True
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        noise_tr = np.random.uniform(0, 1, size=(batch_size, 100))
        y2 = np.zeros([batch_size, 2])
        y2[:, 1] = 1

        discriminator.trainable = False
        for layer in discriminator.layers:
            layer.trainable = False
        g_loss = gan_model.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        if e % 10 == 9:
            generator.save_weights('G0_weights.h5')
            discriminator.save_weights('D0_weights.h5')
            noise = np.random.uniform(0, 1, size=(100, 100))
            generated_images = generator.predict(noise)
            np.save('/Users/zhangguanghua/Desktop/Stampede/generated_images_0.npy', generated_images)

        print(("Iteration: {0} / {1}, G-Loss: {2:.4f}".format(e, nb_epoch, float(g_loss))))


train_for_n(nb_epoch=2000, batch_size=128)
