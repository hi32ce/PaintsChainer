import numpy as np
import chainer
import os
import cv2
import argparse

from chainer import cuda, optimizers, serializers, Variable

import unet


def save_as_img(array, name):
    array = array.transpose(1, 2, 0)
    array = array.clip(0, 255).astype(np.uint8)
    array = cuda.to_cpu(array)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
    else:
        img = cv2.cvtColor(array, cv2.COLOR_YUV2BGR)
    cv2.imwrite(name, img)


def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input', '-i')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', default='./models/unet_128_standard')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(args.out, os.path.basename(input_path))

    print('load model')
    model_path = args.model
    cnn_128 = unet.UNET()
    serializers.load_npz(model_path, cnn_128)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        cnn_128.to_gpu()  # Copy the model to the GPU

    image1 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (128, 128), interpolation=cv2.INTER_AREA)
    image1 = np.asarray(image1, np.float32)
    if image1.ndim == 2:
        image1 = image1[:, :, np.newaxis]

    image1 = np.insert(image1, 1, -512, axis=2)
    image1 = np.insert(image1, 2, 128, axis=2)
    image1 = np.insert(image1, 3, 128, axis=2)

    image_ref = cv2.imread('images/ref/ref.png', cv2.IMREAD_UNCHANGED)
    image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST)
    b, g, r, a = cv2.split(image_ref)
    # image_ref = cvt2YUV( cv2.merge((b, g, r)) )
    image_ref = cv2.cvtColor(cv2.merge((b, g, r)), cv2.COLOR_RGB2YUV)

    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            if a[x][y] != 0:
                for ch in range(3):
                    image1[x][y][ch + 1] = image_ref[x][y][ch]

    x = (image1.transpose(2, 0, 1))
    print(x.shape)

    x_container = np.zeros(
        (1, 4, x.shape[1], x.shape[2]), dtype='f')
    x_container[0, :] = x
    print(x_container.shape)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            result = cnn_128.calc(Variable(x_container))
    print(result.shape)
    print(result[0].shape)
    save_as_img(result.data[0], output)


if __name__ == "__main__":
    main()
