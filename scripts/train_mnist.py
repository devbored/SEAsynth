#!/usr/bin/env python

import struct
from pathlib import Path
from tensorflow.compat.v1 import keras

def train():
    # Load the data
    minst = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = minst.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Create custom PynqNet model
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28,28)),
        keras.layers.Reshape(target_shape=(28,28,1)),
        keras.layers.Conv2D(12, kernel_size=(3,3),
            strides=(1,1), activation='relu', input_shape=(28,28,1), use_bias=False),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax', use_bias=False)
    ])

    # Train the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_images, train_labels, epochs=10)
    print("[Info]: PynqNet successfully trained.")

    # Test accuracy
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Extract trained layer values
    # (layer1 = 'conv2d', layer5 = 'dense')
    npa_conv2d = model.layers[1].get_weights()[0]
    npa_dense = model.layers[5].get_weights()[0]

    # Output raw hexadecimal float32 values to 'cell-weights.mem' file
    weights_conv2d_dir = "build/raw-mnist-weights-float32/conv2d"
    weights_fc1_dir = "build/raw-mnist-weights-float32/fc"
    Path(weights_conv2d_dir).mkdir(parents=True, exist_ok=True)
    Path(weights_fc1_dir).mkdir(parents=True, exist_ok=True)
    print("[Info]: Converting trained weight data to raw hexadecimal float32 values...")

    # Convert conv2d layer values
    cell_idx = 0
    for row in npa_conv2d:
        for col in row:
            # Open weight-file-[cell-idx].mem
            cell_file = open(Path(weights_conv2d_dir + "/cell-weight-" + str(cell_idx) + ".mem"), 'w')

            # Get weights for the current cell idx
            for kernel_px in col[0]:
                f32_val = float(kernel_px)
                byte_arr = bytearray(struct.pack("f", f32_val))
                byte_lst = ["%02x" % b for b in byte_arr]
                byte_lst.reverse()
                raw_f32_val = "".join(byte_lst) + "\n"
                cell_file.write(raw_f32_val)

            # Close the cell's weight file; move to next cell
            cell_file.close()
            cell_idx += 1

    # Convert dense layer values
    cell_file = open(Path(weights_fc1_dir + "/cell-fc.mem"), 'w')
    for ip in npa_dense:
        for op in ip:
            f32_val = float(op)
            byte_arr = bytearray(struct.pack("f", f32_val))
            byte_lst = ["%02x" % b for b in byte_arr]
            byte_lst.reverse()
            raw_f32_val = "".join(byte_lst) + "\n"
            cell_file.write(raw_f32_val)

    # Close the 'cell_file'
    cell_file.close()
    print("[Info]: Done.")

# =====================================================================================================================
if __name__ == "__main__":
    train()