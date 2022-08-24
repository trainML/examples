import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
import os
import time
import _pickle as cPickle

parser = argparse.ArgumentParser(description="Tensorflow CIFAR-10 Training")

parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="adam",
    type=str,
    choices=["adam", "rmsprop", "sgd", "adagrad"],
    help="optimizer to use for training",
)


def load_batch(fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data.
    Args:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, "rb") as f:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    path = f"{os.environ.get('TRAINML_DATA_PATH')}/cifar-10-batches-py"
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if tf.keras.backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


class BatchTimestamp(object):
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
            self.batch_index, self.timestamp
        )


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, log_steps):
        """Callback for logging performance.

        Args:
        batch_size: Total batch size.
        log_steps: Interval of steps between logging of batch level stats.
        """
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.log_steps = log_steps
        self.global_steps = 0

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []

    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.global_steps += 1
        if self.global_steps == 1:
            self.start_time = time.time()
            self.timestamp_log.append(
                BatchTimestamp(self.global_steps, self.start_time)
            )

    def on_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        if self.global_steps % self.log_steps == 0:
            timestamp = time.time()
            elapsed_time = timestamp - self.start_time
            examples_per_second = (
                self.batch_size * self.log_steps
            ) / elapsed_time
            self.timestamp_log.append(
                BatchTimestamp(self.global_steps, timestamp)
            )
            print(
                f"BenchmarkMetric: {{'global step': {self.global_steps:d}, 'time_taken': {elapsed_time:f}, 'examples_per_second': {examples_per_second:f}}}"
            )
            self.start_time = timestamp

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)
        print(
            f"BenchmarkMetric: {{'epoch': {(epoch + 1):d}, 'time_taken': {epoch_run_time:f}}}"
        )


def build_callbacks(batch_size):
    time_callback = TimeHistory(batch_size, 100)
    callbacks = [time_callback]

    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{os.environ.get('TRAINML_OUTPUT_PATH')}/logs",
            histogram_freq=1,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(os.environ.get("TRAINML_OUTPUT_PATH"), "model-best.ckpt"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(os.environ.get("TRAINML_OUTPUT_PATH"), "model-last.ckpt"),
            save_weights_only=True,
        )
    )
    return callbacks


def main():
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    workers = args.workers
    optimizer = args.optimizer

    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    print("x_train.shape:", x_train.shape)
    print("y_train.shape", y_train.shape)
    K = len(set(y_train))
    print("number of classes:", K)
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    train_generator = data_generator.flow(x_train, y_train, batch_size)
    steps_per_epoch = x_train.shape[0] // batch_size

    model = resnet50.ResNet50(
        include_top=False,
        weights=None,
        input_shape=x_train[0].shape,
        pooling="max",
        classes=K,
        classifier_activation="softmax",
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = build_callbacks(batch_size)

    history = model.fit(
        train_generator,
        validation_data=(x_test, y_test),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        workers=workers,
    )
    print(f"Training Results: {history.history}")


if __name__ == "__main__":
    main()