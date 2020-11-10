import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# Define arguments
parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--batch_size', type=int, help='Batch size to train with')
parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for')
args = parser.parse_args()

# Get arguments from parser
INPUT_NUM_EPOCHS = args.num_epochs
INPUT_BATCH_SIZE = args.batch_size

# Downloading the dataset 
datasets, info = tfds.load('beans', with_info=True, as_supervised=True)
beans_train, beans_test = datasets['train'], datasets['test']

# Initializing the distributed learning algorithm
strategy = tf.distribute.MirroredStrategy()

# Just checking the number of devices available for distributing
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Defining some hyperparameters
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE = INPUT_BATCH_SIZE * strategy.num_replicas_in_sync

# Batching the dataset and keeping a memory buffer for better performance
train_dataset = beans_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = beans_test.map(scale).batch(BATCH_SIZE)

# Building our model
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(500, 500, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)])
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
]

model.fit(train_dataset, epochs=INPUT_NUM_EPOCHS, callbacks=callbacks)
