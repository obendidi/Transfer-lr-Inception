import random
import tensorflow as tf
from utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset



#===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('dataset_file', "labelsxdata.txt", 'String: Your dataset txt file')

flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

flags.DEFINE_float('validation_size', 0.1, 'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

flags.DEFINE_string('tfrecord_filename', "pos_eo_coord", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main():

    #=============CHECKS==============
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    if _dataset_exists(dataset_file = FLAGS.dataset_file, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==========END OF CHECKS============

    photo_filenames, labels = _get_filenames_and_classes(FLAGS.dataset_file)

    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]
    training_labels = labels[num_validation:]
    validation_labels = labels[:num_validation]

    _convert_dataset('train', training_filenames, training_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, validation_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()
