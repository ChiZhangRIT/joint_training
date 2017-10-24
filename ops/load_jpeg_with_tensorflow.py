# Typical setup to include TensorFlow.
import tensorflow as tf
import pdb

def get_image_tensor(filename):

    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(filename))
    # filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
    #     '/dirs/home/phd/cxz2081/data/mscoco/train2014/COCO_train2014_000000490481.jpg'))


    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image_orig = tf.image.decode_jpeg(image_file)

    # resize image to dimensions of Inception v3 input images
    image_resize = tf.image.resize_images(image_orig, size=[299, 299])
    image_shape=tf.stack([299,299,3])
    image_resize= tf.reshape(image_resize,image_shape)
    # insert batch dimensions.
    image = tf.expand_dims(image_resize, 0)
    # pdb.set_trace()

    return image

    # # Start a new session to show example output.
    # with tf.Session() as sess:
    #     # Required to get the filename matching to run.
    #     tf.global_variables_initializer().run()
    #
    #     # Coordinate the loading of image files.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     # Get an image tensor and print its value.
    #     image_tensor = sess.run(image)
    #     pdb.set_trace()
    #     # print(image_tensor)
    #
    #     # Finish off the filename queue coordinator.
    #     coord.request_stop()
    #     coord.join(threads)

    # return image_tensor
