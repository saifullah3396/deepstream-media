import argparse
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import os


def main():
    parser = argparse.ArgumentParser(
        description='Generate a tensorboard file from tf frozen graph.')
    parser.add_argument('--graph', help='Model graph path.', required=True)
    args = parser.parse_args()
    directory = os.path.dirname(args.graph)

    with tf.Session() as sess:
        with gfile.FastGFile(args.graph, 'rb') as graph:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph.read())
            graph_in = tf.import_graph_def(graph_def)
        train_writer = tf.summary.FileWriter('{}/logs_dir'.format(directory))
        train_writer.add_graph(sess.graph)


if __name__ == "__main__":
    main()
