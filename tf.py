from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from hdfs import InsecureClient
import errno as _errno
import sys as _sys
from json import dumps

from tensorflow.python.platform import flags
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export
import time
import argparse
import os.path
import re
import sys
import tarfile
import os
import datetime
import math
import numpy as np
from six.moves import urllib
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from time import gmtime, strftime

# Create unique image name
uniqueid = 'tf_uuid_{0}_{1}'.format('img',strftime("%Y%m%d%H%M%S",gmtime()))

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
# pylint: enable=line-too-long

class NodeLookup(object):

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, "imagenet_2012_challenge_label_map_proto.pbtxt")
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, "imagenet_synset_to_human_label_map.txt")
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal("File does not exist %s", uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal("File does not exist %s", label_lookup_path)

    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r"[n\d]*[ \S,]*")
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith("  target_class:"):
        target_class = int(line.split(": ")[1])
      if line.startswith("  target_class_string:"):
        target_class_string = line.split(": ")[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal("Failed to locate: %s", val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ""
    return self.node_lookup[node_id]

def create_graph():
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, "classify_image_graph_def.pb"), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")


def run_inference_on_image(image):
  if not tf.gfile.Exists(image):
    tf.logging.fatal("File does not exist %s", image)
  image_data = tf.gfile.FastGFile(image, "rb").read()

  create_graph()

  # write output to HDFS
  client = InsecureClient('http://princeton0.field.hortonworks.com:50070', user='root')

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
    predictions = sess.run(softmax_tensor,
                           {"DecodeJpeg/contents:0": image_data})
    predictions = np.squeeze(predictions)
    node_lookup = NodeLookup()

    # sort the predictions in order of score
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    row = {}

    # output all of our human string and scores
    for node_id in top_k:
      # get the printable string for that id
      human_string = node_lookup.id_to_string(node_id)
      # get the prediction score for that id
      score = predictions[node_id]
      # let"s print it as a nice table in zeppelin
      print("{0}\t{1}\t{2}%\n".format(str(node_id), str(human_string),  float(score) * 100.0))
      row['node_id{0}'.format(str(node_id))] = '{0}'.format(str(node_id))
      row['humanstr{0}'.format(str(node_id))] = '{0}'.format(str(human_string))
      row['score{0}'.format(str(node_id))] = '{0}'.format(round(float(score) * 100.0,1))

    client.write('/tfyarn/' + uniqueid + '.json', dumps(row))

def maybe_download_and_extract():
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split("/")[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
        pass
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    statinfo = os.stat(filepath)
  tarfile.open(filepath, "r:gz").extractall(dest_directory)

def main(_):
  maybe_download_and_extract()
  img_name = "/opt/demo/images/photo1.jpg"
  run_inference_on_image(img_name)

parser = argparse.ArgumentParser()

parser.add_argument(
      "--model_dir",
      type=str,
      default="/opt/demo/imagenet",
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """)
parser.add_argument(
      "--image_file",
      type=str,
      default="",
      help="Absolute path to image file." )
parser.add_argument(
      "--num_top_predictions",
      type=int,
      default=5,
      help="Display this many predictions.")
FLAGS, unparsed = parser.parse_known_args()
main(FLAGS)
