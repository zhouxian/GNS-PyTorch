import os
import json
import pickle
import argparse
import functools
import numpy as np
import tensorflow.compat.v1 as tf

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='path to tf dataset', type=str)
    return parser.parse_args()

def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out


def parse_serialized_simulation_example(example_proto, metadata):
  """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  return context, parsed_features

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def input_fn(data_path, split):
    # Loads the metadata of the dataset.
    metadata = _read_metadata(data_path)
    # Create a tf.data.Dataset from the TFRecord.
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
            parse_serialized_simulation_example, metadata=metadata))
    return ds

def main():

    args = arg_parse()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    metadata = _read_metadata(args.data_path)

    data_name = args.data_path.split('/')[-1]
    save_dir =  f'./data/{data_name}'
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metadata, open(os.path.join(save_dir, 'metadata.json'), 'wt'))

    for split in ['train', 'valid', 'test']:
        ds = input_fn(args.data_path, split=split)
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        iterator = tf.data.make_one_shot_iterator(ds)
        data = iterator.get_next()

        i = 0
        while True:
            print(i)
            try:
                traj = sess.run(data)
                traj = {'particle_type': traj[0]['particle_type'], 
                        'position': traj[1]['position']}
                pickle.dump(traj, open(os.path.join(split_dir, str(i)+'.pkl'), "wb"))
                i += 1

            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
        main()
