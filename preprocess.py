from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pprint
import tempfile
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_schema

VOCAB_SIZE = 20000
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 200
NUM_TRAIN_INSTANCES = 25000
NUM_TEST_INSTANCES = 25000

CONTEXT_KEY = 'Context'
UTTERANCE_KEY = 'Utterance'
LABEL_KEY = 'Label'

"""--------------------------------------------- Train Data Schema ----------------------------------------"""
TRAIN_RAW_DATA_FEATURE_SPEC = {
    CONTEXT_KEY: tf.io.FixedLenFeature([], tf.string),
    UTTERANCE_KEY: tf.io.FixedLenFeature([], tf.string),
    LABEL_KEY: tf.io.FixedLenFeature([], tf.int64)
}

TRAIN_RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(TRAIN_RAW_DATA_FEATURE_SPEC))

"""--------------------------------------------- Test Data Schema ----------------------------------------"""
TEST_RAW_DATA_FEATURE_SPEC = {
    CONTEXT_KEY: tf.io.FixedLenFeature([], tf.string),
    'Ground Truth Utterance': tf.io.FixedLenFeature([], tf.string),
    'Distractor_0': tf.io.FixedLenFeature([], tf.string),
    'Distractor_1': tf.io.FixedLenFeature([], tf.string),
    'Distractor_2': tf.io.FixedLenFeature([], tf.string),
    'Distractor_3': tf.io.FixedLenFeature([], tf.string),
    'Distractor_4': tf.io.FixedLenFeature([], tf.string),
    'Distractor_5': tf.io.FixedLenFeature([], tf.string),
    'Distractor_6': tf.io.FixedLenFeature([], tf.string),
    'Distractor_7': tf.io.FixedLenFeature([], tf.string),
    'Distractor_8': tf.io.FixedLenFeature([], tf.string)
}

TEST_RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(TEST_RAW_DATA_FEATURE_SPEC))
"""--------------------------------------------------------------------------------------------------------------"""
DELIMITERS = '.,!?() '

SHUFFLED_TRAIN_DATA_FILEBASE = 'train_shuffled'
SHUFFLED_TEST_DATA_FILEBASE = 'test_shuffled'
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
TRANSFORM_TEMP_DIR = 'tft_temp'
EXPORTED_MODEL_DIR = 'exported_model_dir'

train_converter_input = tft.coders.CsvCoder(['Context','Utterance','Label'],TRAIN_RAW_DATA_METADATA.schema,delimiter=',')
test_converter_input = tft.coders.CsvCoder(['Context','Ground Truth Utterance','Distractor_0','Distractor_1','Distractor_2','Distractor_3','Distractor_4','Distractor_5','Distractor_6','Distractor_7','Distractor_8'],TEST_RAW_DATA_METADATA.schema,delimiter=',')

@beam.ptransform_fn
def ReadTrainData(pcoll, filepatterns): 
    train_filepattern = filepatterns[0]

    train_examples = (
        pcoll
        | 'Read Train Examples' >> beam.io.ReadFromText(train_filepattern,skip_header_lines=1)
        | 'Parse Input CSV' >> beam.Map(train_converter_input.decode))

    return train_examples | 'Make Train Instances' >> beam.Map(lambda p: {CONTEXT_KEY: p['Context'], UTTERANCE_KEY: p['Utterance'], LABEL_KEY: p['Label']})

@beam.ptransform_fn
def ReadTestData(pcoll, filepatterns): 
    test_filepattern = filepatterns[1]

    test_examples = (
        pcoll
        | 'Read Test Examples' >> beam.io.ReadFromText(test_filepattern,skip_header_lines=1)
        | 'Parse Input CSV' >> beam.Map(test_converter_input.decode))

    return test_examples | 'Make Test Instances' >> beam.Map(
        lambda p: {'Context': p['Context'], 
                'Ground Truth Utterance': p['Ground Truth Utterance'],
                'Distractor_0': p['Distractor_0'],
                'Distractor_1': p['Distractor_1'],
                'Distractor_2': p['Distractor_2'],
                'Distractor_3': p['Distractor_3'],
                'Distractor_4': p['Distractor_4'],
                'Distractor_5': p['Distractor_5'],
                'Distractor_6': p['Distractor_6'],
                'Distractor_7': p['Distractor_7'],
                'Distractor_8': p['Distractor_8']
                })

def read_data(train_filepattern,test_filepattern, working_dir):

    with beam.Pipeline() as pipeline:
        train_coder = tft.coders.ExampleProtoCoder(TRAIN_RAW_DATA_METADATA.schema)
        test_coder = tft.coders.ExampleProtoCoder(TEST_RAW_DATA_METADATA.schema)

        _ = (
            pipeline
            | 'Read Train Data' >> ReadTrainData((train_filepattern,test_filepattern))
            | 'Encode Train Data' >> beam.Map(train_coder.encode)
            | 'Write Train Data' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, SHUFFLED_TRAIN_DATA_FILEBASE)))

        _ = (
            pipeline
            | 'Read Test Data' >> ReadTestData((train_filepattern,test_filepattern))
            | 'Encode Test Data' >> beam.Map(test_coder.encode)
            | 'Write Test Data' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, SHUFFLED_TEST_DATA_FILEBASE)))

def transform_data(working_dir):

    with beam.Pipeline() as pipeline:
        with tft_beam.Context(
            temp_dir=os.path.join(working_dir, TRANSFORM_TEMP_DIR)):
            train_coder = tft.coders.ExampleProtoCoder(TRAIN_RAW_DATA_METADATA.schema)
            test_coder = tft.coders.ExampleProtoCoder(TEST_RAW_DATA_METADATA.schema)
            train_data = (
                pipeline
                | 'ReadTrain' >> beam.io.ReadFromTFRecord(
                    os.path.join(working_dir, SHUFFLED_TRAIN_DATA_FILEBASE + '*'))
                | 'DecodeTrain' >> beam.Map(train_coder.decode))

            def preprocessing_fn_train(inputs):
                """Preprocess input columns into transformed columns."""
                context = inputs[CONTEXT_KEY]
                utterance = inputs[UTTERANCE_KEY]
                #z = context + utterance
                #tf.print(z,output_stream=sys.stdout)
                #context_tokens = tf.compat.v1.string_split(context, DELIMITERS)
                transformed_context = tft.compute_and_apply_vocabulary(context, top_k=VOCAB_SIZE, frequency_threshold= 3,vocab_filename='anantvir_vocab_context')
                transformed_utterance = tft.compute_and_apply_vocabulary(utterance, top_k=VOCAB_SIZE, frequency_threshold= 3,vocab_filename='anantvir_vocab_utterance')
                
                return {
                    CONTEXT_KEY: transformed_context,
                    UTTERANCE_KEY: transformed_utterance,
                    LABEL_KEY: inputs[LABEL_KEY]
                }
            (transformed_train_data, transformed_metadata), transform_fn = (
                (train_data, TRAIN_RAW_DATA_METADATA)
                | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn_train))
            transformed_data_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema)

            _ = (
                transformed_train_data
                | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
                | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

            _ = (
                transform_fn
                | 'WriteTransformFn' >>
                tft_beam.WriteTransformFn(working_dir))

def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):

  def input_fn():
    """Input function for training and eval."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        shuffle=True)

    transformed_features = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    transformed_labels = transformed_features.pop(LABEL_KEY)
    return transformed_features, transformed_labels
  return input_fn

def _make_serving_input_fn(tf_transform_output):
  raw_feature_spec = TRAIN_RAW_DATA_FEATURE_SPEC.copy()
  raw_feature_spec.pop(LABEL_KEY)

  def serving_input_fn():

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    raw_features = serving_input_receiver.features
    transformed_features = tf_transform_output.transform_raw_features(
        raw_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)
  return serving_input_fn


def get_feature_columns(tf_transform_output):
  del tf_transform_output  # unused
  review_column = tf.feature_column.categorical_column_with_identity(
      REVIEW_KEY, num_buckets=VOCAB_SIZE + 1)
  weighted_reviews = tf.feature_column.weighted_categorical_column(
      review_column, REVIEW_WEIGHT_KEY)

  return [weighted_reviews]


def train_and_evaluate(working_dir,
                       num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):

  tf_transform_output = tft.TFTransformOutput(working_dir)

  run_config = tf.estimator.RunConfig()

  estimator = tf.estimator.LinearClassifier(
      feature_columns=get_feature_columns(tf_transform_output),
      config=run_config,
      loss_reduction=tf.losses.Reduction.SUM)

  train_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
      batch_size=TRAIN_BATCH_SIZE)
  estimator.train(
      input_fn=train_input_fn,
      max_steps=TRAIN_NUM_EPOCHS * num_train_instances / TRAIN_BATCH_SIZE)

  eval_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*'),
      batch_size=1)
  result = estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)

  serving_input_fn = _make_serving_input_fn(tf_transform_output)
  exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  estimator.export_saved_model(exported_model_dir, serving_input_fn)

  return result


def main():
    input_data_dir = 'D:\\Courses\\Chatbot\\dataset_trimmed\\data'
    working_dir = tempfile.mkdtemp(dir=input_data_dir)

    train_filepattern = os.path.join(input_data_dir, 'train.csv')
    test_filepattern = os.path.join(input_data_dir, 'test.csv')

    read_data(train_filepattern,test_filepattern,working_dir)
    transform_data(working_dir)
    results = train_and_evaluate(working_dir)

    pprint.pprint(results)


if __name__ == '__main__':
  main()