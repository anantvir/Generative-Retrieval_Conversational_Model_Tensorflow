"""
The idea for this code has been borrowed from official Tensorflow Github sentiment analysis repository
https://github.com/tensorflow/transform/blob/599691c8b94bbd6ee7f67c11542e7fef1792a566/examples/sentiment_example.py

It has then been modified according to our requirements.

Contributors:
@Anantvir_Singh 
@Hang Chen
"""

# https://www.tensorflow.org/tfx/transform/get_started#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tempfile
import sys
import apache_beam as beam
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

# from tensorflow_transform.tf_metadata import dataset_schema

"""--------------------------------------------- File Paths ----------------------------------------""" 

# Used during read
TFRECORD_TRAIN_DATA_FILEBASE = 'train_tfrecord'
TFRECORD_TEST_DATA_FILEBASE = 'test_tfrecord'
# Used during transform
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
TRANSFORM_TEMP_DIR = 'tft_temp'

"""--------------------------------------------- Data Schemas ----------------------------------------""" 

# Shall use VarLenFeature?

TRAIN_RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
    'Context': tf.io.FixedLenFeature([], tf.string),
    'Utterance': tf.io.FixedLenFeature([], tf.string),
    'Label': tf.io.FixedLenFeature([], tf.int64)
}))

TEST_RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
    'Context': tf.io.FixedLenFeature([], tf.string),
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
}))

"""--------------------------------------------- CSV MISCS ----------------------------------------""" 

train_csv_columns = ['Context','Utterance','Label']
test_csv_columns = ['Context','Ground Truth Utterance','Distractor_0','Distractor_1','Distractor_2','Distractor_3','Distractor_4','Distractor_5','Distractor_6','Distractor_7','Distractor_8']

train_converter_input = tft.coders.CsvCoder(train_csv_columns, TRAIN_RAW_DATA_METADATA.schema, delimiter=',')
test_converter_input = tft.coders.CsvCoder(test_csv_columns, TEST_RAW_DATA_METADATA.schema, delimiter=',')

DELIMITERS = ',!?() '

"""--------------------------------------------- Read Data ----------------------------------------""" 
# https://github.com/tensorflow/transform/blob/599691c8b94bbd6ee7f67c11542e7fef1792a566/examples/sentiment_example.py
# ptransform Decorater https://beam.apache.org/releases/pydoc/2.7.0/apache_beam.transforms.ptransform.html

@beam.ptransform_fn
def read_train_data(pcoll, train_file_path): 

    train_examples = (
        pcoll
        | 'Read Train Examples' >> beam.io.ReadFromText(train_file_path,skip_header_lines=1)
        | 'Parse Input CSV' >> beam.Map(train_converter_input.decode))

    return train_examples | 'Make Train Instances' >> beam.Map(lambda p: {'Context': p['Context'], 'Utterance': p['Utterance'], 'Label': p['Label']})

@beam.ptransform_fn
def read_test_data(pcoll, test_file_path): 

    test_examples = (
        pcoll
        | 'Read Test Examples' >> beam.io.ReadFromText(test_file_path,skip_header_lines=1)
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

def read_data(train_file_path, test_file_path, working_dir):

    with beam.Pipeline() as pipeline:
        train_coder = tft.coders.ExampleProtoCoder(TRAIN_RAW_DATA_METADATA.schema)
        test_coder = tft.coders.ExampleProtoCoder(TEST_RAW_DATA_METADATA.schema)

        _ = (
            pipeline
            | 'Read Train Data' >> read_train_data(train_file_path)
            | 'Encode Train Data' >> beam.Map(train_coder.encode)
            | 'Write Train Data' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, TFRECORD_TRAIN_DATA_FILEBASE)))

        _ = (
            pipeline
            | 'Read Test Data' >> read_test_data(test_file_path)
            | 'Encode Test Data' >> beam.Map(test_coder.encode)
            | 'Write Test Data' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, TFRECORD_TEST_DATA_FILEBASE)))

"""--------------------------------------------- Transform Data ----------------------------------------""" 

def transform_data(working_dir):

    with beam.Pipeline() as pipeline:
        with tft_beam.Context(
            temp_dir=os.path.join(working_dir, TRANSFORM_TEMP_DIR)):
            train_coder = tft.coders.ExampleProtoCoder(TRAIN_RAW_DATA_METADATA.schema)
            test_coder = tft.coders.ExampleProtoCoder(TEST_RAW_DATA_METADATA.schema)
            
            train_data = (
                pipeline
                | 'Read Train' >> beam.io.ReadFromTFRecord(
                    os.path.join(working_dir, TFRECORD_TRAIN_DATA_FILEBASE + '*'))
                | 'Decode Train' >> beam.Map(train_coder.decode))
            
            test_data = (
                pipeline
                | 'Read Test' >> beam.io.ReadFromTFRecord(
                    os.path.join(working_dir, TFRECORD_TEST_DATA_FILEBASE + '*'))
                | 'Decode Test' >> beam.Map(test_coder.decode))

            def preprocessing_fn_train(inputs):
                """Preprocess input columns into transformed columns."""
                context = inputs['Context']
                utterance = inputs['Utterance']
                vocab = tf.concat([context, utterance], 0)

                context_tokens = tf.compat.v1.string_split(context, DELIMITERS)
                utterance_tokens = tf.compat.v1.string_split(utterance, DELIMITERS)
                vocab_tokens = tf.compat.v1.string_split(vocab, DELIMITERS)
               
                vocab_mapping_file_path = tft.vocabulary(vocab_tokens, vocab_filename='anantvir_train_vocab')

                mapped_context = tft.apply_vocabulary(context_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                print(mapped_context)

                mapped_utterance = tft.apply_vocabulary(utterance_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)

                return {
                    'Context': mapped_context,
                    'Utterance': mapped_utterance,
                }

            def preprocessing_fn_test(inputs):
                """Preprocess input columns into transformed columns."""
                context = inputs['Context']
                ground_truth_utterance = inputs['Ground Truth Utterance']
                distractor_0 = inputs['Distractor_0']
                distractor_1 = inputs['Distractor_1']
                distractor_2 = inputs['Distractor_2']
                distractor_3 = inputs['Distractor_3']
                distractor_4 = inputs['Distractor_4']
                distractor_5 = inputs['Distractor_5']
                distractor_6 = inputs['Distractor_6']
                distractor_7 = inputs['Distractor_7']
                distractor_8 = inputs['Distractor_8']
                vocab = tf.concat([context, ground_truth_utterance, distractor_0, distractor_1, distractor_2, distractor_3, distractor_4, distractor_5, distractor_6, distractor_7, distractor_8], 0)

                context_tokens = tf.compat.v1.string_split(context, DELIMITERS)
                ground_truth_utterance_tokens = tf.compat.v1.string_split(ground_truth_utterance, DELIMITERS)
                distractor_0_tokens = tf.compat.v1.string_split(distractor_0, DELIMITERS)
                distractor_1_tokens = tf.compat.v1.string_split(distractor_1, DELIMITERS)
                distractor_2_tokens = tf.compat.v1.string_split(distractor_2, DELIMITERS)
                distractor_3_tokens = tf.compat.v1.string_split(distractor_3, DELIMITERS)
                distractor_4_tokens = tf.compat.v1.string_split(distractor_4, DELIMITERS)
                distractor_5_tokens = tf.compat.v1.string_split(distractor_5, DELIMITERS)
                distractor_6_tokens = tf.compat.v1.string_split(distractor_6, DELIMITERS)
                distractor_7_tokens = tf.compat.v1.string_split(distractor_7, DELIMITERS)
                distractor_8_tokens = tf.compat.v1.string_split(distractor_8, DELIMITERS)

                vocab_tokens = tf.compat.v1.string_split(vocab, DELIMITERS)
               
                vocab_mapping_file_path = tft.vocabulary(vocab_tokens, vocab_filename='anantvir_test_vocab')

                mapped_context = tft.apply_vocabulary(context_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_ground_truth_utterance = tft.apply_vocabulary(ground_truth_utterance_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_0 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_1 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_2 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_3 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_4 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_5 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_6 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_7 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)
                mapped_distractor_8 = tft.apply_vocabulary(distractor_0_tokens, deferred_vocab_filename_tensor=vocab_mapping_file_path)

                return {
                    'Context': mapped_context,
                    'Ground Truth Utterance': mapped_ground_truth_utterance,
                    'Distractor_0': mapped_distractor_0,
                    'Distractor_1': mapped_distractor_1,
                    'Distractor_2': mapped_distractor_2,
                    'Distractor_3': mapped_distractor_3,
                    'Distractor_4': mapped_distractor_4,
                    'Distractor_5': mapped_distractor_5,
                    'Distractor_6': mapped_distractor_6,
                    'Distractor_7': mapped_distractor_7,
                    'Distractor_8': mapped_distractor_8,
                }

            # train_transform_fn = (
            #     # data, metadata = dataset
            #     (train_data, TRAIN_RAW_DATA_METADATA)
            #     | 'Analyze' >> tft_beam.AnalyzeDataset(
            #         preprocessing_fn_train))
            
            (transformed_train_data, transformed_train_metadata), train_transform_fn = (
                (train_data, TRAIN_RAW_DATA_METADATA)
                | 'AnalyzeAndTransformTrain' >> tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn_train))
            # https://stackoverflow.com/questions/46406419/collecting-output-from-apache-beam-pipeline-and-displaying-it-to-console

            def print_row(row):
                #raw_inputs = row['Context']
                #padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,padding='post')
                print(row)

            _ = (
                transformed_train_data
                | 'print' >> beam.Map(print_row))

            transformed_train_data_coder = tft.coders.ExampleProtoCoder(
                transformed_train_metadata.schema)

            (transformed_test_data, transformed_test_metadata), test_transform_fn = (
                (test_data, TEST_RAW_DATA_METADATA)
                | 'AnalyzeAndTransformTest' >> tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn_test))
            transformed_test_data_coder = tft.coders.ExampleProtoCoder(
                transformed_test_metadata.schema)

            _ = (
                transformed_train_data
                | 'EncodeTrainData' >> beam.Map(transformed_train_data_coder.encode)
                | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

            _ = (
                transformed_test_data
                | 'EncodeTestData' >> beam.Map(transformed_test_data_coder.encode)
                | 'WriteTestData' >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))


def main():
    input_data_dir = 'dataset_trimmed/data'
    working_dir = tempfile.mkdtemp(dir=input_data_dir)

    train_file_path = os.path.join(input_data_dir, 'train.csv')
    test_file_path = os.path.join(input_data_dir, 'test.csv')

    read_data(train_file_path, test_file_path, working_dir)
    transform_data(working_dir)
    # results = train_and_evaluate(working_dir)

    # pprint.pprint(results)


if __name__ == '__main__':
  main()
