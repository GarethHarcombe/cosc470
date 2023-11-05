import pandas as pd
import datetime as dt
from sklearn import svm
import random
from eval import evaluate
from eval import iou
from dataloader import read_activity
from eval_track import get_known_tracks
import pickle
# import tensorflow as tf
# from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
# from tensorflow.keras.models import Model
# import numpy as np
# tf.experimental.numpy.experimental_enable_numpy_behavior()

TIME_THRESHOLD = 12

def grouping_1d(preds, grouping_method="mean"):
    """Given an input series of timestamps, 
    group timestamps together and only take the first timestamp."""
    i = 0
    final_preds = []
    while i < len(preds):
        group = preds[(preds[i] <= preds) & 
                      (preds <= preds[i] + TIME_THRESHOLD)]
        
        if grouping_method == "mean":
            final_preds.append(round(group.mean(), 3))
        else:
            # pick the first one as we want 
            # the start of the lap/change in speed
            final_preds.append(group.iloc[0])
        i = group.last_valid_index() + 1

    return final_preds



class Model:

    def __init__(self, params, requires_training=True, epochs=10):
        self.params = params
        self.requires_training = requires_training
        self.epochs = epochs
        
    def test(self, activities):
        pass

    def train(self, activities):
        pass


class Acceleration(Model):
    
    def __init__(self, params=None, requires_training=False, epochs=None, FULL_PIPELINE=True):
        self.params = params
        self.requires_training = requires_training
        self.epochs = epochs
        self.ACCELERATION_THRES = 0.6
        self.FULL_PIPELINE = FULL_PIPELINE
        track_csv_path = "data/known_track_locations.csv"
        self.tracks = get_known_tracks(track_csv_path)

    def predict(self, points, events=None, laps=None):
        points = points[points.dist_to_track < 100]

        predictions = grouping_1d(points[points.acceleration > self.ACCELERATION_THRES].time_after_start.reset_index(drop=True) - 6, grouping_method="first")

        return predictions

    
    def test(self, activities, metric="evaluation"):
        """Simple predictor that makes predictions off the acceleration of the athlete"""

        errors = []

        for i, row in activities.iterrows():
            # if self.FULL_PIPELINE:
            #     points, laps, events = read_activity("data/" + row.New_Filename)
            # else:
            #     points, laps, events = read_activity("data/" + row.New_Filename, self.tracks[row.TRACK])
            # points.to_pickle("data/" + file[:-3] + "points" + file[-4:])
            # laps.to_pickle("data/"   + file[:-3] + "laps"   + file[-4:])
            # events.to_pickle("data/" + file[:-3] + "events" + file[-4:])
            points = pd.read_pickle("data/" + row.New_Filename[:-3] + "points" + row.New_Filename[-4:])
            laps = pd.read_pickle("data/"   + row.New_Filename[:-3] + "laps"   + row.New_Filename[-4:])
            events = pd.read_pickle("data/" + row.New_Filename[:-3] + "events" + row.New_Filename[-4:])

            predictions = self.predict(points)
            
            print("Predicted: ", predictions)
            print("Actual: ", laps.time_after_start.values)
            if metric == "evaluate":
                errors.append(evaluate(laps.time_after_start.values, predictions))
            elif metric == "iou":
                errors.append(iou(laps.time_after_start.values, predictions))
            else:
                errors.append(tuple(evaluate(laps.time_after_start.values, predictions)) + 
                              (iou(laps.time_after_start.values, predictions),))

        return errors
    

    def train(self, activities):
        raise TypeError("This model does not require training")



class SlidingWindow(Model):

    def __init__(self, params=None, requires_training=True, epochs=1, DIVISOR=4, FULL_PIPELINE=True):
        super().__init__(params, requires_training, epochs)
        self.CLASS_WEIGHT = 230   # 250
        self.clf = svm.SVC(kernel='rbf', class_weight={0: 1.0, 1: self.CLASS_WEIGHT / DIVISOR}) # positive class is weighted higher
        self.FULL_PIPELINE = FULL_PIPELINE
        if not self.FULL_PIPELINE:
            self.CLASS_WEIGHT = min([230, self.CLASS_WEIGHT])
        track_csv_path = "data/known_track_locations.csv"
        self.tracks = get_known_tracks(track_csv_path)

    def save(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self.clf, f)


    def load(self, filename):
        with open(filename, 'rb') as f:
            self.clf = pickle.load(f)
        
    def generate_slices(self, points, events, laps=None):
        """df with 6 rows, third row is at t
        1 if there is a lap time in interval [t, t+1)
        0 if not"""
        time_slice_duration = 1  # in seconds
        num_padding_points = 7   # number of points either side of the query time.

        start_time = int(points.iloc[num_padding_points].time_after_start)
        end_time = int(points.iloc[-num_padding_points-1].time_after_start)

        data = []
        labels = []
        times = []

        for t in range(start_time, end_time, time_slice_duration):
            if laps is not None:
                label = 1 if len(laps[(t <= laps.time_after_start) & (laps.time_after_start < t+1)]) > 0 else 0
                labels.append(label)

            query_time = pd.Timestamp(events.iloc[0].Timestamp + dt.timedelta(seconds=t))
            closest_index = points.set_index('Timestamp').index.get_indexer([query_time], method='nearest')[0]

            subset = points[closest_index-num_padding_points:closest_index+num_padding_points][["track_x", "track_y", 
                                                                                                "dist_to_track", "speed",
                                                                                                "acceleration", "time_after_start"]]
            subset["time"] = subset.time_after_start - t
            subset = subset.drop(columns=["time_after_start"])

            data.append(subset.values.ravel())
            times.append(t)

        return data, labels, times
    
    def generate_data(self, activities):
        data = []
        labels = []
        window_times = []
        all_laps = []

        for i, row in activities.iterrows():
            if self.FULL_PIPELINE:
                points, laps, events = read_activity("data/" + row.New_Filename)
            else:
                points, laps, events = read_activity("data/" + row.New_Filename, self.tracks[row.TRACK])
            # print(file)
            # points, laps, events = read_activity("data/" + file)
            # points.to_pickle("data/" + file[:-3] + "points" + file[-4:])
            # laps.to_pickle("data/"   + file[:-3] + "laps"   + file[-4:])
            # events.to_pickle("data/" + file[:-3] + "events" + file[-4:])
            # points = pd.read_pickle("data/" + file[:-3] + "points" + file[-4:])
            # laps = pd.read_pickle("data/"   + file[:-3] + "laps"   + file[-4:])
            # events = pd.read_pickle("data/" + file[:-3] + "events" + file[-4:])

            points = points[points.dist_to_track < 100]

            new_data, new_labels, times = self.generate_slices(points, events, laps)
            data.append(new_data)
            labels.append(new_labels)
            window_times.append(times)
            all_laps.append(laps)

        return data, labels, window_times, all_laps

    def predict(self, points, events, laps=None):
        data, _, window_times = self.generate_slices(points, events, laps)

        test_labels_preds = self.clf.predict(data)
        workout_output = []

        for i, label in enumerate(test_labels_preds):
            if label == 1:
                workout_output.append(window_times[i])

        return grouping_1d(pd.Series(workout_output))

    def generate_predictions(self, test_data, test_window_times):
        output = []

        for data, window_times in zip(test_data, test_window_times):
            test_labels_preds = self.clf.predict(data)
            workout_output = []

            for i, label in enumerate(test_labels_preds):
                if label == 1:
                    workout_output.append(window_times[i])

            output.append(workout_output)

        return output

    def preprocess_train_data(self, data, labels):
        flattened_data = []
        flattened_labels = []

        for data, label in zip(data, labels):
            flattened_data += data
            flattened_labels += label

        print("Num training examples", len(flattened_data))

        pos_trains = sum(flattened_labels)  # number of positive training examples
        neg_trains = len(flattened_labels) - pos_trains
        
        neg_indices = [i for i, label in enumerate(flattened_labels) if label == 0]
        pos_indices = [i for i, label in enumerate(flattened_labels) if label == 1]

        sampled_indices = random.sample(neg_indices, int(pos_trains * self.CLASS_WEIGHT))
        new_indices = set(pos_indices) | set(sampled_indices)

        condensed_data = []
        condensed_labels = []

        for i, (data, label) in enumerate(zip(flattened_data, flattened_labels)):
            if i in new_indices:
                condensed_data.append(data)
                condensed_labels.append(label)

        print("New Num training examples", len(condensed_data))
        return condensed_data, condensed_labels

    def train(self, activities):
        train_data, train_labels, _, _ = self.generate_data(activities)

        condensed_data, condensed_labels = self.preprocess_train_data(train_data, train_labels)
        
        self.clf.fit(condensed_data, condensed_labels)

    def test(self, activities, metric="evaluate"):
        test_data, test_labels, test_window_times, laps = self.generate_data(activities)

        output = self.generate_predictions(test_data, test_window_times)
        
        errors = []

        for i, workout_output in enumerate(output):
            workout_output = grouping_1d(pd.Series(workout_output))  # thin the predictions
            # if i % 10 == 0:
            print("Predicted: ", workout_output)
            print("Actual: ", laps[i].time_after_start.values)
            if metric == "evaluate":
                errors.append(evaluate(laps[i].time_after_start.values, workout_output))
            elif metric == "iou":
                errors.append(iou(laps[i].time_after_start.values, workout_output))
            else:
                errors.append(tuple(evaluate(laps[i].time_after_start.values, workout_output)) + 
                              (iou(laps[i].time_after_start.values, workout_output),))

        return errors


# class EncoderDecoder(tf.keras.Model):

#   def __init__(self):
#     super().__init__()
#     self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#     self.max_input_seq_length = 2000
#     self.max_output_seq_length = 50

#     # Define the vocabulary sizes for input and output sequences
#     self.input_size = 6  # ["track_x", "track_y", "dist_to_track", "speed", "acceleration", "time_after_start"]
#     self.output_size = 1  # ["time"]

#     # Define the dimensionality of the hidden state of the LSTM
#     self.hidden_units = 64   # TODO: may need to be 6??

#     # Define the encoder model
#     # encoder_inputs = Input(shape=(self.max_input_seq_length, self.input_size))
#     # encoder_embedding = Embedding(input_vocab_size, hidden_units)(encoder_inputs)
#     self.encoder = LSTM(self.hidden_units, return_state=True)

#     # Define the decoder model
#     # decoder_inputs = Input(shape=(self.max_output_seq_length, self.output_size))
#     # decoder_embedding = Embedding(output_vocab_size, hidden_units)(decoder_inputs)

#     # We set up our decoder to return full output sequences and to return internal states as well.
#     self.decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True)

#     self.decoder_dense = Dense(self.output_size, activation='softmax')

#   def call(self, encoder_inputs, decoder_inputs, training=False):
#     # encoder_inputs, decoder_inputs = inputs

#     encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
    
#     # We discard `encoder_outputs` and only keep the states
#     encoder_states = [state_h, state_c]

#     decoder_outputs, _, _ = self.decoder_lstm(decoder_inputs, initial_state=encoder_states)

#     decoder_outputs = self.decoder_dense(decoder_outputs)

#     return decoder_outputs


# class Encoder(Model):
    
#     def __init__(self, params=None, requires_training=True, epochs=10, FULL_PIPELINE=True):
#         super().__init__(params, requires_training, epochs)
#         self.FULL_PIPELINE = FULL_PIPELINE

#         self.model = EncoderDecoder()

#         # Compile the model
#         # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#     def save(self, filename):
#         self.model.save("encoder_decoder.h5")   # 'model_name.h5'

#     def load(self, filename):
#         self.model = tf.keras.models.load_model("encoder_decoder.h5")

#     def predict(self, points, events, laps=None):
#         data, _, window_times = self.generate_slices(points, events, laps)

#         test_labels_preds = self.clf.predict(data)
#         workout_output = []

#         for i, label in enumerate(test_labels_preds):
#             if label == 1:
#                 workout_output.append(window_times[i])

#         return grouping_1d(pd.Series(workout_output))

#     def pad2d(self, tensor, length, addition=1.0):
#         return np.append(tensor, np.zeros((length - tensor.shape[0], tensor.shape[-1])) + addition).reshape(-1, tensor.shape[-1])

#     def pad1d(self, tensor, length, addition=1.0):
#         return np.append(tensor, np.zeros((length - tensor.shape[0],)) + addition)


#     def generate_data(self, activities):
#         # Example training data preparation (not the actual implementation):
#         encoder_input_data = []
#         decoder_input_data = []
#         decoder_target_data = []
#         seq_lengths = []
#         pred_lengths = []

#         for i, row in activities.iterrows():
#             # if self.FULL_PIPELINE:
#             #     points, laps, events = read_activity("data/" + row.New_Filename)
#             # else:
#             #     points, laps, events = read_activity("data/" + row.New_Filename, self.tracks[row.TRACK])

#             points = pd.read_pickle("data/" + row.New_Filename[:-3] + "points" + row.New_Filename[-4:])
#             laps = pd.read_pickle("data/"   + row.New_Filename[:-3] + "laps"   + row.New_Filename[-4:])
#             events = pd.read_pickle("data/" + row.New_Filename[:-3] + "events" + row.New_Filename[-4:])

#             points = points[["track_x", "track_y", "dist_to_track", "speed", "acceleration", "time_after_start"]]
#             laps = laps[:-1].time_after_start / points.iloc[-1].time_after_start

#             seq_lengths.append(len(points)-1)
#             pred_lengths.append(len(laps.values))

#             encoder_inputs = tf.convert_to_tensor(self.pad2d(points.values, 
#                                                            self.model.max_input_seq_length, 0.0), dtype=tf.float32)
#             decoder_inputs = tf.convert_to_tensor(self.pad1d(np.insert(laps.values, 0, 0.0),
#                                                            self.model.max_output_seq_length).reshape(-1, 1), dtype=tf.float32)
#             decoder_target = tf.convert_to_tensor(self.pad1d(np.append(laps.values, 1.0), 
#                                                            self.model.max_output_seq_length), dtype=tf.float32)

#             # print(encoder_inputs)
#             # print(decoder_inputs)
#             # print(decoder_target)

#             encoder_input_data.append(encoder_inputs)
#             decoder_input_data.append(decoder_inputs)
#             decoder_target_data.append(decoder_target)

#         return encoder_input_data, decoder_input_data, decoder_target_data, seq_lengths, pred_lengths

#     def train(self, activities):
#         # Now, you can train this model using your dataset. You'll need to preprocess your input and output sequences as required and one-hot encode your output sequences.
#         encoder_input_data, decoder_input_data, decoder_target_data, seq_lengths, pred_lengths = self.generate_data(activities)
        
#         for i in range(len(encoder_input_data)):
#             input1 = tf.expand_dims(encoder_input_data[i], axis=0)
#             input2 = tf.expand_dims(decoder_input_data[i], axis=0)
#             output = tf.expand_dims(decoder_target_data[i], axis=0)
#             with tf.GradientTape() as tape:
#                 predictions = self.model(input1, input2).reshape(((-1, self.model.max_output_seq_length)))
#                 loss = tf.keras.metrics.categorical_crossentropy(output, predictions)

#             gradients = tape.gradient(loss, self.model.trainable_variables)
#             self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  


#     def test(self, activities, metric="evaluate"):
#         # Once the model is trained, you can use it for inference by defining separate encoder and decoder models for the encoding and decoding processes.
#         encoder_input_data, decoder_input_data, decoder_target_data, seq_lengths, pred_lengths = self.generate_data(activities)

#         errors = []

#         for i in range(len(encoder_input_data)):
#             input1 = tf.expand_dims(encoder_input_data[i], axis=0)
#             input2 = tf.expand_dims(decoder_input_data[i], axis=0)
#             # output = tf.expand_dims(decoder_target_data[i], axis=0)

#             outputs = self.model(input1, input2)
#             # print("Outputs: ", outputs)
#             outputs = outputs[outputs < 0.99]
#             # print("Outputs: ", outputs)
#             preds = (outputs * encoder_input_data[i][seq_lengths[i]][5]).numpy().reshape((-1))
#             actual = (decoder_target_data[i] * encoder_input_data[i][seq_lengths[i]][5]).numpy().reshape((-1))[:pred_lengths[i]]
#             # print("Predicted: ", preds)
#             # print("Actual: ", actual)

#             if metric == "evaluate":
#                 errors.append(evaluate(actual, preds))
#             elif metric == "iou":
#                 errors.append(iou(actual, preds))
#             else:
#                 errors.append(tuple(evaluate(actual, preds)) + 
#                               (iou(actual, preds),))

#         return errors