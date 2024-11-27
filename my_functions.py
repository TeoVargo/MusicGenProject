# Collections of utility functions used in music gen project
import sqlite3
import collections
import pandas as pd
import pretty_midi
import numpy as np
import tensorflow as tf


# Function to extract notes from the wjazzd.db and do a join of melody and solo_info
def extract_notes() -> pd.DataFrame :
    try:
        conn = sqlite3.connect("wjazzd.db")
    except Exception as e:
        print(e)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(f"Table Name : {cursor.fetchall()}")

    df = pd.read_sql_query('''
                        SELECT melody.melid, 
                           melody.pitch, 
                           melody.onset as start,
                           melody.onset + melody.duration as end,
                           melody.duration, 
                           solo_info.instrument
                        FROM melody
                        JOIN solo_info
                        ON melody.melid = solo_info.melid
                        ''', conn)
    conn.close()
    return df

# Generate a contour of intervals
def contour(interval: int):
  if 12 >= interval > 4:
    return 10
  elif interval > 12:
    return 20
  elif -12 <= interval < -4:
    return -10
  elif interval < -12:
    return -20
  else:
    return interval

# from tensorFlow MusGen tutorial 
# Create a tensor flow dataset of sequences to use to train the monster.
def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size: int,
    key_order,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch 
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels
    #return inputs, labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

# Function to take our notes and convert them into midi
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

# Function to convert a midi file into a sequence of "Notes"
def midi_to_notes(midi_file: str):
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  instrument = midi_data.instruments[0]
  notes = collections.defaultdict(list)

  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['note_name'].append(pretty_midi.note_number_to_name(note.pitch))
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start-prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
