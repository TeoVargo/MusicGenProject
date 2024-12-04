# Collections of utility functions used in music gen project
import sqlite3
import collections
import pandas as pd
import pretty_midi
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from typing import Optional
import seaborn as sns



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
                           melody.pitch / 128 as pitch_norm,
                           melody.onset as start,
                           melody.onset + melody.duration as end,
                           melody.duration, 
                           solo_info.instrument,
                           solo_info.key,
                           solo_info.style,
                           solo_info.avgtempo as tempo,
                           solo_info.rhythmfeel as feel,
                           solo_info.title,
                           solo_info.performer
                        FROM melody
                        JOIN solo_info
                        ON melody.melid = solo_info.melid
                        ''', conn)
    conn.close()
    return df

# Generate a contour of intervals
def contour(interval: int):
  if 12 >= interval > 4:
    return 8
  elif interval > 12:
    return 16
  elif -12 <= interval < -4:
    return -8
  elif interval < -12:
    return -16
  else:
    return interval

# Function to take a frame of notes and convert them into a midi file
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI(initial_tempo=notes['tempo'][0])
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  start = 0
  for i, note in notes.iterrows():
    next_start = float(start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    start = next_start

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

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))
  
  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))