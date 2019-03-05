from mido import MidiFile, MidiTrack, Message
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

import numpy as np
import music21
import os

NOTES              = 96
SAMPLE_PER_MEASURE = 96
MEASURES           = 16
KEYS               = 12

TIME_SIGNATURE = 'time_signature'
NOTE_ON        = 'note_on'
NOTE_OFF       = 'note_off'
VELOCITY       = 127

def get_label(path):
    try:
        score = music21.converter.parse(path)
        key   = score.analyze('key')
        label = f'{key.tonic.name}_{key.mode}'
        return label
    except:
        return None

def find_tpm(tracks, tpm):
    for track in tracks:
        for msg in track:
            if msg.type == TIME_SIGNATURE:
                new_tpm = msg.numerator * tpm / msg.denominator

                if new_tpm != tpm: return None
                tpm = new_tpm

    return tpm

def find_notes(tracks, tpm):
    notes = {}

    for track in tracks:
        abs_t = 0
        note  = None

        for msg in track:
            abs_t += msg.time

            if msg.type == NOTE_ON:
                if msg.velocity == 0:
                    continue

                note = int(np.floor((msg.note - (128 - NOTES) * 0.5) % NOTES))
                if note not in notes:
                    notes[note] = []
                else:
                    if len(notes[note][-1]) == 1:
                        notes[note][-1].append(notes[note][-1][0] + 1)
                notes[note].append([int(np.floor(abs_t * SAMPLE_PER_MEASURE / tpm))])

            if note != None and msg.type == NOTE_OFF:
                if len(notes[note][-1]) != 1:
                    continue

                notes[note][-1].append(int(np.floor(abs_t * SAMPLE_PER_MEASURE / tpm)))

    for note in notes:
        for start_end in notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0] + 1)

    return notes

def find_measures(notes):
    max_measure = int(np.ceil(
        np.max([
            np.max(notes[note])
            for note in notes
        ]) / SAMPLE_PER_MEASURE
    ))

    measures = np.zeros((NOTES, max_measure * SAMPLE_PER_MEASURE))

    for note in notes:
        for start, end in notes[note]:
            measures[note, start] = 1

    return measures

def midi2img(path):
    try:
        midi = MidiFile(path)

        tpb = midi.ticks_per_beat
        tpm = 4 * tpb
        tpm = find_tpm(midi.tracks, tpm)
        if tpm is None: return None

        notes    = find_notes(midi.tracks, tpm)
        measures = find_measures(notes)

        return np.array(measures, dtype=np.uint8)
    except:
        return None

def img2midi(img, path, thresh=0.5):
    midi  = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    tpb = midi.ticks_per_beat
    tpm = 4 * midi.ticks_per_beat
    tps = tpm / SAMPLE_PER_MEASURE

    abs_t  = 0
    last_t = 0

    for t in range(img.shape[1]):
        abs_t += tps

        for n in range(img.shape[0]):
            note = int(np.floor(n + (128 - NOTES) * 0.5))

            if img[n, t] >= thresh and (t == 0 or img[n, t - 1] < thresh):
                detla_t = int(np.floor(abs_t - last_t))
                track.append(Message(NOTE_ON, note=note, velocity=VELOCITY, time=detla_t))
                last_t = abs_t

            if img[n, t] >= thresh and (t == img.shape[1] - 1 or img[n, t + 1] < thresh):
                detla_t = int(np.floor(abs_t - last_t))
                track.append(Message(NOTE_OFF, note=note, velocity=VELOCITY, time=detla_t))
                last_t = abs_t

    midi.save(path)

def plot(fig, ax, measures):
    ax.imshow(measures, origin='lower')
    ax.axis('off')

def save_img(measures, path):
    img = Image.fromarray(measures * 255)
    img.save(path)

def load_img(path):
    img = Image.open(path)
    return np.array(img, dtype=np.uint8) / 255

def build_image(data):
    path            = data['path']
    destination_dir = data['dst_dir']

    label = get_label(path)
    if label is None: return

    dst_dir = os.path.join(destination_dir, label)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    measures = midi2img(path)
    if measures is None: return

    save_img(measures, os.path.join(
        dst_dir,
        path.split('/')[-1].replace('.mid', '.png')
    ))

def build(data_dir, destination_dir):
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)

    files = [{
        'path'   : os.path.join(data_dir, file),
        'dst_dir': destination_dir
    } for file in os.listdir(data_dir)]

    with Pool(4) as p:
        res = list(p.imap(build_image, tqdm(
            files,
            total = len(files),
            desc  = 'Building dataset'
        )))
        p.close()
        p.join()
