from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm

import numpy as np
import music21
import os

NOTES              = 96
SAMPLE_PER_MEASURE = 96
MEASURES           = 16
KEYS               = 12

MAJORS = dict([('A-', 4),('G#', 4),('A', 3),('A#', 2),('B-', 2),('B', 1),('C', 0),('C#', -1),('D-', -1),('D', -2),('D#', -3),('E-', -3),('E', -4),('F', -5),('F#', 6),('G-', 6),('G', 5)])
MINORS = dict([('G#', 1), ('A-', 1),('A', 0),('A#', -1),('B-', -1),('B', -2),('C', -3),('C#', -4),('D-', -4),('D', -5),('D#', 6),('E-', 6),('E', 5),('F', 4),('F#', 3),('G-', 3),('G', 2)])

def info(score):
    key = score.analyze('key')
    return (key.tonic.name, key.mode)

def read(path):
    score = music21.converter.parse(path)
    return score

def transpose(path, destination_dir):
    try:
        score = music21.converter.parse(path)
    except:
        print(f'Removed {path}: can\'t be parsed')
        os.remove(path)
        return

    key   = score.analyze('key')

    if key.mode == "major":
        halfSteps = MAJORS[key.tonic.name]

    elif key.mode == "minor":
        halfSteps = MINORS[key.tonic.name]

    newscore = score.transpose(halfSteps)
    key      = newscore.analyze('key')

    new_path = os.path.join(destination_dir, f"transposed_{path.split('/')[-1]}")
    newscore.write('midi', fp=new_path)

def image(path):
    mid = MidiFile(path)

    ticks_per_beat    = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat

    time_sig  = False
    flag_warn = False

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                new_tmp = msg.numerator * ticks_per_measure / msg.denominator

                if time_sig and new_tmp != ticks_per_measure:
                    flag_warn = True

                ticks_per_measure = new_tmp
                time_sig          = True

    if flag_warn:
        return []

    notes = {}
    for track in mid.tracks:
        abs_t = 0
        note  = None

        for msg in track:
            abs_t += msg.time
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    continue

                note = (msg.note - (128 - NOTES) * 0.5) % NOTES

                if note not in notes:
                    notes[note] = []
                else:
                    single_note = notes[note][-1]
                    if len(single_note) == 1:
                        single_note.append(single_note[0] + 1)

                notes[note].append([abs_t * SAMPLE_PER_MEASURE / ticks_per_measure])

            elif note != None and msg.type == 'note_off':
                if len(notes[note][-1]) != 1:
                    continue

                notes[note][-1].append(abs_t * SAMPLE_PER_MEASURE / ticks_per_measure)

    for note in notes:
        for start_end in notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0] + 1)

    samples = []
    for note in notes:
        for start, end in notes[note]:
            sample_ix = int(np.floor(start / SAMPLE_PER_MEASURE))
            while len(samples) <= sample_ix:
                samples.append(np.zeros((SAMPLE_PER_MEASURE, NOTES), dtype=np.uint8))

            sample = samples[sample_ix]
            start_ix = int(np.floor(start - sample_ix * SAMPLE_PER_MEASURE))

            sample[start_ix, int(np.floor(note))] = 1

    return samples

def cut(samples):
    while len(samples) < MEASURES:
        samples.append(np.zeros((SAMPLE_PER_MEASURE, NOTES), dtype=np.uint8))

    if len(samples) == MEASURES:
        return samples

    middle = int(np.floor((len(samples) - 1) * 0.5))
    step   = int(MEASURES * 0.5)
    start  = middle - step
    end    = middle + step

    return samples[start:end]

def save(samples, path, thresh=0.5):
    mid   = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat    = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat
    ticks_per_sample  = ticks_per_measure / SAMPLE_PER_MEASURE

    abs_t  = 0
    last_t = 0

    for sample in samples:
    	for y in range(sample.shape[0]):
            abs_t += ticks_per_sample
            for x in range(sample.shape[1]):
                note = int(np.floor(x + (128 - NOTES) * 0.5))

                if sample[y, x] >= thresh and (y == 0 or sample[y - 1, x] < thresh):
                    delta_t = int(np.floor(abs_t - last_t))
                    track.append(Message('note_on', note=note, velocity=127, time=delta_t))
                    last_t  = abs_t

                if sample[y, x] >= thresh and (y == sample.shape[0] - 1 or sample[y + 1, x] < thresh):
                    delta_t = int(np.floor(abs_t - last_t))
                    track.append(Message('note_off', note=note, velocity=127, time=delta_t))
                    last_t  = abs_t

    mid.save(path)

def shift(imgs, step):
    assert(step > 0 and step < (NOTES // KEYS))

    split   = 8 * KEYS - step * KEYS
    for i in range(len(imgs)):
        halfs   = [imgs[i][split:], imgs[i][:split]]
        imgs[i] = np.concatenate(halfs)

    return imgs

def build_transpose(data_dir, destination_dir):
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)

    midi_names = [name for name in os.listdir(data_dir) if '.mid' in name]
    midi_paths = [os.path.join(data_dir, name) for name in midi_names]

    pbar = tqdm(enumerate(midi_names), total=len(midi_names), desc='Building midi transpose')
    for i, midi_name in pbar:
        transpose(midi_paths[i], destination_dir)
        pbar.set_description(f'Building midi transpose {midi_name}')

def save_imgs(imgs, path):
    np.save(path, np.array(imgs))

def build_images(data_dir, t_data_dir, destination_dir):
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)

    midi_names = [name for name in os.listdir(data_dir) if '.mid' in name]
    midi_paths = [os.path.join(data_dir, name) for name in midi_names]

    t_midi_names = [name for name in os.listdir(t_data_dir) if '.mid' in name]
    t_midi_paths = [os.path.join(t_data_dir, name) for name in t_midi_names]

    names = midi_names + t_midi_names
    paths = midi_paths + t_midi_paths

    idx  = 0
    pbar = tqdm(paths, total=len(paths), desc='Building images')
    for path in pbar:
        try:
            infos  = info(read(path))
            folder = f'{infos[0]}{infos[1]}'
        except:
            print(f'Failed to read {path} info')
            continue

        destination_dir_folder = os.path.join(destination_dir, folder)
        if not os.path.isdir(destination_dir_folder):
            os.mkdir(destination_dir_folder)

        try:
            tracks = [cut(image(path))]
        except:
            print(f'Failed to read {path} as an image')
            continue

        for i in range(1, 8):
            tracks.append(shift(tracks[0], i))

        for track in tracks:
            save_imgs(track, os.path.join(
                destination_dir_folder,
                f'sound_{idx:06d}.mid'
            ))
            idx += 1
