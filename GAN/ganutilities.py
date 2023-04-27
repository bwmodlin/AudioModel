import pickle
import numpy as np
import pygame
import io
import pretty_midi as pm

# This file contains many useful functions for GAN training and visualization

# Plays a midi file from its path with pygame
def play_midi(midi_file):
    pygame.init()
    pygame.mixer.music.load(midi_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass

    pygame.quit()
    
# Plays a midi file from its raw bytes
def play_midi_from_bytes(midi_file):
    midi_file = io.BytesIO(midi_file)
    pygame.init()
    pygame.mixer.music.load(midi_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass

    pygame.quit()

# deprecated function to add a channel axis (before changing data preprocessing)
def format_batch_dcgan(batch):
    formatted_batch = batch[:, :, :, None]
    return formatted_batch

# gives standardized time resolution for ticks
def get_time_resolution():
    return 0.00089

# this class manages baches for GAN training
class manage_batch_2c():
    def __init__(self, pickle_file, batch_size):
        with open(pickle_file, 'rb') as f:
            self.sequence_matrix = pickle.load(f)
        
        self.batch_size = batch_size
    
    # shuffles dataset and gives new batch (for new epoch)
    def new_epoch(self):
        np.random.shuffle(self.sequence_matrix)
        possible_batches = self.sequence_matrix.shape[0] // self.batch_size
        batches = np.split(self.sequence_matrix[:(self.batch_size*possible_batches), :, :, :], possible_batches, axis=0)
        return batches

# builds tone maps to convert between index and tone for feature matrices
class build_tone_maps():
    def __init__(self):
        tones = [36, 38, 40, 37, 48, 50, 45, 47, 43, 58, 46, 26, 42, 22, 44, 49, 55, 57, 52, 51, 59, 53]
        self.tone_to_index = {}
        self.index_to_tone = {}
        tones.sort()
        
        for i, tone in enumerate(tones):
            self.tone_to_index[tone] = i
            self.index_to_tone[i] = tone
    
    def get_tone_to_index(self):
        return self.tone_to_index
    
    def get_index_to_tone(self):
        return self.index_to_tone

# prepares the generator output to be one channel (with velocity=0 meaning no tone present)
def prepare_generator_output(gen_output, input_ticks):
    velocity_array = np.zeros((input_ticks, 22))
    
    for i in range(input_ticks):
        for j in range(22):
            cell = gen_output[0][i][j]
            
            if cell[0] > 0:
                velocity_array[i][j] = (cell[1] * 64) + 64
                
    return velocity_array

# creates a midi file based on the prepared generator output
def create_midi_file(velocity_array, output_path, tone_map, time_multiply=1, time_offset = 0, velocity_multiplier = 1):
    midi = pm.PrettyMIDI()

    drums = pm.Instrument(program=0, is_drum=True)
    
    time_resolution = get_time_resolution()

    for i, row in enumerate(velocity_array):
        for tone, velocity in enumerate(row):
            if velocity != 0:
                index = i
                # loops down ticks to see how long note lasts
                while (index+1 < velocity_array.shape[0]):
                    if (velocity_array[index+1][tone] != 0):
                        velocity_array[index+1][tone] = 0
                    else:
                        break
                    
                    index+=1
                
                drums.notes.append(pm.Note(
                        
                        velocity = int(min(velocity * velocity_multiplier, 127)),
                        pitch = int(tone_map[tone]),
                        start = i * time_resolution * time_multiply,
                        end = (index + time_offset) * time_resolution * time_multiply
                ))    
    # add the drums instrument to the MIDI file
    midi.instruments.append(drums)

    # write the MIDI file to disk
    midi.write(output_path)

            