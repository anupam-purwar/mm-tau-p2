import time
import datetime

import soundfile as sf
from scipy.io import wavfile
import os
import shutil

import logging
logger = logging.getLogger(__name__)

class LLMResponseWrapper:
    
    def __init__(self, iterator):
        self.iterator = iter(iterator)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.iterator)
        return chunk.choices[0].delta.content        

def read_file(path):
    with open(path, 'r') as f:
        out = f.read()
    return out

def update_file(path, data):
    with open(path, 'a+') as f:
        f.write(str(data))
    return

def stitch_wav_files(wav_buffer=None, file_out='', out_dir = '', save_files = False, sample_rate=44100, temp_dir='.temp'):
        if not wav_buffer:
            return None
        file_out = os.path.join(temp_dir, file_out) if file_out else os.path.join(temp_dir, str(datetime.datetime.now()).replace(' ','-')+'_stiched.wav')
        wav_buffer_data = []
        rate = sample_rate
        for file in wav_buffer:
            try:
                wav_buffer_data.extend(wavfile.read(file)[1])
                if save_files:
                    shutil.move(file, f'{out_dir}/')
                else:
                    os.remove(file)
            except FileNotFoundError:
                logger.warning('File Not Found while stitching: %s', file)
            except Exception as e:
                logger.error('Error Stitching Files. Current File (%s) Generating File %s: %s', file, file_out, e)
            
        sf.write(file_out, wav_buffer_data, rate)
        return file_out

def pretty_print(conversation:list[dict[str,str]]):
    for msg in conversation:
        if msg['role'] == 'tool':
            continue
        print(f"{msg['role'].upper()}: {msg['content']}")