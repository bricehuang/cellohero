#####################################################################
#
# input_demo.py
#
# Copyright (c) 2017, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

# contains example code for some simple input (microphone) processing.
# Requires aubio (pip install aubio).


import sys
sys.path.append('..')

from common.core import *
from common.audio import *
from common.writer import *
from common.mixer import *
from common.gfxutil import *
from common.wavegen import *
from buffers import *

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate

from random import randint
import aubio

NUM_CHANNELS = 2

class PitchDetector(object):
    def __init__(self):
        super(PitchDetector, self).__init__()
        # number of frames to present to the pitch detector each time
        self.buffer_size = 1024

        # set up the pitch detector
        self.pitch_o = aubio.pitch("yin", 2048, self.buffer_size, Audio.sample_rate)
        self.pitch_o.set_tolerance(.5)
        self.pitch_o.set_unit("midi")

        # buffer allows for always delivering a fixed buffer size to the pitch detector
        self.buffer = FIFOBuffer(self.buffer_size * 8, buf_type=np.float32)

        self.cur_pitch = 0

    # Add incoming data to pitch detector. Return estimated pitch as floating point
    # midi value.
    # Returns 0 if a strong pitch is not found.
    def write(self, signal):
        conf = 0

        self.buffer.write(signal) # insert data

        # read data in the fixed chunk sizes, as many as possible.
        # keep only the highest confidence estimate of the pitches found.
        while self.buffer.get_read_available() > self.buffer_size:
            p, c = self._process_window(self.buffer.read(self.buffer_size))
            if c > conf:
                self.cur_pitch = p
        return self.cur_pitch

    # helper function for finding the pitch of the fixed buffer signal.
    def _process_window(self, signal):
        pitch = self.pitch_o(signal)[0]
        conf = self.pitch_o.get_confidence()
        return pitch, conf

# Same as WaveSource interface, but is given audio data explicitly.
class WaveArray(object):
    def __init__(self, np_array, num_channels):
        super(WaveArray, self).__init__()

        self.data = np_array
        self.num_channels = num_channels

    # start_frame and end_frame are in units of frames,
    # so take into account num_channels when accessing sample data
    def get_frames(self, start_frame, end_frame) :
        start_sample = start_frame * self.num_channels
        end_sample = end_frame * self.num_channels
        return self.data[start_sample : end_sample]

    def get_num_channels(self):
        return self.num_channels

# this class is a generator. It does no actual buffering across more than one call.
# So underruns/overruns are likely, resulting in pops here and there.
# But code is simpler to deal with and it reduces latency.
# Otherwise, it would need a FIFO read-write buffer
class IOBuffer(object):
    def __init__(self):
        super(IOBuffer, self).__init__()
        self.buffer = None

    # add data
    def write(self, data):
        self.buffer = data

    # send that data to audio
    def generate(self, num_frames, num_channels) :
        num_samples = num_channels * num_frames

        # if nothing was added, just send out zeros
        if self.buffer is None:
            return np.zeros(num_samples), True

        # if the data added recently is not of the proper size, just resize it.
        # this will cause some pops here and there. So, not great for a real solution,
        # but ok for now.
        if num_samples != len(self.buffer):
            tmp = self.buffer.copy()
            tmp.resize(num_samples)
            if num_samples < len(self.buffer):
                print('IOBuffer:overrun')
            else:
                print('IOBuffer:underrun')

        else:
            tmp = self.buffer

        # clear out buffer because we just used it
        self.buffer = None
        return tmp, True

NOTE_SPEED = 200

NOW_BAR_X = 100
PADDING = 100

BOTTOM_LANE_PITCH = 48
TOP_LANE_PITCH = 60
LANE_HEIGHT = (Window.height - 2 * PADDING) / (TOP_LANE_PITCH - BOTTOM_LANE_PITCH + 1)


# display for a single note at a position
class NoteDisplay(InstructionGroup):
    LIVE = 'live'
    HIT = 'hit'
    MISSED = 'missed'
    def __init__(self, pitch, start_time, duration):
        super(NoteDisplay, self).__init__()
        self.pitch = pitch
        self.start_time = start_time
        self.duration = duration

        self.status = NoteDisplay.LIVE

        vert_position = np.interp(
            pitch,
            (BOTTOM_LANE_PITCH, TOP_LANE_PITCH),
            (0+PADDING, Window.height - PADDING - LANE_HEIGHT)
        )
        horiz_position = NOW_BAR_X + start_time * NOTE_SPEED

        self.pos = np.array([horiz_position, vert_position])

        self.color = Color(1,1,1)
        self.add(self.color)

        self.rect = Rectangle(pos=self.pos, size=(duration*NOTE_SPEED, LANE_HEIGHT))
        self.add(self.rect)

    def on_hit(self):
        self.status = NoteDisplay.HIT
        self.color.rgb = (0,1,0)

    def on_pass(self):
        self.status = NoteDisplay.MISSED
        self.color.rgb = (1,0,0)

    def on_update(self, dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.rect.pos = self.pos

    def get_x_bounds(self):
        return (self.pos[0], self.pos[0] + self.duration*NOTE_SPEED)

class BeatMatchDisplay(InstructionGroup):
    def __init__(self, note_info):
        # note_info is List[(pitch, start_time, duration)]
        super(BeatMatchDisplay, self).__init__()

        self.add(Line(points=[NOW_BAR_X,0,NOW_BAR_X,Window.height], width=2))
        self.notes = []
        for pitch, start_time, duration in note_info:
            note = NoteDisplay(pitch, start_time, duration)
            self.add(note)
            self.notes.append(note)

    # called by Player. Causes the right thing to happen
    def note_hit(self, gem_idx):
        self.notes[gem_idx].on_hit()

    # called by Player. Causes the right thing to happen
    def note_pass(self, gem_idx):
        self.notes[gem_idx].on_pass()

    # call every frame to make gems and barlines flow down the screen
    def on_update(self, dt) :
        # update all components that animate
        for note in self.notes:
            note.on_update(dt)

    def current_note(self):
        for i in range(len(self.notes)):
            note = self.notes[i]
            left, right = note.get_x_bounds()
            if left <= NOW_BAR_X and right >= NOW_BAR_X:
                return (i, note.pitch)
        return None

    def missed_notes(self):
        idxes = []
        for i in range(len(self.notes)):
            note = self.notes[i]
            left, right = note.get_x_bounds()
            if note.status == NoteDisplay.LIVE and right < NOW_BAR_X:
                idxes.append(i)
        return idxes

ACCEPTABLE_PITCH_INTERVAL = 0.2
class Cellist(object):
    def __init__(self, display, score_cb):
        super(Cellist, self).__init__()
        self.display = display
        self.cur_pitch = 0
        self.score = 0
        self.score_cb = score_cb

    def notify_pitch(self, pitch):
        self.cur_pitch = pitch

    def on_update(self):
        current_note = self.display.current_note()
        if current_note is not None:
            idx, pitch = current_note
            pitch_diff = np.abs(pitch - self.cur_pitch)
            if pitch_diff < ACCEPTABLE_PITCH_INTERVAL:
                if self.display.notes[idx].status == NoteDisplay.LIVE:
                    self.score += 1000
                    self.score_cb(self.score)
                self.display.note_hit(idx)
        missed_notes = self.display.missed_notes()
        for idx in missed_notes:
            self.display.note_pass(idx)

SEC_PER_MIN = 60.
class SongData(object):
    def __init__(self, filepath):
        self.notes = []
        f = open(filepath, 'r')
        header = f.readline()
        bpm = float(header.rstrip())
        bps = bpm / SEC_PER_MIN

        body_line = f.readline()
        while body_line:
            line = body_line.rstrip()
            pitch, start, duration = line.split()
            self.notes.append((float(pitch), float(start)/bps, float(duration)/bps))
            body_line = f.readline()
        f.close()

class MainWidget1(BaseWidget):
    def __init__(self):
        super(MainWidget1, self).__init__()

        self.audio = Audio(NUM_CHANNELS, input_func=self.receive_audio)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.io_buffer = IOBuffer()
        self.mixer.add(self.io_buffer)
        self.pitch = PitchDetector()

        self.recording = False
        self.channel_select = 0
        self.input_buffers = []
        self.live_wave = None

        self.info = topleft_label()
        self.add_widget(self.info)

        self.cur_pitch = 0
        self.song_data = SongData('input.txt')

        self.objects = AnimGroup()
        self.canvas.add(self.objects)
        self.display = BeatMatchDisplay(self.song_data.notes)
        self.objects.add(self.display)

        self.cellist = Cellist(self.display, self.update_score)
        self.score = 0

        # note_names = ['C3','C#3','D3','Eb3','E3','F3','F#3','G3','Ab3','A3','Bb3','B3','C4']
        # def _build_note(text, lane):
        #     with self.canvas:
        #         label = Label(font_size=40)
        #         label.pos = (10,PADDING + 26 + (lane + 0.5) * LANE_HEIGHT )
        #         label.text = text
        # _build_note('C3',0)
        # _build_note('D3',2)
        # _build_note('C#3',1)

    def update_score(self, score):
        self.score = score

    def on_update(self) :
        self.audio.on_update()
        self.cellist.on_update()
        self.objects.on_update()

        self.info.text = "pitch: %.1f\n" % self.cur_pitch
        self.info.text += "score: %d\n" % self.score

    def receive_audio(self, frames, num_channels) :
        # handle 1 or 2 channel input.
        # if input is stereo, mono will pick left or right channel. This is used
        # for input processing that must receive only one channel of audio (RMS, pitch, onset)
        if num_channels == 2:
            mono = frames[self.channel_select::2] # pick left or right channel
        else:
            mono = frames

        # Microphone volume level, take RMS, convert to dB.
        # display on meter and graph
        rms = np.sqrt(np.mean(mono ** 2))
        rms = np.clip(rms, 1e-10, 1) # don't want log(0)
        db = 20 * np.log10(rms)      # convert from amplitude to decibels

        # pitch detection: get pitch and display on meter and graph
        self.cur_pitch = self.pitch.write(mono)
        self.cellist.notify_pitch(self.cur_pitch)

        # record to internal buffer for later playback as a WaveGenerator
        if self.recording:
            self.input_buffers.append(frames)

    def on_key_down(self, keycode, modifiers):
        # toggle recording
        if keycode[1] == 'r':
            if self.recording:
                self._process_input()
            self.recording = not self.recording

        # play back live buffer
        if keycode[1] == 'p':
            if self.live_wave:
                self.mixer.add(WaveGenerator(self.live_wave))

        if keycode[1] == 'c' and NUM_CHANNELS == 2:
            self.channel_select = 1 - self.channel_select

        # adjust mixer gain
        gf = lookup(keycode[1], ('up', 'down'), (1.1, 1/1.1))
        if gf:
            new_gain = self.mixer.get_gain() * gf
            self.mixer.set_gain( new_gain )

    def _process_input(self) :
        data = combine_buffers(self.input_buffers)
        print('live buffer size:', len(data) / NUM_CHANNELS, 'frames')
        write_wave_file(data, NUM_CHANNELS, 'recording.wav')
        self.live_wave = WaveArray(data, NUM_CHANNELS)
        self.input_buffers = []

# pass in which MainWidget to run as a command-line arg
run(MainWidget1)
