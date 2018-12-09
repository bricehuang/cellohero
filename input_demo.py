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
from kivy.graphics.vertex_instructions import RoundedRectangle
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
#from kivy.core.image import Image
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from random import randint
import aubio
import math
import numpy as np

NUM_CHANNELS = 1

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

STAFF_LEFT_X = 100
BIG_BLACK_BOX_X = 250
NOW_BAR_X = 500
PADDING = 100
NOW_BAR_VERT_OVERHANG = 100
NOTE_RECT_MARGIN = 2

MIDDLE_C_ID = 28
E2_ID = 16
LANE_HEIGHT = 50
LANE_SEP = LANE_HEIGHT/2
STAFF_Y_VALS = (500,550,600,650,700)
MID_C_LOWER_Y = STAFF_Y_VALS[-1]+LANE_SEP

NOTE_RADIUS = LANE_SEP - NOTE_RECT_MARGIN

NOTE_SCORE_RATE = 1000
PERCENT_NOTE_TO_HIT = .8
ACCEPTABLE_PITCH_INTERVAL = 0.2

def note_to_lower_left(noteid):
    return MID_C_LOWER_Y + (noteid - MIDDLE_C_ID)*LANE_SEP

class FeedbackArrow(InstructionGroup):
    def __init__(self,):
        super(FeedbackArrow, self).__init__()

        # variables for sensitivity of the arrow
        self.number_last_pitches_to_consider = 10
        self.outlier_pitch_amt = 2
        self.last_heard_pitches = []

        # arrow
        self.horiz_position = NOW_BAR_X - 100
        self.vert_position = Window.height/2 # can start in any position
        self.size = (100, 50)
        self.arrow = CRectangle(texture = Image(source = 'images/white_arrow.png').texture, cpos=(self.horiz_position, self.vert_position), csize=self.size)

        self.color = Color(1,0,0,0)


        # accidental
        accidental_size = (self.size[1], self.size[1])
        self.accidental_horiz_position = self.horiz_position - .5 * self.size[1] - 100
        self.accidental = CRectangle(cpos = (self.accidental_horiz_position, self.vert_position), csize = accidental_size)

        self.accidental_color = Color(1,0,0,1)

        # rotations
        self.angle = 0
        rotation_origin = (self.arrow.get_cpos()[0] + self.size[0], self.arrow.get_cpos()[1] + self.size[1] * .5, ) #furtherest right
        self.rotation = Rotate(axis = (0,0,1), angle= self.angle, origin = rotation_origin)
        self.anti_rotation = Rotate(axis = (0,0,1), angle= - self.angle, origin = rotation_origin)

        # adding to canvas
        self.add(self.rotation)
        self.add(self.color)
        self.add(self.arrow)
        self.add(self.accidental_color)
        self.add(self.accidental)
        self.add(self.anti_rotation)

    def set_visible(self):
        self.color.a = 1

    def set_invisible(self):
        self.color.a = 0
        self.accidental_color.a = 0

    def set_accidental_visible(self, isSharp): # isSharp is boolean indicating either sharp or natural
        self.accidental_color.a = 1
        if isSharp:
            self.accidental.texture = Image(source = 'images/white_sharp.png').texture
        else:
            self.accidental.texture = Image(source = 'images/white_natural.png').texture

    def set_red(self):
        self.color.rgb = (1,0,0)
        self.accidental_color.rgb = (1,0,0)

    def set_green(self):
        self.color.rgb = (0,1,0)
        self.accidental_color.rgb = (0,1,0)

    def set_white(self):
        self.color.rgb = (1,1,1)
        self.accidental_color.rgb = (1,1,1)

    def set_accidental_invisible(self):
        self.accidental_color.a = 0

    def set_vertical_position(self, y_pos):
        self.vert_position = y_pos
        self.arrow.cpos = (self.horiz_position, self.vert_position)
        self.accidental.cpos = (self.accidental_horiz_position, self.vert_position)

    def set_rotation(self, angle = 0):
        self.arrow.angle = angle
        self.rotation.angle = self.arrow.angle
        self.anti_rotation.angle = -self.arrow.angle


    def on_update(self, dt, current_expected_note, pitch_heard):
        # ignore outliers to provent sporadic movement
        if len(self.last_heard_pitches) < self.number_last_pitches_to_consider or abs(pitch_heard - np.mean(self.last_heard_pitches)) < self.outlier_pitch_amt:

            if pitch_heard > 35: # only display cello range notes
                self.set_visible()
                self.set_rotation() # reset orientation to 0

                # get the position of the note
                if current_expected_note and abs(current_expected_note.pitch - pitch_heard) < 1:  # these are cases which could round away to other positions
                    # they are on the same line
                    position = current_expected_note.noteid
                    sharp_needed = current_expected_note.acc == '#'
                else:
                    pitch_heard_closest = int(pitch_heard)
                    # MIDI_TO_NOTE = {0:'C', 2:'D', 4:'E', 5:'F', 7:'G', 9:'A', 11:'B'}
                    if pitch_heard_closest % 12 in MIDI_TO_NOTE:
                        sharp_needed = False
                        midi_note = MIDI_TO_NOTE[pitch_heard_closest % 12]
                        octave = math.floor(pitch_heard_closest/12)
                    else:
                        sharp_needed = True
                        midi_note = MIDI_TO_NOTE[(pitch_heard_closest - 1) % 12]
                        octave = math.floor((pitch_heard_closest - 1)/12)
                    position = NOTE_TO_ID[midi_note] + (octave - 1) * NOTES_IN_OCTAVE

                place_here = note_to_lower_left(position)
                self.set_vertical_position(place_here + LANE_SEP)

                natural_needed = False
                if current_expected_note:
                    if abs(current_expected_note.pitch - pitch_heard) < 1: # within half step of correct note
                        angle = np.interp(current_expected_note.pitch - pitch_heard, (-1, 1), (-90,90))
                        self.set_rotation(angle)
                        if abs(current_expected_note.pitch - pitch_heard) <= ACCEPTABLE_PITCH_INTERVAL:
                            self.set_green()
                        else:
                            self.set_red()
                    else:
                        # heard pitch and expected pitch displayed on same line
                        # will already show the sharp, but need to show natural in this case
                        self.set_red()
                        if position == current_expected_note.noteid and not sharp_needed:
                            natural_needed = True
                else:
                    self.set_white()

                # turn on/off sharp sign if necessary (important, must happen after color changes):
                if(sharp_needed):
                    self.set_accidental_visible(True)
                elif(natural_needed):
                    self.set_accidental_visible(False)
                else:
                    self.set_accidental_invisible()

            else:
                self.set_invisible()
                self.set_accidental_invisible()

        self.last_heard_pitches.append(pitch_heard)
        self.last_heard_pitches = self.last_heard_pitches[-self.number_last_pitches_to_consider:]


LEDGER_LINE_WIDTH = 55
class LedgerLine(InstructionGroup):
    def __init__(self, note, left_px):
        super(LedgerLine, self).__init__()
        assert (note%2 == 0 and (note >= MIDDLE_C_ID or note <= E2_ID))
        self.color = Color(.8,.8,.2)
        self.add(self.color)
        self.pos = np.array((left_px, note_to_lower_left(note) + LANE_SEP))
        self.line = Line(points=[self.pos[0],self.pos[1],self.pos[0]+LEDGER_LINE_WIDTH,self.pos[1]],width=4)
        self.add(self.line)

    def on_update(self,dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.line.points = [self.pos[0],self.pos[1],self.pos[0]+LEDGER_LINE_WIDTH,self.pos[1]]

    @staticmethod
    def get_ledger_lines(note, left_px):
        if (note < MIDDLE_C_ID and note > E2_ID):
            return []
        elif (note >= MIDDLE_C_ID):
            return [LedgerLine(note2, left_px) for note2 in range(MIDDLE_C_ID, note+1, 2)]
        else:
            return [LedgerLine(note2, left_px) for note2 in range(E2_ID, note-1, -2)]

STEM_LENGTH=75
class NoteFigure(InstructionGroup):

    @staticmethod
    def get_image(dur, note):
        suffix = '' if dur == 4 else ('-d' if note>22 else '-u')
        img = Image(source = 'notes/' + str(dur) + suffix + '.png')
        print(img.source)
        return img.texture

    @staticmethod
    def note_offset(dur, note):
        down = (note > 22)
        if dur == 4:
            return np.array((-2,-4))
        elif dur == 3 and down:
            return np.array((0,-90))
        elif dur == 3 and not down:
            return np.array((0,3))
        elif dur == 2 and down:
            return np.array((0,-90))
        elif dur == 2 and not down:
            return np.array((0,3))
        elif dur == 1.5 and down:
            return np.array((0,-95))
        elif dur == 1.5 and not down:
            return np.array((-5,2))
        elif dur == 1 and down:
            return np.array((-22,-100))
        elif dur == 1 and not down:
            return np.array((-18,-12))
        elif dur == 0.5 and down:
            return np.array((0,-90))
        elif dur == 0.5 and not down:
            return np.array((0,0))
        return np.array((0,0))

    @staticmethod
    def note_size(dur, note):
        if dur == 4:
            return np.array((55,55))
        elif dur == 3:
            return np.array((70,135))
        elif dur == 2:
            return np.array((50,135))
        elif dur == 1.5:
            return np.array((65,140))
        elif dur == 1:
            return np.array((90,160))
        elif dur == 0.5 and note>22:
            return np.array((40,130))
        elif dur == 0.5 and note <= 22:
            return np.array((63,130))
        return np.array((0,0))

    def __init__(self, note, left_px, duration_beats, acc):
        super(NoteFigure, self).__init__()
        self.dur = round(duration_beats, 1)
        self.note = note
        self.add(Color(.8,.8,.2))
        self.pos = np.array((left_px + 3, note_to_lower_left(note) + 3))

        # if duration_beats == 4:
        texture = NoteFigure.get_image(self.dur, self.note)
        pos = self.pos + NoteFigure.note_offset(self.dur, self.note)
        size = NoteFigure.note_size(self.dur, self.note)

        self.body = Rectangle(
            texture = texture,
            pos = pos,
            size = size
        )
        self.add(self.body)

        self.acc = None
        if acc == '#':
            self.acc = Rectangle(
                texture=Image(source='images/white_sharp.png').texture,
                pos = self.pos + np.array((-40,5)),
                size = np.array((40,40))
            )
        elif acc == 'b':
            self.acc = Rectangle(
                texture=Image(source='images/white_flat.png').texture,
                pos = self.pos + np.array((-40,5)),
                size = np.array((40,40))
            )
        if self.acc:
            self.add(self.acc)

    def on_update(self, dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.body.pos = self.pos + NoteFigure.note_offset(self.dur, self.note)
        if self.acc:
            self.acc.pos = self.pos + np.array((-40,5))

# display for a single note at a position
class NoteDisplay(InstructionGroup):
    LIVE = 'live'
    HIT = 'hit'
    MISSED = 'missed'
    def __init__(self, parsed_pitch, start_time_and_beats, duration_and_beats):
        super(NoteDisplay, self).__init__()
        pitch, noteid, acc = parsed_pitch
        self.pitch = pitch
        self.noteid = noteid
        self.acc = acc

        start_time, start_beats = start_time_and_beats
        self.start_time = start_time
        self.start_beats = start_beats

        duration_time, duration_beats = duration_and_beats
        self.duration_time = duration_time
        self.duration_beats = duration_beats

        self.status = NoteDisplay.LIVE

        vert_position = note_to_lower_left(noteid)
        horiz_position = NOW_BAR_X + start_time * NOTE_SPEED

        self.pos = np.array([horiz_position, vert_position+NOTE_RECT_MARGIN])

        self.color = Color(1,1,1)
        self.add(self.color)

        self.rect = RoundedRectangle(radius=[NOTE_RADIUS]*4, pos=self.pos, size=(duration_time*NOTE_SPEED, LANE_HEIGHT-2*NOTE_RECT_MARGIN))
        self.add(self.rect)

        # ledger lines
        self.ledger_lines = LedgerLine.get_ledger_lines(self.noteid, self.pos[0])
        for ll in self.ledger_lines:
            self.add(ll)

        # note head
        self.figure = NoteFigure(self.noteid, self.pos[0], self.duration_beats, self.acc)
        self.add(self.figure)
        # self.add(Line(points=[700,750,800,750],width=2))

        self.duration_hit = 0
        self.duration_passed = 0

        #arrow_buffer = 50

        # self.up_arrow_position = np.array([
        #     self.rect.pos[0] + .5 * self.rect.size[0],
        #     self.rect.pos[1] + .5 * self.rect.size[1] - arrow_buffer])
        # self.down_arrow_position = np.array([
        #     self.rect.pos[0] + .5 * self.rect.size[0],
        #     self.rect.pos[1] + .5 * self.rect.size[1] + arrow_buffer])

        # print ("up: ", up_arrow_position)
        # self.intonationManager = IntonationManager(up_arrow_position, down_arrow_position)
        # self.add(self.intonationManager)


    def on_hit(self, dt):
        self.status = NoteDisplay.HIT
        #self.color.rgb = (0,1,0)
        self.duration_passed += dt
        self.duration_hit += dt

    def on_pass(self, dt):
        self.status = NoteDisplay.MISSED
        self.color.rgb = (1,0,0)
        self.duration_passed += dt

    def on_update(self, dt):
        for ll in self.ledger_lines:
            ll.on_update(dt)
        self.figure.on_update(dt)

        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.rect.pos = self.pos
        if self.status == NoteDisplay.HIT:
            self.color.rgb = (0, min(1, self.duration_hit/self.duration_time * .8 + .4), 0)

    def get_x_bounds(self):
        return (self.pos[0], self.pos[0] + self.duration_time*NOTE_SPEED)

    def enough_note_hit(self, out_of_total):
    # percent of note is float, out_of_total is boolean true if out of total duration, false if so far in duration played
        if out_of_total:
            return self.duration_hit/self.duration_time >= PERCENT_NOTE_TO_HIT
        else:
            if self.duration_passed == 0:
                return False
            return self.duration_hit/self.duration_passed >= PERCENT_NOTE_TO_HIT

    def get_center_position(self):
        return np.array([self.pos[0] + .5 * self.rect.size[0], self.pos[1] - NOTE_RECT_MARGIN])

    def get_up_arrow_pos(self):
        return self.up_arrow_position
    def get_down_arrow_pos(self):
        return self.down_arrow_position

class BarLine(InstructionGroup):
    def __init__(self, time):
        super(BarLine, self).__init__()
        self.color = Color(1,1,1)
        self.add(self.color)

        self.x = NOW_BAR_X + time*NOTE_SPEED
        self.line = Line(
            points=[
                self.x,
                STAFF_Y_VALS[0],
                self.x,
                STAFF_Y_VALS[-1]
            ],
            width=3
        )
        self.add(self.line)

    def on_update(self, dt):
        self.x -= NOTE_SPEED*dt
        self.line.points = [self.x, STAFF_Y_VALS[0], self.x, STAFF_Y_VALS[-1]]

class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data):
        # note_info is List[(parsed_pitch, start_time, duration)]
        super(BeatMatchDisplay, self).__init__()
        self.max_score = song_data.max_score
        note_info = song_data.notes
        bar_info = song_data.barlines

        # draw staff lines
        self.add(Color(1,1,1))
        for y in STAFF_Y_VALS:
            self.add(Line(points=[BIG_BLACK_BOX_X,y,Window.width,y], width=2))

        # self.add(Line(points=[BIG_BLACK_BOX_X,STAFF_Y_VALS[0],BIG_BLACK_BOX_X,STAFF_Y_VALS[-1]], width=2))
        self.notes = []
        for parsed_pitch, start_time, duration in note_info:
            note = NoteDisplay(parsed_pitch, start_time, duration)
            self.add(note)
            self.notes.append(note)

        self.bars = []
        for bar_time in bar_info:
            bar = BarLine(bar_time)
            self.add(bar)
            self.bars.append(bar)

        # this makes note content disappear once it passes the now bar
        self.add(Color(0,0,0))
        self.add(Rectangle(pos=(0,0),size=(BIG_BLACK_BOX_X,Window.height-200),texture = Image(source = "images/gradient.png").texture))

        # draw part of staff left of big black box
        self.add(Color(1,1,1))
        for y in STAFF_Y_VALS:
            self.add(Line(points=[STAFF_LEFT_X,y,BIG_BLACK_BOX_X,y], width=2))
        self.add(Line(
            points=[
                STAFF_LEFT_X,
                STAFF_Y_VALS[0],
                STAFF_LEFT_X,
                STAFF_Y_VALS[-1],
            ],
            width=2)
        )

        # draw now bar
        self.add(Line(
            points=[
                NOW_BAR_X,
                STAFF_Y_VALS[0]-NOW_BAR_VERT_OVERHANG,
                NOW_BAR_X,
                STAFF_Y_VALS[-1]+NOW_BAR_VERT_OVERHANG
            ],
            width=2)
        )

        # draw bass clef
        self.bassClef = Rectangle(texture = Image(source = "white_bass_clef.png").texture, pos=(STAFF_LEFT_X, STAFF_Y_VALS[1]), size=(abs(BIG_BLACK_BOX_X - STAFF_LEFT_X), abs(STAFF_Y_VALS[-1] - STAFF_Y_VALS[1])))
        self.add(self.bassClef)

        # add intonation adjustion arrows
        # ARROW_BUFFER = 200
        # up_arrow_positions = []
        # down_arrow_positions = []
        # for note in self.notes:
        #     center_position = note.get_center_position()
        #     up_arrow_positions.append(center_position - np.array([0, ARROW_BUFFER]))
        #     down_arrow_positions.append(center_position + np.array([0, ARROW_BUFFER]))
        # self.intonationDisplay = IntonationDisplay(up_arrow_positions, down_arrow_positions)
        # self.add(self.intonationDisplay)
        self.arrow_feedback = FeedbackArrow()
        self.add(self.arrow_feedback)

    # called by Player. Causes the right thing to happen
    def note_hit(self, gem_idx, time_passed):
        self.notes[gem_idx].on_hit(time_passed)

    # called by Player. Causes the right thing to happen
    def note_pass(self, gem_idx, time_passed):
        self.notes[gem_idx].on_pass(time_passed)

    # call every frame to make gems and barlines flow down the screen
    def on_update(self, dt) :
        # update all components that animate
        for note in self.notes:
            note.on_update(dt)

        #self.intonationDisplay.on_update(dt)

        for bar in self.bars:
            bar.on_update(dt)

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

    def enough_note_hit(self, gem_idx, out_of_total):
        return self.notes[gem_idx].enough_note_hit(out_of_total)

GOOD_CUTOFF = 0.6
EXCELLENT_CUTOFF = 0.9
class Cellist(object):
    def __init__(self, display, score_cb, end_game_cb):
        super(Cellist, self).__init__()
        self.display = display
        self.cur_pitch = 0
        self.score = 0
        self.score_cb = score_cb
        self.end_game_cb = end_game_cb

    def notify_pitch(self, pitch):
        self.cur_pitch = pitch

    def on_update(self):
        time_passed = kivyClock.frametime
        current_note = self.display.current_note()
        if current_note is not None:
            idx, pitch = current_note
            pitch_diff = np.abs(pitch - self.cur_pitch)
            if pitch_diff < ACCEPTABLE_PITCH_INTERVAL:
                if not self.display.enough_note_hit(idx, True):
                    self.score += NOTE_SCORE_RATE * time_passed
                    self.score_cb(self.score)
                    self.display.note_hit(idx, time_passed)
        missed_notes = self.display.missed_notes()
        for idx in missed_notes:
            self.display.note_pass(idx, time_passed)

        if self.display.bars[-1].x < NOW_BAR_X:
            # we're done
            max_score = self.display.max_score
            rating = (
                1 +
                (1 if self.score >= GOOD_CUTOFF*max_score else 0) +
                (1 if self.score >= EXCELLENT_CUTOFF*max_score else 0)
            )
            self.end_game_cb(self.score, rating)


        current_note = self.display.current_note()
        if current_note:
            current_note = self.display.notes[current_note[0]]

        self.display.arrow_feedback.on_update(time_passed, current_note, self.cur_pitch)

PITCHES_IN_OCTAVE = 12
MIDI_ADJ = 12
NOTES_IN_OCTAVE = 7
NOTE_TO_MIDI = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}
MIDI_TO_NOTE = {0:'C', 2:'D', 4:'E', 5:'F', 7:'G', 9:'A', 11:'B'}
ACC_TO_MIDI_ADJUST = {'#':1, '':0, 'b':-1}
NOTE_TO_ID = {'C':0, 'D':1, 'E':2, 'F':3, 'G':4, 'A':5, 'B':6}
def parse_pitch(str_pitch):
    if str_pitch[1] in {'#','b'}:
        note = str_pitch[0]
        acc = str_pitch[1]
        octave = int(str_pitch[2:])
    else:
        note = str_pitch[0]
        acc = ''
        octave = int(str_pitch[1:])
    midi = NOTE_TO_MIDI[note] + ACC_TO_MIDI_ADJUST[acc] + octave*PITCHES_IN_OCTAVE + MIDI_ADJ
    note = NOTE_TO_ID[note] + octave * NOTES_IN_OCTAVE
    return (midi, note, acc)

SEC_PER_MIN = 60.
class SongData(object):
    def __init__(self, filepath):
        self.notes = []
        f = open(filepath, 'r')
        header = f.readline()
        bpm = float(header.rstrip())
        bps = bpm / SEC_PER_MIN

        beats_per_measure = f.readline()
        beats_per_measure = int(beats_per_measure.rstrip())

        body_line = f.readline()
        while body_line:
            line = body_line.rstrip()
            str_pitch, start, duration = line.split()
            self.notes.append((
                parse_pitch(str_pitch),
                (float(start)/bps, float(start)),
                (float(duration)/bps, float(duration))
            ))
            body_line = f.readline()
        f.close()
        last_note = self.notes[-1]
        last_beat = last_note[1][1] + last_note[2][1]
        last_barline_beat = np.ceil(last_beat / beats_per_measure) * beats_per_measure
        self.barlines = [float(beats_per_measure + bar)/bps for bar in range(0, int(last_barline_beat), beats_per_measure)]

        self.max_score = NOTE_SCORE_RATE * PERCENT_NOTE_TO_HIT * sum(duration[0] for _, _, duration in self.notes)

START_MENU = "start"
IN_GAME = "game"
END_GAME = "end"
class MainWidget1(BaseWidget):
    BEAR_SIZE = 500
    def __init__(self):
        super(MainWidget1, self).__init__()

        self.audio = Audio(NUM_CHANNELS, input_func=self.receive_audio)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.io_buffer = IOBuffer()
        self.mixer.add(self.io_buffer)
        self.pitch = PitchDetector()
        self.cur_pitch = 0

        self.recording = False
        self.channel_select = 0
        self.input_buffers = []
        self.live_wave = None

        self.state = START_MENU
        self.info = topleft_label()
        self.info.text = ""
        self.add_widget(self.info)

        self.objects = AnimGroup()
        self.canvas.add(self.objects)

        self.score = 0
        self.song_data = None
        self.display = None
        self.cellist = None

        # CELLO HERO
        self.logo = Image(
            source = "images/cellohero.png",
            size = (Window.width/2, Window.height/4),
            pos = (Window.width/4, Window.height*2/3)
        )
        self.add_widget(self.logo)

        # BEAR ANIMATION!
        self.bear = Image(
            source = "images/going_to_practice_bear.gif",
            size = (self.BEAR_SIZE, self.BEAR_SIZE),
            pos = (Window.width/2 - self.BEAR_SIZE/2, Window.height/4)
        )
        self.add_widget(self.bear)

        self.reset_button = None
        self.replay_button = None
        self.song_name = None
        self.score_label = None
        self.reset()

    def create_button(self, text, id, pos):
        button = Button(text=text, id=id, pos=pos, size=(500,300), font_size=40)
        button.bind(state= self.select_song_callback)
        self.add_widget(button)
        self.buttons.append(button)

    # button callback
    def select_song_callback(self, instance, value):
        if value == 'down':
            self.start_song(instance.id)

    def start_song(self, filename):
        # start screen -> game
        self.state = IN_GAME
        self.song_name = filename

        for button in self.buttons:
            self.remove_widget(button)
        self.buttons = []
        self.clear_end_screen_buttons()

        self.bear.pos = (Window.width/2 - self.BEAR_SIZE/2, 0)
        self.bear.source = "images/good_job_bear.gif"

        self.song_data = SongData('music/'+filename+'.txt')
        self.display = BeatMatchDisplay(self.song_data)
        self.cellist = Cellist(self.display, self.update_score, self.end_song)
        self.objects.add(self.display)

    def clear_end_screen_buttons(self):
        if self.reset_button:
            self.remove_widget(self.reset_button)
            self.reset_button = None
        if self.replay_button:
            self.remove_widget(self.replay_button)
            self.replay_button = None
        if self.score_label:
            self.remove_widget(self.score_label)
            self.score_label = None

    # called by game logic at game end
    # rating is 1,2,3 (bad, good, excellent)
    def end_song(self, score, rating):
        # game -> end screen
        self.state = END_GAME

        self.score_label = Label(text= "Score: %d\n" % score, pos = (Window.width/2, Window.height/2), font_size = "40sp")
        self.add_widget(self.score_label)

        if self.info:
            self.remove_widget(self.info)
            self.info = None

        if rating == 1:
            self.bear.source = "images/cello_smash.gif"
        elif rating == 2:
            self.bear.source = "images/okay_bear.gif"
        elif rating == 3:
            self.bear.source = "images/clapping_bear.gif"

        self.objects.remove(self.display)
        self.song_data = None
        self.display = None
        self.cellist = None

        self.reset_button = Button(text='Main Menu', pos=(0,100), size=(500,200), font_size=40)
        self.reset_button.bind(state= self.reset_callback)
        self.add_widget(self.reset_button)

        self.replay_button = Button(text='Play Again', id=self.song_name, pos=(1100,100), size=(500,200), font_size=40)
        self.replay_button.bind(state=self.select_song_callback)
        self.add_widget(self.replay_button)

    def reset_callback(self, instance, value):
        if value == 'down':
            self.reset()

    # reset to start screen
    def reset(self):
        # reset bear
        self.bear.source = "images/going_to_practice_bear.gif"
        self.bear.pos = (Window.width/2 - self.BEAR_SIZE/2, Window.height/4)

        # reset whatever state
        self.song_name = None

        # set up buttons
        self.clear_end_screen_buttons()

        # set up text in the corner
        if not self.info:
            self.info = topleft_label()
            self.info.text = ""
            self.add_widget(self.info)

        self.buttons = []
        self.create_button('C Major Scale', 'cmaj', (0,0))
        self.create_button('Mary Had a Little Lamb', 'mary', (500,0))
        self.create_button('Rigadoon', 'rigadoon', (1000,0))

    def update_score(self, score):
        self.score = score

    def on_update(self) :
        if self.state == IN_GAME:
            self.audio.on_update()
            self.cellist.on_update()
            self.objects.on_update()

            if self.info:
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

        # changing bear image
        if keycode[1] == '1':
            self.bear.source = 'images/cello_smash.gif'
        if keycode[1] == '2':
            self.bear.source = 'images/clapping_bear.gif'
        if keycode[1] == '3':
            self.bear.source = 'images/good_job_bear.gif'

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