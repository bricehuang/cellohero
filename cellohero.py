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
from kivy.core.text import Label as CoreLabel

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

def set_global_lengths():
    # parts sizing
    global NOW_BAR_VERT_OVERHANG, LANE_HEIGHT, LANE_SEP
    NOW_BAR_VERT_OVERHANG = Window.height / 10
    LANE_HEIGHT = Window.height / 25
    LANE_SEP = LANE_HEIGHT / 2

    # note speed
    global NOTE_SPEED
    NOTE_SPEED = Window.width / 12

    # geography
    global STAFF_LEFT_X, NOW_BAR_X, STAFF_Y_VALS, MIDDLE_C_Y
    STAFF_LEFT_X = Window.width / 18
    NOW_BAR_X = STAFF_LEFT_X + NOTE_SPEED * 4 # TODO: not staff left x
    STAFF_Y_VALS = tuple(Window.height/2 + i * LANE_HEIGHT for i in range(-2,3))
    MIDDLE_C_Y = STAFF_Y_VALS[-1] + LANE_HEIGHT

    # note sizing
    global NOTE_RECT_MARGIN, NOTE_RADIUS, PROGRESS_BAR_RADIUS, NOTE_RADIUS_DOWN, PROGRESS_BAR_RADIUS_DOWN
    NOTE_RECT_MARGIN = LANE_HEIGHT / 20
    NOTE_RADIUS = LANE_SEP - NOTE_RECT_MARGIN
    PROGRESS_BAR_RADIUS = LANE_HEIGHT / 20
    NOTE_RADIUS_DOWN = np.array((0,-NOTE_RADIUS))
    PROGRESS_BAR_RADIUS_DOWN = np.array((0,-PROGRESS_BAR_RADIUS))

    # ledger lines
    global LEDGER_LINE_XOFFSET, LEDGER_LINE_LENGTH, LEDGER_LINE_WIDTH
    LEDGER_LINE_XOFFSET = NOTE_RADIUS / 4
    LEDGER_LINE_LENGTH = NOTE_RADIUS + 2 * LEDGER_LINE_XOFFSET
    LEDGER_LINE_WIDTH = PROGRESS_BAR_RADIUS * 2

    # barlines
    global BARLINE_WIDTH
    BARLINE_WIDTH = 3

set_global_lengths()

MIDDLE_C_ID = 28
E2_ID = 16
NOTE_SCORE_RATE = 1000
PERCENT_NOTE_TO_HIT = .8
ACCEPTABLE_PITCH_INTERVAL = 0.2


def note_y_coordinate(noteid):
    return MIDDLE_C_Y + (noteid - MIDDLE_C_ID)*LANE_SEP

class FeedbackArrow(InstructionGroup):
    def __init__(self,):
        super(FeedbackArrow, self).__init__()

        # variables for sensitivity of the arrow
        self.number_last_pitches_to_consider = 10
        self.outlier_pitch_amt = 2
        self.last_heard_pitches = []

        # arrow
        self.horiz_position = NOW_BAR_X - 200
        self.vert_position = Window.height/2 # can start in any position
        self.size = (100, 50)
        self.arrow = CRectangle(texture = Image(source = 'images/white_arrow.png').texture, cpos=(self.horiz_position, self.vert_position), csize=self.size)

        self.color = Color(1,1,1,0)

        # rotations
        self.angle = 0
        self.rotation_origin = self.update_rotation_origin(False)
        self.rotation = Rotate(axis = (0,0,1), angle= self.angle, origin = self.rotation_origin)
        self.anti_rotation = Rotate(axis = (0,0,1), angle= - self.angle, origin = self.rotation_origin)

        # adding to canvas
        self.add(self.rotation)
        self.add(self.color)
        self.add(self.arrow)
        self.add(self.anti_rotation)

        self.offset = self.arrow.csize[1]/2

    def set_visible(self):
        self.color.a = 1

    def set_invisible(self):
        self.color.a = 0

    def set_red(self):
        self.color.rgb = (1,0,0)

    def set_green(self):
        self.color.rgb = (0,1,0)

    def set_black(self):
        self.color.rgb = (1,1,1)

    def set_white(self):
        self.color.rgb = (1,1,1)

    def update_rotation_origin(self, update_rotation_objects = True):
        self.rotation_origin = (self.arrow.get_cpos()[0] + .5 * self.arrow.csize[0], self.arrow.get_cpos()[1]) #furtherest right
        if(update_rotation_objects):
            self.rotation.origin = self.rotation_origin
            self.anti_rotation.origin = self.rotation_origin
        return self.rotation_origin

    def set_vertical_position(self, y_pos):
        self.vert_position = y_pos + self.offset
        self.arrow.cpos = (self.horiz_position, self.vert_position)
        self.update_rotation_origin()


    def set_rotation(self, angle = 0):
        self.arrow.angle = angle
        self.rotation.angle = self.arrow.angle
        self.anti_rotation.angle = -self.arrow.angle


    def on_update(self, dt, current_expected_note, pitch_heard):
        # ignore outliers to provent sporadic movement
        if not current_expected_note:
            self.set_invisible()
            #self.last_heard_pitches = []

        # consider whether or not this is an outlier pitch
        elif len(self.last_heard_pitches) < self.number_last_pitches_to_consider or abs(pitch_heard - np.mean(self.last_heard_pitches)) < self.outlier_pitch_amt:

            if pitch_heard > 35 and current_expected_note: # only display cello range notes

                #  set position
                position = current_expected_note.noteid
                place_here = note_y_coordinate(position) - LANE_SEP
                self.set_vertical_position(place_here)
                self.set_rotation(90)
                pitch_diff = current_expected_note.pitch - pitch_heard
                # set angle (more dramatic angle change when closer)
                if abs(pitch_diff) < 1: # within half step of correct note
                    angle = np.interp(pitch_diff, (-1, 1), (-45,45))
                    self.set_rotation(angle)

                    if abs(current_expected_note.pitch - pitch_heard) <= ACCEPTABLE_PITCH_INTERVAL:
                        self.set_green()
                    else:
                        self.set_red()
                else:
                    # heard pitch and expected pitch displayed on same line
                    # will already show the sharp, but need to show natural in this case
                    self.set_red()
                    max_half_steps_away = 8
                    if pitch_diff < 0: # negative
                        angle = np.interp(pitch_diff, (-max_half_steps_away, 0), (-90,-45))
                        if(pitch_diff < -max_half_steps_away):
                            angle = -90
                    else:
                        angle = np.interp(pitch_diff, (0, max_half_steps_away), (45,90))
                        if(pitch_diff > max_half_steps_away):
                            angle = 90
                    self.set_rotation(angle)
                self.set_visible()
            else:
                self.set_invisible()

        self.last_heard_pitches.append(pitch_heard)
        self.last_heard_pitches = self.last_heard_pitches[-self.number_last_pitches_to_consider:]

class LedgerLine(InstructionGroup):
    def __init__(self, note, left_px):
        super(LedgerLine, self).__init__()
        assert (note%2 == 0 and (note >= MIDDLE_C_ID or note <= E2_ID))
        self.color = Color(0,0,0)
        self.add(self.color)
        self.pos = np.array((left_px, note_y_coordinate(note)))
        self.line = Line(points=self.points(), width=LEDGER_LINE_WIDTH)
        self.add(self.line)

    def points(self):
        return [
            self.pos[0] - LEDGER_LINE_XOFFSET,
            self.pos[1],
            self.pos[0] - LEDGER_LINE_XOFFSET + LEDGER_LINE_LENGTH,
            self.pos[1]
        ]

    def on_update(self,dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.line.points = self.points()

    @staticmethod
    def get_ledger_lines(note, left_px):
        if (note < MIDDLE_C_ID and note > E2_ID):
            return []
        elif (note >= MIDDLE_C_ID):
            return [LedgerLine(note2, left_px) for note2 in range(MIDDLE_C_ID, note+1, 2)]
        else:
            return [LedgerLine(note2, left_px) for note2 in range(E2_ID, note-1, -2)]

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
        orientation_multiplier = -1 if down else 1
        if dur == 4:
            return np.array((0,0))
        elif dur == 3:
            return (
                np.array((0.450, 1.980)) * orientation_multiplier * NOTE_RADIUS
                + np.array((0.7*NOTE_RADIUS if down else 0, 1))
            )
        elif dur == 2:
            return np.array((0, 1.980)) * orientation_multiplier * NOTE_RADIUS
        elif dur == 1.5:
            return (
                np.array((0.373, 2.094)) * orientation_multiplier * NOTE_RADIUS
                + np.array((0.7*NOTE_RADIUS if down else 0, 1))
            )
        elif dur == 1:
            return np.array((0, 1.855)) * orientation_multiplier * NOTE_RADIUS
        elif dur == 0.5 and down:
            return np.array((0,-1.663)) * NOTE_RADIUS
        elif dur == 0.5 and not down:
            return np.array((0.411,1.633)) * NOTE_RADIUS
        assert False, 'should not get here'

    @staticmethod
    def note_size(dur, note):
        if dur == 4:
            return np.array((2, 2)) * NOTE_RADIUS
        elif dur == 3:
            return np.array((2.952, 5.544)) * NOTE_RADIUS
        elif dur == 2:
            return np.array((2, 5.566)) * NOTE_RADIUS
        elif dur == 1.5:
            return np.array((2.745, 5.500)) * NOTE_RADIUS
        elif dur == 1:
            return np.array((2, 5.490)) * NOTE_RADIUS
        elif dur == 0.5 and note>22:
            return np.array((2,4.853)) * NOTE_RADIUS
        elif dur == 0.5 and note <= 22:
            return np.array((2.822, 4.766)) * NOTE_RADIUS
        assert False, 'should not get here'

    def __init__(self, note, left_px, duration_beats, acc):
        super(NoteFigure, self).__init__()
        self.dur = round(duration_beats, 1)
        self.note = note
        self.add(Color(0,0,0))
        self.pos = np.array((left_px+NOTE_RADIUS, note_y_coordinate(note)))

        self.body = CRectangle(
            texture = NoteFigure.get_image(self.dur, self.note),
            cpos = self.body_cpos(),
            size = NoteFigure.note_size(self.dur, note)
        )
        self.add(self.body)

        self.acc = None
        if acc == '#':
            self.acc = CRectangle(
                texture = Image(source='images/white_sharp.png').texture,
                cpos = self.acc_cpos(),
                size = self.acc_size()
            )
        elif acc == 'b':
            self.acc = CRectangle(
                texture = Image(source='images/white_flat.png').texture,
                cpos = self.acc_cpos(),
                size = self.acc_size()
            )
        if self.acc:
            self.add(self.acc)

    def body_cpos(self):
        return self.pos + NoteFigure.note_offset(self.dur, self.note)

    def acc_cpos(self):
        return self.pos + np.array((-1.5*NOTE_RADIUS,0))

    def acc_size(self):
        return np.array((NOTE_RADIUS,NOTE_RADIUS))

    def on_update(self, dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.body.cpos = self.body_cpos()
        if self.acc:
            self.acc.cpos = self.acc_cpos()

# display for a single note at a position
class NoteDisplay(InstructionGroup):
    LIVE = 'live'
    HIT = 'hit'
    MISSED = 'missed'
    def __init__(self, parsed_pitch, start_time_and_beats, duration_and_beats, score_cb):
        super(NoteDisplay, self).__init__()
        pitch, noteid, acc = parsed_pitch
        self.pitch = pitch
        self.noteid = noteid
        self.acc = acc
        self.score_cb = score_cb

        start_time, start_beats = start_time_and_beats
        self.start_time = start_time
        self.start_beats = start_beats

        duration_time, duration_beats = duration_and_beats
        self.duration_time = duration_time
        self.duration_beats = duration_beats

        self.status = NoteDisplay.LIVE
        self.is_passed = False
        self.score = 0

        vert_position = note_y_coordinate(noteid)
        horiz_position = NOW_BAR_X + start_time * NOTE_SPEED

        self.pos = np.array((horiz_position, vert_position))
        self.width = duration_time * NOTE_SPEED

        self.color = Color(.763,.706,.371)
        self.add(self.color)

        self.rect = RoundedRectangle(
            radius=[NOTE_RADIUS]*4,
            pos = self.pos + NOTE_RADIUS_DOWN,
            size=(self.width, 2*NOTE_RADIUS)
        )
        self.add(self.rect)

        # ledger lines
        self.ledger_lines = LedgerLine.get_ledger_lines(self.noteid, self.pos[0])
        for ll in self.ledger_lines:
            self.add(ll)

        # note figure
        self.figure = NoteFigure(self.noteid, self.pos[0]+NOTE_RECT_MARGIN, self.duration_beats, self.acc)
        self.add(self.figure)

        self.duration_hit = 0
        self.duration_passed = 0
        self.add(Color(0,0,1))
        self.progress_rect = RoundedRectangle(
            radius=[PROGRESS_BAR_RADIUS]*4,
            pos=self.pos + PROGRESS_BAR_RADIUS_DOWN,
            size=(self.width, 2*PROGRESS_BAR_RADIUS)
        )
        self.add(self.progress_rect)

        self.add(Color(0,1,0))
        self.progress_rect_green = RoundedRectangle(
            radius=[PROGRESS_BAR_RADIUS]*4,
            pos=self.pos + PROGRESS_BAR_RADIUS_DOWN,
            size=(self.width * self.score_fraction(), 2*PROGRESS_BAR_RADIUS)
        )
        self.add(self.progress_rect_green)

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

    def score_fraction(self):
        return min(1, self.duration_hit/(self.duration_time * 0.8))

    def on_hit(self, dt):
        self.status = NoteDisplay.HIT
        self.duration_passed += dt
        self.duration_hit += dt

    def on_pass(self, dt):
        self.status = NoteDisplay.MISSED
        self.duration_passed += dt

    def on_update(self, dt, pitch_heard):
        for ll in self.ledger_lines:
            ll.on_update(dt)
        self.figure.on_update(dt)

        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.rect.pos = self.pos + NOTE_RADIUS_DOWN
        self.progress_rect.pos = self.pos + PROGRESS_BAR_RADIUS_DOWN
        self.progress_rect_green.pos = self.pos + PROGRESS_BAR_RADIUS_DOWN
        self.progress_rect_green.size = (self.width * self.score_fraction(), 2*PROGRESS_BAR_RADIUS)

        # update score (only if current note)
        x_bounds = self.get_x_bounds()
        if x_bounds[0] <= NOW_BAR_X and x_bounds[1] >= NOW_BAR_X:
            if abs(self.pitch - pitch_heard) < ACCEPTABLE_PITCH_INTERVAL:
                if(not self.enough_note_hit(True)):
                    self.score += NOTE_SCORE_RATE * dt
        if self.just_passed():
            self.score_cb(self.score)


    def get_x_bounds(self):
        return (self.pos[0], self.pos[0] + self.duration_time*NOTE_SPEED)

    def note_is_over(self):
        return self.get_x_bounds()[1] < NOW_BAR_X

    def just_passed(self):
        # can be called once, once is passed
        if not self.is_passed and self.note_is_over():
            self.is_passed = True
            return True
        return False

    def enough_note_hit(self, out_of_total):
    # percent of note is float, out_of_total is boolean true if out of total duration, false if so far in duration played
        if out_of_total:
            return self.duration_hit/self.duration_time >= PERCENT_NOTE_TO_HIT
        else:
            if self.duration_passed == 0:
                return False
            return self.duration_hit/self.duration_passed >= PERCENT_NOTE_TO_HIT

    # def get_center_position(self):
    #     return np.array([self.pos[0] + .5 * self.rect.size[0], self.pos[1] - NOTE_RECT_MARGIN])

    # def get_up_arrow_pos(self):
    #     return self.up_arrow_position
    # def get_down_arrow_pos(self):
    #     return self.down_arrow_position

class BarLine(InstructionGroup):
    def __init__(self, time):
        super(BarLine, self).__init__()
        self.color = Color(0,0,0)
        self.add(self.color)

        self.x = NOW_BAR_X + time*NOTE_SPEED
        self.line = Line(points = self.points(), width = BARLINE_WIDTH)
        self.add(self.line)

    def points(self):
        return [
            self.x,
            STAFF_Y_VALS[0],
            self.x,
            STAFF_Y_VALS[-1]
        ]

    def on_update(self, dt):
        self.x -= NOTE_SPEED*dt
        self.line.points = self.points()

SCORE_RED = (191/255.0,0,0,1)
SCORE_GREY = (231/255, 230/255, 230/255, 1)
RED_BORDER = "images/red_border.png"

# class ScoreDisplay(InstructionGroup):
#     def __init__(self):
#         super(ScoreDisplay, self).__init__()
#         self.score = score

#         self.color = Color(1,1,1,1)
#         self.add(self.color)
#         self.border = self.score_box = CRectangle(texture = Image(source = RED_BORDER).texture, cpos=(Window.width/2, Window.height/2), csize=(100,100))
#         self.add(self.border)

#     def set_visible(self):
#         self.color = (1,1,1,1)

#     def set_invisible(self):
#         self.color = (1,1,1,0)

#     def set_positions_sizes(self):
#         if self.state == START_MENU:
#             self.border.color

#         if self.state == IN_GAME:
#             self.label.font_size = Window.height/12
#             self.label.pos = (Window.width/15, 7/8*Window.height)

#         if self.state == END_GAME:
#             self.label.font_size = Window.height/12
#             self.label.pos  =  (Window.width/2, Window.height/2)

#     def on_update(self, dt):
#         pass


class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data):
        # note_info is List[(parsed_pitch, start_time, duration)]
        super(BeatMatchDisplay, self).__init__()
        self.max_score = song_data.max_score
        self.cur_pitch = 0
        self.current_score = 0
        note_info = song_data.notes
        bar_info = song_data.barlines

        self.score_cb = None
        self.score = 0

        # draw staff lines
        self.add(Color(0,0,0))
        for y in STAFF_Y_VALS:
            self.add(Line(points=[STAFF_LEFT_X,y,Window.width,y], width=2))
        self.add(Line(
            points=[
                STAFF_LEFT_X,
                STAFF_Y_VALS[0],
                STAFF_LEFT_X,
                STAFF_Y_VALS[-1],
            ],
            width=2)
        )

        self.notes = []
        for parsed_pitch, start_time, duration in note_info:
            note = NoteDisplay(parsed_pitch, start_time, duration, self.update_score)
            self.add(note)
            self.notes.append(note)

        self.bars = []
        for bar_time in bar_info:
            bar = BarLine(bar_time)
            self.add(bar)
            self.bars.append(bar)

        # TODO get the fade

        # draw now bar
        self.add(Color(0,0,0))
        self.add(Line(
            points=[
                NOW_BAR_X,
                STAFF_Y_VALS[0]-NOW_BAR_VERT_OVERHANG,
                NOW_BAR_X,
                STAFF_Y_VALS[-1]+NOW_BAR_VERT_OVERHANG
            ],
            width=2)
        )

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
            note.on_update(dt, self.cur_pitch)

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

    def notify_pitch(self, pitch):
        self.cur_pitch = pitch

    def update_score(self, score_addition):
        self.score += score_addition
        if self.score_cb:
            self.score_cb(self.score, score_addition)


GOOD_CUTOFF = 0.6
EXCELLENT_CUTOFF = 0.9
class Cellist(object):
    def __init__(self, display, score_cb, end_game_cb, update_bear_cb):
        super(Cellist, self).__init__()
        self.display = display
        self.cur_pitch = 0 # current pitch being played by cellist, not necessarily what is supposed to be played.
        self.cur_note_idx = 0
        self.score = 0
        self.score_cb = score_cb
        self.end_game_cb = end_game_cb
        self.update_bear_cb = update_bear_cb

        #set callback for display
        self.display.score_cb = self.update_score

    def notify_pitch(self, pitch):
        self.cur_pitch = pitch
        self.display.notify_pitch(self.cur_pitch)

    def on_update(self):
        NUM_HISTORIC_NOTES_TO_CONSIDER = 5

        time_passed = kivyClock.frametime
        current_note = self.display.current_note()
        if current_note is not None:
            idx, pitch = current_note
            pitch_diff = np.abs(pitch - self.cur_pitch)

            if pitch_diff < ACCEPTABLE_PITCH_INTERVAL:
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


        #current_note = self.display.current_note() # seems redundant?
        if current_note:
            current_note_index = current_note[0]
            current_note = self.display.notes[current_note[0]]

            #  update the bear image
            if current_note_index >= NUM_HISTORIC_NOTES_TO_CONSIDER - 1:
                start = current_note_index - NUM_HISTORIC_NOTES_TO_CONSIDER + 1
                end = current_note_index + 1
                past_notes = self.display.notes[start:end]
                hits = 0
                for historic_note in past_notes:
                    if historic_note.status == NoteDisplay.HIT:
                        hits += 1
                percent_hits = hits/len(past_notes)
                if percent_hits < .6:
                    self.update_bear_cb("lost_bear")
                elif percent_hits >= .6 and percent_hits < .8:
                    self.update_bear_cb("spinning_bear")
                else:
                    self.update_bear_cb("heart_bear")

        self.display.arrow_feedback.on_update(time_passed, current_note, self.cur_pitch)
        self.display.on_update(time_passed)

    def update_score(self, total_score, score_addition):
        self.score = total_score # could change if you want the score over the course of many runs
        self.score_cb(self.score,  score_addition) # change args for cb


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
            start_parsed = float(start) + beats_per_measure
            duration_parsed = float(duration)
            self.notes.append((
                parse_pitch(str_pitch),
                (start_parsed/bps, start_parsed),
                (duration_parsed/bps, duration_parsed)
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
BASS_MASK_DIMS = (412,1000)
BASS_MASK_POS = (0,400)

class MainWidget1(BaseWidget):
    BUTTON_IMAGE = "images/redder_button_image.png"
    FONT_NAME = "fonts/Sniglet/Sniglet-Regular.ttf"
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
        self.playback_wav = None

        self.state = START_MENU

        self.bear_size = 3*Window.height/4
        self.padding = Window.height/20

        self.score = 0

        self.song_data = None
        self.display = None
        self.cellist = None

        # Background
        self.background = Image(
            source = "images/parchment2.png",
            size = (1.4*Window.width, 1.2*Window.height),
            pos = (-0.2*Window.width,-0.1*Window.height),
            allow_stretch = True
        )
        self.add_widget(self.background)

        # load score
        self.info = None #label in the top left corner telling the score
        self.add_top_score_box()

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
            size = (self.bear_size, self.bear_size),
            pos = (Window.width/2 - self.bear_size/2, Window.height/4),
            allow_stretch = True
        )
        self.add_widget(self.bear)

        self.objects = AnimGroup()
        self.canvas.add(self.objects)

        # self.bassclef = Image(
        #     source = "images/bassclef2.png",
        #     size = BASS_MASK_DIMS,
        #     pos = BASS_MASK_POS
        # )

        self.reset_button = None
        self.replay_button = None
        self.buttons = []
        self.song_name = None
        self.score_label = None

        self.resize_elements()
        self.reset()


        Window.bind(on_resize=self.resize_elements)

    def resize_elements(self, instance = None, x = 0, y= 0):
        padding = Window.height/20
        self.padding = padding

        # resize logo
        if (self.logo):
            self.logo.size = ((Window.width - 4 * padding)/2, Window.height/4)
            self.logo.pos = ((Window.width - self.logo.size[0])/2, 3*Window.height/4)

        # bear icon
        if (self.bear):
            self.position_bear(len(self.buttons) > 0) # hack to know on main screen


        # buttons
        navigation_button_size = (Window.width/4, Window.height/3)
        naviation_button_font_size = Window.height/20
        if (self.reset_button):
            self.reset_button.size = navigation_button_size
            self.reset_button.pos = (padding, padding)
            self.reset_button.font_size = naviation_button_font_size
        if (self.replay_button):
            self.replay_button.size = navigation_button_size
            self.replay_button.pos = (Window.width - navigation_button_size[0] - padding, padding)
            self.replay_button.font_size = naviation_button_font_size

        button_num = 0
        for button in self.buttons:
            button.size = (Window.width/4, (Window.height - 2 * padding - .5 * padding * len(self.buttons))/len(self.buttons))
            button.pos = (padding, padding + button_num * (button.size[1]+.5 *padding))
            button.font_size = Window.height/30
            button_num += 1

        # labels
        if (self.score_label):
            self.score_label.font_size = Window.height/12
            self.score_label.pos  =  (Window.width/2, Window.height/2)
        if (self.info):
            self.info.font_size = Window.height/12
            self.info.pos = (Window.width/15, 7/8*Window.height)

        # regenerate global lengths
        if (self.background):
            self.background.size = (1.4*Window.width, 1.2*Window.height)
            self.background.pos = (-0.2*Window.width,-0.1*Window.height)
        set_global_lengths()

    def position_bear(self, is_main_menu):
        if (is_main_menu):
            self.bear_size = 3*Window.height/4
            self.bear.size = (self.bear_size, self.bear_size)
            self.bear.pos = (2 * self.padding + Window.width/4, self.padding)
        else:
            self.bear_size = Window.height/3.5
            self.bear.size = (self.bear_size, self.bear_size)
            self.bear.pos = (Window.width/2 - self.bear_size/2, self.padding)

    def create_button(self, text, id, pos):
        button = Button(text=text, id=id, pos=pos, size=(500,300), font_size=40, background_normal = self.BUTTON_IMAGE, font_name = self.FONT_NAME) # many of these are rewritten by resize function
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

        self.position_bear(False)
        self.bear.source = "images/spinning_bear.gif"


        self.add_top_score_box()

        self.score = 0
        self.song_data = SongData('music/'+filename+'.txt')
        self.display = BeatMatchDisplay(self.song_data)
        self.cellist = Cellist(self.display, self.update_score, self.end_song, self.set_bear)
        self.objects.add(self.display)

        # self.add_widget(self.bassclef)

        # stop playing playback
        self.stop_sound_playback()

        # start recording
        self.recording = True

        self.resize_elements()

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

    def remove_top_score_box(self):
        if self.info:
            self.remove_widget(self.info)
            self.info = None

    def add_top_score_box(self):
        # TODO
        if not self.info:
            #self.info = Label(text= "", font_size = 40, font_name = self.FONT_NAME, valign= 'center', halign= 'center', color= SCORE_GREY, outline_color=SCORE_RED, outline_width = 8, texture = Image(source = self.BUTTON_IMAGE).texture)
            self.info = Label(text= "", font_size = 40, font_name = self.FONT_NAME, valign= 'center', halign= 'center', color= SCORE_GREY, outline_color=SCORE_RED, outline_width = 8)
            self.add_widget(self.info)

    # called by game logic at game end
    # rating is 1,2,3 (bad, good, excellent)
    def end_song(self, score, rating):
        # game -> end screen
        self.state = END_GAME

        self.score_label = Label(text= "Score\n%d" % score, pos = (Window.width/2, Window.height/2), font_size = 40, font_name = self.FONT_NAME, valign= 'center', halign= 'center', color= SCORE_GREY, outline_color=SCORE_RED, outline_width = 8, texture = Image(source = self.BUTTON_IMAGE).texture)
        self.score_label.texture_update()
        self.add_widget(self.score_label)

        self.remove_top_score_box()

        if rating == 1:
            self.bear.source = "images/cello_smash.gif"
        elif rating == 2:
            self.bear.source = "images/okay_bear.gif"
        elif rating == 3:
            self.bear.source = "images/clapping_bear.gif"

        # self.remove_widget(self.bassclef)

        self.objects.remove(self.display)
        self.song_data = None
        self.display = None
        self.cellist = None

        self.reset_button = Button(text='Main Menu', pos=(0,100), size=(500,200), font_size=40, background_normal = self.BUTTON_IMAGE, font_name = self.FONT_NAME)
        self.reset_button.bind(state= self.reset_callback)
        self.add_widget(self.reset_button)

        self.replay_button = Button(text='Play Again', id=self.song_name, pos=(1100,100), size=(500,200), font_size=40, background_normal = self.BUTTON_IMAGE, font_name = self.FONT_NAME)
        self.replay_button.bind(state=self.select_song_callback)
        self.add_widget(self.replay_button)

        self.resize_elements()

        # playback of performance
        self._process_input()
        self.recording = False
        if self.live_wave:
            self.playback_wav = WaveGenerator(self.live_wave)
            self.mixer.add(self.playback_wav)

    def reset_callback(self, instance, value):
        if value == 'down':
            self.reset()

    # reset to start screen
    def reset(self):
        # reset bear
        self.bear.source = "images/going_to_practice_bear.gif"
        self.position_bear(True)

        # reset whatever state
        self.song_name = None

        # set up buttons
        self.clear_end_screen_buttons()

        # set up text in the corner/ clear score in the corner
        self.remove_top_score_box()

        self.buttons = []

        self.create_button('C Major Scale', 'cmaj', (0,0))
        self.create_button('Mary Had a Little Lamb', 'mary', (0,0))
        self.create_button('Rigadoon', 'rigadoon', (0,0))
        self.create_button('Open Strings', 'open_strings', (0, 400))
        self.create_button('Bach Prelude', 'bach', (0, 400))

        self.resize_elements()

        self.stop_sound_playback()

    def update_score(self, total_score, incremental_score):
        self.score = total_score

    def stop_sound_playback(self):
        # stop sound playback (if happening)
        if self.playback_wav:
            try:
                self.mixer.remove(self.playback_wav)
            except:
                print("already removed")
            self.playback_wav = None
            self.live_wave = None

    def on_update(self):
        if self.state == IN_GAME:
            self.cellist.on_update()
            self.objects.on_update()

            if self.info:
                # self.info.text = "pitch: %.1f\n" % self.cur_pitch
                self.info.text = "%d" % self.score
        self.audio.on_update()

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
        if self.cellist:
            self.cellist.notify_pitch(self.cur_pitch)

        # record to internal buffer for later playback as a WaveGenerator
        if self.recording:
            self.input_buffers.append(frames)

    def on_key_down(self, keycode, modifiers):
        if keycode[1] == 'c' and NUM_CHANNELS == 2:
            self.channel_select = 1 - self.channel_select

        # adjust mixer gain
        gf = lookup(keycode[1], ('up', 'down'), (1.1, 1/1.1))
        if gf:
            new_gain = self.mixer.get_gain() * gf
            self.mixer.set_gain( new_gain )

    def set_bear(self, image_identifier):
        if not image_identifier in self.bear.source:
            self.bear.source = 'images/' +  image_identifier + '.gif'


    def _process_input(self) :
        data = combine_buffers(self.input_buffers)
        print('live buffer size:', len(data) / NUM_CHANNELS, 'frames')
        write_wave_file(data, NUM_CHANNELS, 'recording.wav')
        self.live_wave = WaveArray(data, NUM_CHANNELS)
        self.input_buffers = []

# pass in which MainWidget to run as a command-line arg
run(MainWidget1)
