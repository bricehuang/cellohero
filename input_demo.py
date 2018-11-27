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
#from kivy.core.image import Image
from kivy.uix.image import Image
from kivy.uix.button import Button

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

STAFF_LEFT_X = 100
NOW_BAR_X = 250
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

PERCENT_NOTE_TO_HIT = .8
ACCEPTABLE_PITCH_INTERVAL = 0.2

def note_to_lower_left(noteid):
    return MID_C_LOWER_Y + (noteid - MIDDLE_C_ID)*LANE_SEP

class TuningArrow(InstructionGroup):
    ON = "on"
    OFF = "off"

    UP = "up"
    DOWN = "down"


    def __init__(self, pos, direction):
        super(TuningArrow, self).__init__()
        self.pos = pos
        self.default_size = 50

        #self.arrow = CEllipse(cpos = self.pos, csize = (self.default_size), segments = 3)
        self.color = Color(1,0,0,0)

        # Attempt to upload picture
        if direction == TuningArrow.UP:
            source = 'arrow_up.png'
        else:
            source = 'arrow_down.png'

        texture = Image(source = source).texture

        self.arrow = CRectangle(texture = texture, cpos=self.pos, csize=(self.default_size, self.default_size))

        self.status = TuningArrow.OFF
        self.direction = direction

        # Attempt to rotate
        #self.status = TuningArrow.ON
        #angle = 200
        #self.rotation = Rotate() #(axis = (1,0,0), angle= angle)
        #self.rotation.set(200, 1, 0, 0)
        #self.anti_rotation = Rotate() #(axis = (1,0,0), angle= -angle)
        #self.anti_rotation.set(90, 0, 0, 1)
        #self.add(self.rotation)
        self.add(self.color)
        self.add(self.arrow)
        #self.add(self.anti_rotation)


    def turn_on(self, size = None):
        if self.status == TuningArrow.OFF:
            self.color.a = 1

            if size:
                self.set_size(size)
            else:
                self.set_size(1)

            self.status = TuningArrow.ON


    def turn_off(self):
        if self.status == TuningArrow.ON:
            self.color.a = 0
            self.status = TuningArrow.OFF

    def set_size(self, size_multiplier):
        size = self.default_size * size_multiplier
        self.arrow.csize = (size, size)

    def on_update(self, dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.arrow.cpos = self.pos

    def is_on(self):
        return self.status == TuningArrow.ON

class IntonationDisplay(InstructionGroup):
    def __init__(self, up_arrow_positions, down_arrow_positions):
        super(IntonationDisplay, self).__init__()
        self.arrows = []
        for i in range(len(up_arrow_positions)):
            up_arrow = TuningArrow(up_arrow_positions[i], TuningArrow.UP)
            down_arrow = TuningArrow(down_arrow_positions[i], TuningArrow.DOWN)
            self.arrows.append({
                TuningArrow.UP: up_arrow,
                TuningArrow.DOWN: down_arrow
                })
            self.add(up_arrow)
            self.add(down_arrow)

        self.size_multiplier = 10

    def on_update(self, dt):
        for i in range(len(self.arrows)):
            self.get_up_arrow(i).on_update(dt)
            self.get_down_arrow(i).on_update(dt)

    def change_intonation_display(self, amount_out_of_tune, gem_idx): # positive amount_out_of_tune = too high
        up_arrow = self.get_up_arrow(gem_idx)
        down_arrow = self.get_down_arrow(gem_idx)

        if(abs(amount_out_of_tune) <= ACCEPTABLE_PITCH_INTERVAL):
            up_arrow.turn_off()
            down_arrow.turn_off()
        else:
            if(abs(amount_out_of_tune) > 1):
                size_multiplier = 2
            else:
                size_multiplier = int(np.interp(abs(amount_out_of_tune), (0, 1), (.5, 2)))

            if(amount_out_of_tune > 0): # too high
                down_arrow.turn_on(size_multiplier)
                up_arrow.turn_off()
            else:
                up_arrow.turn_on(size_multiplier)
                down_arrow.turn_off()

        # turn off the previous note's arrows
        if gem_idx > 0:
            if self.atleast_one_arrow_on(gem_idx - 1):
                self.get_up_arrow(gem_idx - 1).turn_off()
                self.get_down_arrow(gem_idx - 1).turn_off()

    def atleast_one_arrow_on(self, gem_idx):
        return self.get_up_arrow(gem_idx).is_on() or self.get_down_arrow(gem_idx).is_on()

    def get_up_arrow(self, gem_idx):
        return self.arrows[gem_idx][TuningArrow.UP]

    def get_down_arrow(self, gem_idx):
        return self.arrows[gem_idx][TuningArrow.DOWN]

LEDGER_LINE_WIDTH = 50
class LedgerLine(InstructionGroup):
    def __init__(self, note, left_px):
        super(LedgerLine, self).__init__()
        assert (note%2 == 0 and (note >= MIDDLE_C_ID or note <= E2_ID))
        self.color = Color(.8,.8,.2)
        self.add(self.color)
        self.pos = np.array((left_px, note_to_lower_left(note) + LANE_SEP))
        self.line = Line(points=[self.pos[0],self.pos[1],self.pos[0]+LEDGER_LINE_WIDTH,self.pos[1]],width=2)
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
    def __init__(self, note, left_px, duration_beats):
        super(NoteFigure, self).__init__()
        beats = round(duration_beats, 1)
        self.note = note
        self.add(Color(.8,.8,.2))
        self.pos = np.array((left_px + 3, note_to_lower_left(note) + 3))
        self.head = Ellipse(pos=self.pos, size=(2*NOTE_RADIUS,2*NOTE_RADIUS), segments=40)
        self.add(self.head)

        self.stem = None
        if duration_beats in {0.5,1.,1.5,2.,3.}:
            if self.note >= 22:
                self.stem = Line(points=[
                    self.pos[0],
                    self.pos[1] + NOTE_RADIUS,
                    self.pos[0],
                    self.pos[1] + NOTE_RADIUS - STEM_LENGTH
                ], width=2)
            else:
                self.stem = Line(points=[
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS,
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS + STEM_LENGTH
                ], width=2)
            self.add(self.stem)

        self.flag = None
        if duration_beats in {0.5}:
            if self.note >= 22:
                self.flag = Line(points=[
                    self.pos[0],
                    self.pos[1]+NOTE_RADIUS-STEM_LENGTH,
                    self.pos[0] - 25,
                    self.pos[1]+NOTE_RADIUS-STEM_LENGTH + 50,
                ], width=2)
            else:
                self.flag = Line(points=[
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS + STEM_LENGTH,
                    self.pos[0] + 2*NOTE_RADIUS + 25,
                    self.pos[1] + NOTE_RADIUS + STEM_LENGTH - 50
                ], width=2)
            self.add(self.flag)

        self.dot = None
        if duration_beats in {1.5,3.}:
            if self.note >= 22:
                self.dot = Ellipse(pos = self.pos+np.array((45,0)), size=(12,12), segments=40)
            else:
                self.dot = Ellipse(pos = self.pos+np.array((-7,36)), size=(12,12), segments=40)
            self.add(self.dot)


        self.add(Color(1,1,1))
        self.headcenter = None
        if duration_beats in {2.,3.,4.}:
            self.headcenter = Ellipse(pos=self.pos+np.array((8,8)), size=(2*(NOTE_RADIUS-8),2*(NOTE_RADIUS-8)), segments=40)
            self.add(self.headcenter)


    def on_update(self, dt):
        self.pos += np.array([-NOTE_SPEED*dt, 0])
        self.head.pos = self.pos
        if self.headcenter is not None:
            self.headcenter.pos = self.pos+np.array((8,8))
        if self.stem is not None:
            if self.note >= 22:
                self.stem.points = [
                    self.pos[0],
                    self.pos[1] + NOTE_RADIUS,
                    self.pos[0],
                    self.pos[1] + NOTE_RADIUS-STEM_LENGTH
                ]
            else:
                self.stem.points =[
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS,
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS+STEM_LENGTH
                ]
        if self.flag is not None:
            if self.note >= 22:
                self.flag.points=[
                    self.pos[0],
                    self.pos[1]+NOTE_RADIUS-STEM_LENGTH,
                    self.pos[0] - 25,
                    self.pos[1]+NOTE_RADIUS-STEM_LENGTH + 50,
                ]
            else:
                self.flag.points = [
                    self.pos[0] + 2*NOTE_RADIUS,
                    self.pos[1] + NOTE_RADIUS + STEM_LENGTH,
                    self.pos[0] + 2*NOTE_RADIUS + 25,
                    self.pos[1] + NOTE_RADIUS + STEM_LENGTH - 50
                ]
        if self.dot is not None:
            if self.note >= 22:
                self.dot.pos = self.pos+np.array((45,0))
            else:
                self.dot.pos = self.pos+np.array((-7,36))

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

        self.rect = Rectangle(pos=self.pos, size=(duration_time*NOTE_SPEED, LANE_HEIGHT-2*NOTE_RECT_MARGIN))
        self.add(self.rect)

        # ledger lines
        self.ledger_lines = LedgerLine.get_ledger_lines(self.noteid, self.pos[0])
        for ll in self.ledger_lines:
            self.add(ll)

        # note head
        self.figure = NoteFigure(self.noteid, self.pos[0], self.duration_beats)
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
            width=1
        )
        self.add(self.line)

    def on_update(self, dt):
        self.x -= NOTE_SPEED*dt
        self.line.points = [self.x, STAFF_Y_VALS[0], self.x, STAFF_Y_VALS[-1]]

class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data):
        # note_info is List[(parsed_pitch, start_time, duration)]
        super(BeatMatchDisplay, self).__init__()
        note_info = song_data.notes
        bar_info = song_data.barlines

        # draw staff lines
        self.add(Color(1,1,1))
        for y in STAFF_Y_VALS:
            self.add(Line(points=[NOW_BAR_X,y,Window.width,y], width=2))

        self.add(Line(points=[NOW_BAR_X,STAFF_Y_VALS[0],NOW_BAR_X,STAFF_Y_VALS[-1]], width=2))
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
        self.add(Rectangle(pos=(0,0),size=(NOW_BAR_X,Window.height-200)))

        # draw part of staff left of now bar
        self.add(Color(1,1,1))
        for y in STAFF_Y_VALS:
            self.add(Line(points=[STAFF_LEFT_X,y,NOW_BAR_X,y], width=2))
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
        self.bassClef = Rectangle(texture = Image(source = "white_bass_clef.png").texture, pos=(STAFF_LEFT_X, STAFF_Y_VALS[1]), size=(abs(NOW_BAR_X - STAFF_LEFT_X), abs(STAFF_Y_VALS[-1] - STAFF_Y_VALS[1])))
        self.add(self.bassClef)

        # draw bear
        #self.bear = Rectangle(texture = Image(source = "images/good_job_bear.gif").texture, pos=(0, 0), size=(400,400))
        #self.add(self.bear)


        # add intonation adjustion arrows
        ARROW_BUFFER = 200
        up_arrow_positions = []
        down_arrow_positions = []
        for note in self.notes:
            center_position = note.get_center_position()
            up_arrow_positions.append(center_position - np.array([0, ARROW_BUFFER]))
            down_arrow_positions.append(center_position + np.array([0, ARROW_BUFFER]))
        self.intonationDisplay = IntonationDisplay(up_arrow_positions, down_arrow_positions)
        self.add(self.intonationDisplay)


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

        self.intonationDisplay.on_update(dt)

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

    def inform_pitch_diff(self, gem_idx, pitch_difference):
        self.intonationDisplay.change_intonation_display(pitch_difference, gem_idx)

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
        time_passed = kivyClock.frametime
        current_note = self.display.current_note()
        if current_note is not None:
            idx, pitch = current_note
            pitch_diff = np.abs(pitch - self.cur_pitch)
            self.display.inform_pitch_diff(idx, self.cur_pitch - pitch)
            #print ("current pitch: ", self.cur_pitch, "playing_pitch: ", pitch)
            if pitch_diff < ACCEPTABLE_PITCH_INTERVAL:
                if not self.display.enough_note_hit(idx, True):
                    self.score += 1000 * time_passed
                    self.score_cb(self.score)
                    self.display.note_hit(idx, time_passed)
        missed_notes = self.display.missed_notes()
        for idx in missed_notes:
            self.display.note_pass(idx, time_passed)

PITCHES_IN_OCTAVE = 12
MIDI_ADJ = 12
NOTES_IN_OCTAVE = 7
NOTE_TO_MIDI = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}
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
        self.display = BeatMatchDisplay(self.song_data)
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


        # BEAR ANIMATION!
        bear_size = 500
        self.bear = Image(source = "images/good_job_bear.gif", size = (bear_size, bear_size), pos = (Window.width/2 - bear_size/2, 0)) #Rectangle(texture = Image(source = "images/good_job_bear.gif").texture, pos=(0, 0), size=(400,400))
        self.add_widget(self.bear)

        # BUTTON EXAMPLE
        # button1 = Button(text = 'Mary Had a Litte Lamb', size = (500, 300), font_size = 40)
        # button1.bind(state = self.select_song_callback)
        # self.add_widget(button1)

    # button callback
    # def select_song_callback(self, instance, value):
    #     print ("instance: ", instance, " value: ", value)

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

        # changing bear image
        if keycode[1] == '1':
            self.bear.source = 'images/spinning_bear.gif'

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