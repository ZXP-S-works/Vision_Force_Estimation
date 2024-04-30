#!/usr/bin/python3
import socket
import struct
import threading
import time
from threading import Thread
import numpy as np
import copy
COUNTS_PER_FORCE = 1000000
COUNTS_PER_TORQUE = 1000000


class FT_reading:
    '''The struct referencing a single FT reading instantiation.
    The class is responsible to keeping a measurement and feeding
    readings back to the user.
    '''

    def __init__(self, measurement=[0] * 6, timestamp=0):
        '''Start data saving instantiation
        Args:
               measurement (list): the 6 values read via the Ft sensor
               timestamp (float): time the sensor value was taken
        '''
        self.measurement = measurement
        self.timestamp = timestamp

    def getMeasurement(self):
        ''' Return the full 6-axis measurement back to the user
        '''
        return self.measurement

    def getForce(self):
        ''' Return the force measurement back to the user
        '''
        return self.measurement[:3]

    def getTorque(self):
        ''' Return the torque measurement back to the user
        '''
        return self.measurement[3:]


class TimeStampQueue:
    '''
    Queue that cyclically write data
    The shape is size x dim
    '''

    def __init__(self, size):
        self.time_stamps = np.zeros(size)
        self.data = [0] * size
        self.idx = -1
        self.size = size
        self.rw_lock = threading.Lock()

    def add(self, data, time_stamp):
        self.rw_lock.acquire()
        self.idx = (self.idx + 1) % self.size
        self.data[self.idx] = data
        self.time_stamps[self.idx] = time_stamp
        self.rw_lock.release()

    def get(self, time_stamp, lag):
        time_stamp -= lag
        self.rw_lock.acquire()
        if self.time_stamps.min() < time_stamp < self.time_stamps.max():
            min_t_gap_idx = np.abs(self.time_stamps - time_stamp).argmin(0)
            self.rw_lock.release()
            return self.data[min_t_gap_idx], self.time_stamps[min_t_gap_idx]
        else:
            print('failed to find data for the time stamp')
            print('t0 {:.02f}, t1 {:.02f}, t2 {:.02f}'.format(0, time_stamp - self.time_stamps.min(),
                                                              self.time_stamps.max() - self.time_stamps.min()))
            self.rw_lock.release()
            return None


class FTSensor:
    '''The class interface for an ATI Force/Torque sensor.
     This class contains all the functions necessary to communicate
     with an ATI Force/Torque sensor with a Net F/T interface
     using RDT.
     '''

    def __init__(self, ip='192.168.1.1', time_drift=[0, 0, 0, 0, 0, 0]):
        # time_drift=[8e-05, -8e-05, 1.96413949e-04, 0, 0, 0]
        '''Start the sensor interface
          This function intializes the class and opens the socket for the
          sensor.
          Args:
               ip (str): The IP address of the Net F/T box.
          '''
        self.ip = ip
        self.port = 49152
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.ip, self.port))
        self.mean = [0.] * 6
        self.mean_lock = threading.Lock()
        self.FT = FT_reading()
        self.stream = False
        self.buffer = TimeStampQueue(1000)
        self.time_drift = time_drift  # force drift along xyz per second
        self.t0 = None

    def receive(self):
        '''receives and unpacks a response from the Net F/T box.
          This function receives and unpacks an RDT response from the Net F/T
          box and saves it to the data class attribute.
          Returns:
               list of float: The force and torque values received. The first three
                    values are the forces recorded, and the last three are the measured
                    torques.
          '''
        rawdata = self.sock.recv(1024)
        data = struct.unpack('!IIIiiiiii', rawdata)[3:]
        return data
        # output = [data[i] - self.mean[i] for i in range(6)]
        # return output

    # def calculateMean(self, n=10):
    #     '''Mean the sensor.
    #       This function takes a given number of readings from the sensor
    #       and averages them. This mean is then stored and subtracted from
    #       all future measurements.
    #       Args:
    #            n (int, optional): The number of samples to use in the mean.
    #                 Defaults to 10.
    #       Returns:
    #            list of float: The mean values calculated.
    #       '''
    #     self.mean = [0] * 6
    #     self.getMeasurements(n=n)
    #     mean = [0] * 6
    #     for i in range(n):
    #         self.receive()
    #         for i in range(6):
    #             mean[i] += self.measurement()[i] / float(n)
    #     self.mean = mean
    #     return mean

    def receiveHandler(self):
        '''A handler to receive and store data.'''
        while self.stream:
            self.measurement()

    def measurement(self):
        '''Get the most recent force/torque measurement
          Returns:
               list of float: The force and torque values received. The first three
                    values are the forces recorded, and the last three are the measured
                    torques.
          '''
        measured_data = [0, 0, 0, 0, 0, 0]
        output = self.receive()
        dt = time.time() - self.t0 if self.t0 is not None else 0
        for i in range(6):
            measured_data[i] = output[i] / COUNTS_PER_FORCE - self.mean[i] - dt * self.time_drift[i]
        self.FT = FT_reading(measured_data, time.time())
        self.buffer.add(np.asarray(measured_data), time.time())
        return measured_data

    def force(self):
        '''Get the most recent force measurement
          Returns:
               list of float: The force values received.
          '''
        return self.FT.getForce()

    def torque(self):
        '''Get the most recent torque measurement
          Returns:
               list of float: The torque values received.
          '''
        return self.FT.getTorque()

    def getMeasurements(self, n):
        '''Request a given number of samples from the sensor
          This function requestes a given number of samples from the sensor. These
          measurements must be received manually using the `receive` function.
          Args:
               n (int): The number of samples to request.
          '''
        self.send(2, count=n)

    def send(self, command, count=0):
        '''Send a given command to the Net F/T box with specified sample count.
          This function sends the given RDT command to the Net F/T box, along with
          the specified sample count, if needed.
          Args:
               command (int): The RDT command.
               count (int, optional): The sample count to send. Defaults to 0.
          '''
        header = 0x1234
        message = struct.pack('!HHI', header, command, count)
        self.sock.sendto(message, (self.ip, self.port))

    def startStreaming(self, threaded=True):
        '''Start streaming data continuously
          This function commands the Net F/T box to start sending data continuouly.
          By default this also starts a new thread with a handler to save all data
          points coming in. These data points can still be accessed with `measurement`,
          `force`, and `torque`. This handler can also be disabled and measurements
          can be received manually using the `receive` function.
          Args:
               handler (bool, optional): If True start the handler which saves data to be
                    used with `measurement`, `force`, and `torque`. If False the
                    measurements must be received manually. Defaults to True.
          '''
        self.getMeasurements(0)
        if threaded:
            self.stream = True
            self.thread = Thread(target=self.receiveHandler)
            self.thread.daemon = True
            self.thread.start()
        time.sleep(1)
        # offset = np.asarray([0.05, -0.03, 0, 0, 0, 0])
        offset = np.asarray([0, 0, 0, 0, 0, 0])
        self.mean = (np.asarray(self.buffer.data[:10]).mean(0) + offset).tolist()
        self.t0 = time.time()

        print('FT sensor started')

    def stopStreaming(self):
        '''Stop streaming data continuously
          This function stops the sensor from streaming continuously as started using
          `startStreaming`.
          '''
        self.stream = False
        time.sleep(0.1)
        self.send(0)


'''How to fundamentally get this to work :)
1. Make sure you can ping the IP of the sensor. I made it a static IP in its configuration for ease (to make sure things
don't change)
You may need to set you network configuration on your computer to be on the same submet (set IP to e.g. 192.168.1.5 and 
netmask to 255.255.255.0 )
2. Enable RDT transfer in the comms page. The higher Hz output, the better (potentially). I have it set to 200hz
currently and that's about as high as I can get out of the mini:)
'''

if __name__ == "__main__":

    # Initiate the sensor and go from there
    s = FTSensor()
    s.startStreaming()

    # Take some measurements and tell us how much time was in between each reading
    tic = time.time()

    # Loop and print out the timing between different readings.
    while True:
        v, t = s.FT.measurement, s.FT.timestamp
        # print(time.time() - tic)
        # tic = time.time()
        print("value: ", np.round(v, 1))
        print('time: ', np.round(t, 1))
        time.sleep(0.2)
