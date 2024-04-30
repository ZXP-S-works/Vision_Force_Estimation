from xmlrpc.client import ServerProxy
from threading import Thread, Lock
import threading
import time
import os
import numpy as np
dirname = os.path.dirname(__file__)

class FTClient:
    def __init__(self, address = 'http://127.0.0.1:8080'):
        self.s = ServerProxy(address)
        #self.shut_down = False

    def zero_ft_sensor(self):
        self.s.zero_ft_sensor()

    def start_ft_sensor(self):
        self.s.start_ft_sensor()

    def read_ft_sensor(self):
        return self.s.read_ft_sensor()

    def shutdown_ft_sensor(self):
        self.s.shutdown_ft_sensor()

if __name__=="__main__":
    # ft_driver = FTClient('http://192.168.9.118:80') #FTClient('http://192.168.0.103:80')
    # ft_driver = FTClient('http://172.29.112.1:8080')
    # ft_driver = FTClient('http://192.168.0.103:80')
    ft_driver = FTClient('http://192.168.0.103:80')
    ft_driver.zero_ft_sensor()
    ft_driver.start_ft_sensor()
    for i in range(3000):
        start_time = time.time()
        print(ft_driver.read_ft_sensor())
        time.sleep(0.1)
    ft_driver.shutdown_ft_sensor()