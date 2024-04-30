from tracemalloc import start
from openhand_node.hands import Model_T42

if __name__ == '__main__':
    import time
    T = Model_T42(port='/dev/ttyUSB0', s1=1, s2=2, dyn_model='XM', s1_min=0.35, s2_min=0.04)    #T.readHand()

    start_time = time.time()
    T.servos[0].read_encoder()
    print(time.time() - start_time)

    for i in range(100):
        start_time = time.time()
        T.readServoInfos()
        print(time.time() - start_time)
    #T.readLoads()
    #T.close(0.3)
    #T.release()
