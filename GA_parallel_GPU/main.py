import numpy as np
import torch
import time
import glob
import zmq
import subprocess
import random
from run_model import *
from test_function import *


if __name__ == '__main__':

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://your IP adress")  # include your IP adress

    name = subprocess.check_output(['bash', '-c', "hostname"])
    name = name.decode('ascii').strip()

    idx = 0
    while True:
        socket.send_pyobj({"request": "params", "name": name})
        message = socket.recv_pyobj()
        print(message)

        if message['task'] == "Finish":
            print("Break")
            break

        # extract parameters and train NN
        score = run_model(**message['task'])
        print(score)

        # send result
        socket.send_pyobj({'request': 'result',
                           'score': score,
                           'name': name,
                           'id': message['id']})
        # wait for Ack signal
        message = socket.recv_pyobj()
        idx += 1


