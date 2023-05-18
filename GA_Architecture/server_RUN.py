import zmq
import time
from datetime import date
import datetime
from manage_results import *

if __name__ == "__main__":
    # SET UP PARAMETERS

    # architecture
    population_size = 20
    n_classes = 4
    inp = 4
    inc_red = 32
    max_num_blocks = 5
    max_num_subblocks = 3
    subblock_types = ['conv', 'conv', 'maxpool', 'avgpool', 'identity']
    conv_kernels = [1, 3]
    out_channels = [16, 32, 64, 128]
    mutation_rate = 0.05
    enc_keys = ['arch_enc', 'outputs', 'inc_channels', 'inp', 'n_classes', 'nn_epochs']

    # training
    nn_epochs = 10

    # json
    day = datetime.now().strftime("%b%d_%H-%M-%S")
    info = "test..."
    train = 'train_small_1_fnusa.csv'
    valid = 'valid_fnusa.csv'
    json_filename = 'results_test' + str(day)

    # INITIALIZE GENERATION MANAGER
    ga = GenerationsManager(population_size=population_size, n_classes=n_classes,
                 inp=inp, inc_red=inc_red, max_num_blocks=max_num_blocks,
                 max_num_subblocks=max_num_subblocks, subblock_types=subblock_types,
                 conv_kernels=conv_kernels, out_channels=out_channels, mutation_rate=mutation_rate,
                 timeout=8000)

    # GENERATE NEW POPULATION
    population = ga.generate_architecture()
    print(population)
    # SAVE POPULATION to a list with parameters from Solution()
    ga.append_population(population)

    # INITIALIZE JSON FILE
    header = ga.json_header(day, info, train, valid, population_size, nn_epochs)

    # SET UP SERVER
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://insert your local IP")  # insert your IP
    print("running...")
    idx = 0
    while True:
        message = socket.recv_pyobj()
        print('testing if it reacts at all...')
        print(message)
        time.sleep(1)

        if message['request'] == "params":
            print("sending message...")
            next_solution = ga.get_next_solution()
            arch_enc, o, i = next_solution.parameters
            next_sol = dict(zip(enc_keys, [arch_enc, o, i, inp, n_classes, nn_epochs]))
            print('Next Solution: {}\n'.format(next_sol))
            socket.send_pyobj({'task': next_sol, 'id': next_solution.id})

        if message['request'] == "result":
            ga.add_result(id=message['id'], score=message['score'])
            # ga.show()
            ga.json_results(id=message['id'])
            ga.json_update(json_filename, header)
            socket.send_pyobj("Ack")
