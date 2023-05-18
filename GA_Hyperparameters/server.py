import zmq
import time
from datetime import date
import datetime
from GA import *

if __name__ == "__main__":
    # Generate initial population
    POPULATION_SIZE = 30
    ga = Generations(POPULATION_SIZE)

    # Initialize population with some of the already known good options
    spectrogram_params = [[0, 6, 1, 1], [1, 6, 1, 2]] # , [1, 6, 1, 2]
    nn_params = [ [128, 3, 128, 2], [256, 3, 128, 1], [64, 3, 128, 3]] #  [64, 3, 128, 3]
    general = [32, 0.001, 0.0001]

    init_population_list = []
    for spect in spectrogram_params:
        for nn in nn_params:
            init_population_list.append(spect + nn + general)
            print(spect + nn + general)
    # Generate rest of random combos

    init_population = []
    for individual in init_population_list:
        init_population.append({
            "WINDOW": individual[0], "NPERSEG": individual[1], "NOVERLAP": individual[2], "NFFT": individual[3],
            "NFILT": individual[4], "KERNEL": individual[5], "HIDDEN": individual[6], "NGRUS": individual[7],
            "BATCH": individual[8], "LR": individual[9], "L2": individual[10]})


    population = ga.generate_NN_parameters(POPULATION_SIZE-len(init_population_list))

    # Check for duplicates
    new_population = init_population + population
    print(new_population)
    population = ga.check_duplicates_dictionaries(new_population)

    # Append the population to a list with parameters from Solution()
    # population = ga.generate_NN_parameters(POPULATION_SIZE)
    ga.append_population(population)

    # json file initialization
    # day = date.today().strftime("%d_%m_%Y")
    day = datetime.now().strftime("%b%d_%H-%M-%S")
    # INFO
    info = "pop size 30"
    train = 'train_small_1_fnusa.csv'
    valid = 'valid_fnusa.csv'
    nn_epochs = 10
    # json_filename = os.path.join('results', ('result_multipleGPUs_' + str(day)))
    json_filename = 'results_multipleGPUs_' + str(day)
    header = ga.json_header(day, info, train, valid, POPULATION_SIZE, nn_epochs)


    # set up server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://your IP adress")  # include your IP adress
    print("running...")
    idx = 0
    while True:
        message = socket.recv_pyobj()
        print(message)
        time.sleep(1)

        if message['request'] == "params":
            print("sending message...")
            next_solution = ga.get_next_solution()
            print('Next Solution: {}\n'.format(next_solution.parameters))
            socket.send_pyobj({'task': next_solution.parameters, 'id': next_solution.id})

        if message['request'] == "result":
            ga.add_result(id=message['id'], score=message['score'])
            # ga.show()
            ga.json_results(id=message['id'])
            ga.json_update(json_filename, header)
            socket.send_pyobj("Ack")
