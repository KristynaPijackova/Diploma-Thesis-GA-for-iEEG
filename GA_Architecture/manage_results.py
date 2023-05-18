import numpy as np
import copy
import random
import string
import itertools
import json
from datetime import datetime
from solution import Solution
from inception_ask_blocks import *

class GenerationsManager:
    def __init__(self, population_size, n_classes,
                 inp=8,
                 inc_red=32,
                 max_num_blocks=3,
                 max_num_subblocks=5,
                 subblock_types = ['conv', 'maxpool', 'avgpool', 'identity'],
                 conv_kernels=[1, 3, 5, 7],
                 out_channels=[16, 32, 64, 128, 256],
                 mutation_rate=0.2,
                 timeout=8000):

        self.population_size = population_size
        self.n_classes = n_classes
        self.inp = inp
        self.inc_red = inc_red
        self.max_num_blocks = max_num_blocks
        self.max_num_subblocks = max_num_subblocks
        self.subblock_types = subblock_types
        self.conv_kernels = conv_kernels
        self.out_channels = out_channels
        self.mutation_rate = mutation_rate
        self.population = []
        self.timeout = timeout  # [seconds] waiting time before the parameters are reassigned
        self.solution = 0
        self.json_result_list = []  # list with dictionary structured data for json data storage
        self.enc_keys = ['arch_enc', 'outputs', 'inc_channels']
        self.ae = ae = ArchitectureEncoding(self.inp, self.max_num_blocks, self.max_num_subblocks, self.inc_red,
                                  self.subblock_types, self.conv_kernels, self.out_channels)

    def __getitem__(self, item):
        return self.population[item]

    def __len__(self):
        return len(self.population)

    def __population__(self):
        return self.population

    def __parameters__(self):
        return [i.parameters for i in self.population]

    def __scores__(self):
        return [i.score for i in self.population]

    def generate_random_id(self):
        id = ''.join(random.choice(string.ascii_letters) for i in range(32))
        return id

    def generate_architecture(self):
        # initialize the ArchitectureEncoding class

        parameters = []

        for i in range(self.population_size):
            arch_enc, outputs, inc_channels = self.ae.architecture_encoding_generator()
            # id_name = self.generate_random_id()
            # name = 'ENCODING_' + id_name
            d = dict(zip(self.enc_keys, [arch_enc, outputs, inc_channels]))
            parameters.append([arch_enc, outputs, inc_channels])
        return parameters

    def append_population(self, population):
        "Call this in the main file where you generate the architecture to save it into the solutions when the specific population is called"
        for p in population:
            self.population.append(Solution(p))

    def add_result(self, id, score):
        kappa, f1, epoch, duration, auroc, auprc, auroc_all, auprc_all, f1_all, trainable_params = score
        print(kappa)
        for i, p in enumerate(self.population):
            if p.id == id:
                self.population[i].score = kappa
                self.population[i].f1 = f1
                self.population[i].auroc = auroc
                self.population[i].auprc = auprc
                self.population[i].auroc_all = auroc_all
                self.population[i].auprc_all = auprc_all
                self.population[i].f1_all = f1_all
                self.population[i].trainable_params = trainable_params
                self.population[i].epoch = epoch
                self.population[i].duration = duration
                self.population[i].finished = True
        # self.population.sort(key=lambda x: x.score, reverse=True)

    def get_next_solution(self, n_best=3):
        self.solution += 1
        for i, p in enumerate(self.population):
            # if not assigned
            if p.assigned == False:
                self.population[i].assign()
                return p

            # if assigned, but didnt finish in [self.timeout] seconds
            now = datetime.timestamp(datetime.now())
            if (p.assigned == True) and (p.finished == False) and (now - p.assigned_timestamp > self.timeout):
                self.population[i].assign()
                print("reassign")
                return p

        # self.population.sort(key=lambda x: x.score,reverse=True)
        candidates_pool = self.population[-self.population_size:]
        idx_candidates = [p.id for p in candidates_pool]

        scores = [p.score for p in self.population]  # get scores of the whole population
        best_idxs = np.argsort(-np.array(scores))[:n_best]  # find indexes of n best
        best_candidates = [self.population[idx] for idx in best_idxs]  # create list of the n best candidates
        id_best = [p.id for p in best_candidates]  # get the id (to compare duplicates with the last N candidates in the pool

        # idxs = self.check_duplicates_between_lists(idx_candidates, id_best, best_idxs)  # check for the duplicates and return list indexes for population which is not already in the pool

        if id_best:
            for idx in best_idxs:
                candidates_pool.append(self.population[idx])  # append the best to the pool

        while True:
            A, B = self.tournament(candidates_pool, 2)
            print(A, B)
            a, o, i = A.parameters
            b, o, i = B.parameters
            cross, o, i = self.ae.architecture_crossover(a, b)
            mut, o, i = self.ae.architecture_mutation(cross, i, self.mutation_rate)
            aen = Architecture_Encoder(in_channels=self.inp,
                                       architecture=mut,
                                       output_channels=o,
                                       inception_output_channels=i,
                                       n_classes=self.n_classes,)
            # d = dict(zip(self.enc_keys, [mut, o, i]))
            new_arch = Solution([mut, o, i])
            new_arch.assign()
            self.population.append(new_arch)
            break

        return new_arch

    def show(self):
        for p in self.population[:self.population_size]:
            print(p)

    @staticmethod
    def tournament(candidates_pool, tournament_size=2):
        """Choose indexes with tournament strategy."""
        winners = []
        for i in range(2):
            idxs = random.sample(range(len(candidates_pool)), tournament_size)  # select tournament candidates
            tmp_scores = [candidates_pool[idx].score for idx in idxs]  # get the score of each individual in tournament
            winners.append(candidates_pool[idxs[np.argmax(tmp_scores)]])  # append the fittest candidate from the tournament
        return winners

    @staticmethod
    def check_duplicates_between_lists(list_a, list_b, best_idxs):
        """Check for duplicates between two lists of lists."""
        idxs = []
        best_idxs = list(best_idxs)
        for individual in list_a:
            for i, new_individual in enumerate(list_b):
                if individual == new_individual:
                    idxs.append(i)

        for idx in sorted(idxs, reverse=True):
            list_b.pop(idx)
            best_idxs.pop(idx)
        return best_idxs

    @staticmethod
    def check_duplicate_in_list(generation):
        """Check and erase duplicates within a list."""
        generation.sort()
        generation = list(generation for generation, _ in itertools.groupby(generation))
        return generation

    @staticmethod
    def check_duplicates_dictionaries(list_of_dicts):
        return [dict(t) for t in {tuple(d.items()) for d in list_of_dicts}]

    # JSON file storing functions
    def json_header(self, day, info, train_data, valid_data,
                    population_size, nn_epochs):
        """Dictionary structure to store infos about the set up and the results and parameters of the individuals."""
        header = {"date": day,
                  "GPUs": info,
                  "training data": train_data,
                  "validation data": valid_data,
                  "nn epochs": nn_epochs,
                  "population size": self.population_size,
                  "number of classes": self.n_classes,
                  "upscale input": self.inp,
                  "inception channel reduction": self.inc_red,
                  "number of inception blocks": self.max_num_blocks,
                  "number of layers in inception block": self.max_num_subblocks,
                  "types of layers": self.subblock_types,
                  "convolution kernels": self.conv_kernels,
                  "output channels": self.out_channels,
                  "mutation rate": self.mutation_rate,
                  "results": self.json_result_list,}
        return header

    def json_results(self, id):
        """Create sub-dictionary to store result of current individual along with its parameters and append in list."""

        for i, p in enumerate(self.population):
            if p.id == id:
                result_data = {"id": id,
                               "parameters": str(self.population[i].parameters),
                               "kappa score": str(self.population[i].score),
                               "f1 score": str(self.population[i].f1),
                               "auroc": self.population[i].auroc,
                               "auprc": self.population[i].auprc,
                               "auroc_all": {"powerline": self.population[i].auroc_all[0], "noise": self.population[i].auroc_all[1],
                                             "pathology": self.population[i].auroc_all[2], "physiology": self.population[i].auroc_all[3]},
                               "auprc_all": {"powerline": self.population[i].auprc_all[0], "noise": self.population[i].auprc_all[1],
                                             "pathology": self.population[i].auprc_all[2], "physiology": self.population[i].auprc_all[3]},
                               "f1_all": {"powerline": self.population[i].f1_all[0], "noise": self.population[i].f1_all[1],
                                          "pathology": self.population[i].f1_all[2], "physiology": self.population[i].f1_all[3]},
                               "best_epoch": str(self.population[i].epoch),
                               "elapsed time": str(self.population[i].duration),
                               "trainable parameters": str(self.population[i].trainable_params),
                               }
        self.json_result_list.append(result_data)  # store the dictionary with the result into list
        return self

    def json_update(self, file_name, data):
        """Update the results key of the created json file (updated in each epoch iteration.)"""
        data["results"] = self.json_result_list  # update the main dictionary and its result content first
        f = open(file_name + '.json', 'w')
        json.dump(data, f, indent=2)
        f.close()
        return self
