import numpy as np
import copy
import random
import itertools
import json
from datetime import datetime
from Solution import Solution

class Generations:
"""This class handles operations that are tight to the parameters settings of the individuals. It can create random population, do crossover and mutation operations, check for duplicates and save results and encodings into a json file."""
    def __init__(self, population_size, timeout=8000):
        self.population_size = population_size
        self.population = []
        self.timeout = timeout  # [seconds] waiting time before the parameters are reassigned
        self.solution = 0
        self.json_result_list = []  # list with dictionary structured data for json data storage

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

    def generate_NN_parameters(self, population_size):
        """Randomly generate possible combinations for NN."""
        parameters = []
        for i in range(population_size):
            # Spectrogram HP
            WINDOW = random.randint(0, 1)
            NPERSEG = random.randint(3, 6)
            NOVERLAP = random.randint(1, 6)
            NFFT = random.randint(0, 5)
            # NN HP
            NFILT = 2 ** random.randint(6, 10)
            NKERN = random.choice([3, 5, 7])
            NHIDDEN = 2 ** random.randint(6, 9)  # GRU hidden
            NGRUS = random.choice([1, 2, 3])  # GRU layers
            # Training HP
            BATCH = random.choice([64, 128, 256])
            LR = random.choice([1e-3, 5e-4, 1e-4])
            L2 = random.choice([0, 1e-6, 1e-4])
            parameters.append({ "WINDOW": WINDOW, "NPERSEG": NPERSEG, "NOVERLAP": NOVERLAP, "NFFT": NFFT,
                                "NFILT": NFILT, "KERNEL": NKERN, "HIDDEN": NHIDDEN, "NGRUS": NGRUS,
                                "BATCH": BATCH, "LR": LR, "L2": L2})
        return parameters

    def append_population(self, population):
        for p in population:
            self.population.append(Solution(p))

    def add_result(self, id, score):
        kappa, f1, epoch, duration, auroc, auprc, auroc_all, auprc_all, f1_all = score
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
                self.population[i].epoch = epoch
                self.population[i].duration = duration
                self.population[i].finished = True
        # self.population.sort(key=lambda x: x.score, reverse=True)

    def get_next_solution(self):
        self.solution += 1
        for i,p in enumerate(self.population):
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

        n_best = 3
        scores = [p.score for p in self.population]  # get scores of the whole population
        best_idxs = np.argsort(-np.array(scores))[:n_best]  # find indexes of n best
        best_candidates = [self.population[idx] for idx in best_idxs]  # create list of the n best candidates
        id_best = [p.id for p in best_candidates]  # get the id (to compare duplicates with the last N candidates in the pool
        idxs = self.check_duplicates_between_lists(idx_candidates, id_best, best_idxs)  # check for the duplicates and return list indexes for population which is not already in the pool

        if idxs:
            for idx in best_idxs:
                candidates_pool.append(self.population[idx])  # append the best to the pool

        while True:
            A, B = self.tournament(candidates_pool, 2)
            C = self.crossover(A, B)
            C = self.mutate(C)
            C.assign()

            if C not in self.population:
                self.population.append(C)
                break
        return C

    def show(self):
        for p in self.population[:self.population_size]:
            print(p)

    @staticmethod
    def tournament(candidates_pool, candidates_scores, tournament_size=2):
        """Choose indexes with tournament strategy."""
        winners = []
        for i in range(2):
            idxs = random.sample(range(len(candidates_pool)), tournament_size)  # select tournament candidates
            tmp_scores = [candidates_pool[idx].score for idx in idxs]  # get the score of each individual in tournament
            winners.append(candidates_pool[idxs[np.argmax(tmp_scores)]])  # append the fittest candidate from the tournament
        return winners

    @staticmethod
    def crossover(A, B, CR=0.5):
    """Do crossover operation between 2 chosen parents."""
        new = Solution(copy.deepcopy(A.parameters))
        for key in A.parameters:
            if np.random.rand() < CR:
                new.parameters[key] = A.parameters[key]
            else:
                new.parameters[key] = B.parameters[key]
        return new

    @staticmethod
    def mutate(individual, mutation_rate=0.01):
        """
        Mutate variables.
            WINDOW = random.randint(0, 1)
            NPERSEG = random.randint(5, 9)
            NOVERLAP = random.randint(1, 6)
            NFFT = random.randint(0, 5)
            # NN HP
            NFILT = 2 ** random.randint(6, 10)
            NKERN = random.choice([3, 5, 7, 11, 15])
            NHIDDEN = 2 ** random.randint(6, 9)  # GRU hidden
            NGRUS = random.choice([1,2, 3])  # GRU layers
            # Training HP
            BATCH = random.choice([[64, 128, 256]])
            LR = random.choice([1e-3, 5e-4, 1e-4])
            L2 = random.choice([0, 1e-6, 1e-4])
        """
        if random.random() < mutation_rate:
            individual.parameters['WINDOW'] = random.randint(0, 1)
        if random.random() < mutation_rate:
            individual.parameters['NPERSEG'] = random.randint(3, 5)
        if random.random() < mutation_rate:
            individual.parameters['NOVERLAP'] = random.randint(1, 6)
        if random.random() < mutation_rate:
            individual.parameters['NFFT'] = random.randint(0, 5)
        # NN HP
        if random.random() < mutation_rate:
            individual.parameters['NFILT'] = 2 ** random.randint(6, 10)
        if random.random() < mutation_rate:
            individual.parameters['KERNEL'] = random.choice([3, 5, 7])
        if random.random() < mutation_rate:
            individual.parameters['HIDDEN'] = 2 ** random.randint(6, 9)
        if random.random() < mutation_rate:
            individual.parameters['NGRUS'] = random.choice([1, 2, 3])
        # Training HP
        if random.random() < mutation_rate:
            individual.parameters['BATCH'] = random.choice([64, 128, 256])
        if random.random() < mutation_rate:
            individual.parameters['LR'] = random.choice([1e-3, 5e-4, 1e-4])
        if random.random() < mutation_rate:
            individual.parameters['L2'] = random.choice([0, 1e-6, 1e-4])

        return individual

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
                  "initial population size": population_size,
                  "nn epochs": nn_epochs,
                  "results": self.json_result_list}
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
