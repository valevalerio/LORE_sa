from lore_sa.neighgen.neighgen import NeighborhoodGenerator
from lore_sa.util import neuclidean, sigmoid
from scipy.spatial.distance import cdist, hamming
from deap import base, creator, tools, algorithms
import pickle
import numpy as np
import random
import numbers


__all__ = ["NeighborhoodGenerator","GeneticGenerator"]
class GeneticGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, encdec = None, verbose=False):
        super(GeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                               nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                               numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.verbose = verbose
        random.seed(random_seed)

    def generate(self, x, num_samples=1000):
        if self.encdec is not None:
            x = x.flatten()
        num_samples_eq = int(np.round(num_samples * 0.5))
        num_samples_noteq = int(np.round(num_samples * 0.5))
        toolbox_eq = self.setup_toolbox(x, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        toolbox_noteq = self.setup_toolbox(x, self.fitness_notequal, num_samples_noteq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_noteq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        Z = super(GeneticGenerator, self).balance_neigh(x, Z, num_samples)
        Z = np.nan_to_num(Z)
        Z[0] = x.copy()
        return Z

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        if len(sorted_array) == 0:
            fitness_value_thr = -np.inf
        else:
            index = np.max(sorted_array)
            fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            # if p.fitness.wvalues[0] > fitness_value_thr:
            Z.append(p)

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)

        return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)
        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        return toolbox

    def setup_toolbox_noteq(self, x, x1, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x1)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def fit(self, toolbox, population_size):
        halloffame_size = int(np.round(population_size * self.halloffame_ratio))
        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=self.verbose)

        return population, halloffame, logbook

    def record_init(self, x):
        return x

    def random_init(self):
        z = self.generate_synthetic_instance()
        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        z = toolbox.clone(x)
        # for i in range(self.nbr_features):
        #         #     if np.random.random() <= self.mutpb:
        #         #         z[i] = np.random.choice(self.feature_values[i], size=1, replace=True)
        z = self.generate_synthetic_instance(from_z=z, mutpb=self.mutpb)
        return z,

    def fitness_equal(self, x, x1):
        if isinstance(self.metric, numbers.Number):
            self.metric = neuclidean
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict(x.reshape(1, -1))[0]
        #y1 = self.bb_predict(x1.reshape(1, -1))[0]
        y = self.apply_bb_predict(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict(x1.reshape(1, -1))[0]


        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score >= self.eta2 else 0.0
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict(x.reshape(1, -1))[0]
        #y1 = self.bb_predict(x1.reshape(1, -1))[0]

        y = self.apply_bb_predict(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict(x1.reshape(1, -1))[0]

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score < self.eta2 else 0.0
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,