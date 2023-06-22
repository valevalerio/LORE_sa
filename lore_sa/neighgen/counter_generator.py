from lore_sa.encoder_decoder import OneHotEnc
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.util import neuclidean

from scipy.stats import binned_statistic
from lore_sa.neighgen.random_genetic_generator import RandomGeneticGenerator
import math
import sys
import numpy as np
import random
import copy

__all__ = ["NeighborhoodGenerator","CounterGenerator"]
class CounterGenerator(NeighborhoodGenerator):
    """
    Class for the generation of the neighborhood based on the extraction of counterfactuals
    here x is already coded, hence the generation of the neighbourhood is in the latent space
    """
    def __init__(self, bb_predict, bb_predict_proba, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, original_data = None, encdec=None, alpha1=0.5, alpha2=0.5,
                 metric=neuclidean, ngen=100, mutpb=0.2, random_seed = None,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, closest=True, max_counter = 1, verbose=False):
        super(CounterGenerator, self).__init__(bb_predict=bb_predict, bb_predict_proba=bb_predict_proba, feature_values=feature_values, features_map=features_map,
                                              nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                              numeric_columns_index= numeric_columns_index, ocr=ocr,
                                              original_data = original_data, encdec=encdec )
        self.closest = closest
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.max_counter = max_counter
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.random_seed = random_seed
        if original_data is None:
            raise ValueError('This method can not be applied without a sample of the original data.')

    #for every feature we define the bins and take the mean value
    def create_bins(self):
        feature_bins = dict()
        #nel caso di variabili in one hot encoding allora le variabili sono binarie
        if type(self.encdec) is OneHotEnc:
            for i in range(self.nbr_real_features):
                if i in self.numeric_columns_index:
                    bins = (max(self.original_data[i]) - min(self.original_data[i])) / float(
                        math.ceil(math.log2(len(self.original_data[i]))) + 1)
                    # create bins
                    binned = pd.cut(self.original_data[i], bins)
                    mean_bins = self.original_data[binned].mean()
                    feature_bins[i] = dict()
                    feature_bins[i]['bins'] = mean_bins.keys()
                    feature_bins[i]['avg'] = mean_bins.values()
                else:
                    # caso onehot, default
                    print('in synth neigh gen ', i)
                    feature_bins[i] = dict()
                    feature_bins[i]['bins'] = self.features_map[i].keys()
                    feature_bins[i]['avg'] = self.features_map[i].values()
        else:
            for f in range(self.original_data.shape[1]):
                #define the ideal number of bins
                bins = math.ceil((max(self.original_data[:, f]) - min(self.original_data[:, f]))/float(math.ceil(math.log2(len(self.original_data[: , f])))+1))
                #create bins
                binned = binned_statistic(self.original_data[:,f], values = self.original_data[:,f], bins=bins, statistic='mean')
                stat = binned.statistic
                correct_values = list()
                for i in binned.binnumber:
                    correct_values.append(stat[i-1])
                feature_bins[f] = dict()
                feature_bins[f]['avg'] = stat
                feature_bins[f]['bins'] = correct_values
                feature_bins[f]['edges'] = binned.bin_edges
        return feature_bins
    #todo sistemare find closest per quando si arriva da possibilties
    def find_closest_counter(self, counter_list, x):
        #in closests inserisco un record per ogni combinazione di features, il record maggiormente vicino
        #in clostes ho il record con le features che sono state cambiate
        #quando ho trovato closests, cerco il record piu vicino, generando a caso records tra l'originale e il nuovo
        closests = list()
        for feat in counter_list.keys():
            vals = list()
            element = None
            closest = sys.maxsize
            for new_record in counter_list[feat]:
                print('new record ', new_record, new_record.shape, x.shape)
                distance = 0
                for f in feat:
                    distance += abs(new_record[0][f] - x[0][f])
                if distance < closest:
                    closest = distance
                    #in element inserisco gli indici delle features che cambio ed il record nuovo che ho ottenuto
                    #in questo modo dopo vado a fare la distanza tra ogni feature cambiata e quella originale
                    element = (feat, new_record)
            closests.append(element)

            '''for v in counter_list[feat]:
                temp = list()
                for p in feat:
                    temp.append(v[:, p])
                #temp.append(v)
            vals.append(temp)
            closest = sys.maxsize
            element = None
            for v in vals:
                dists = list()
                for fea in range(0, len(v)-1):
                    distance =  distance + abs(v[feat[fea]]-x[:, feat[fea]])
                    dists.append(abs(v[feat[fea]]-x[:, feat[fea]]))
                if distance < closest:
                    closest = distance
                    element = v
            #in element ho tutti gli elementi cambiati e come ultimo il vettore record completo
            #todo check che len(element[0]-1 == element[1])
            #in feat ho gli index delle feature che sono state cambiate
            #devo estrarre da x tutti gli elementi
            closests.append((element, feat))'''
        new_c = list()
        #print('CLOSEST ', len(closests))
        for c in closests:
            print('generate closest ', c[0])
            #ciclo su tutti gli index delle features che ho cambiato
            trial = np.copy(x)
            bb_x = self.apply_bb_predict(x)
            cicla = True
            while cicla:
                for el in c[0]:
                    if c[1][0][el] < x[0][el]:
                        trial[0][el] = np.random.uniform(low=c[1][0][el], high=x[0][el], size=1)
                    else:
                        trial[0][el] = np.random.uniform(low=x[0][el], high=c[1][0][el], size=1)
                pred = self.bb_predict(trial)
                if pred != bb_x:
                    cicla = False
                #se rispetto al record di prima sto andando verso il record da spiegare
                #genero a partire da questo
                print('prima dell errore ', self.apply_bb_predict_proba(trial))
                print('prima prima ', self.apply_bb_predict_proba(c[1]))
                if self.apply_bb_predict_proba(trial)[:, bb_x] > self.apply_bb_predict_proba(c[1])[:,bb_x]:
                    new_c.append(trial)

            '''questo codice funziona solo nel caso di una sola variabile cambiata
            if c[1] > c[2]:
                randoms = np.random.uniform(low=c[1],high=c[2],size=100)
            else:
                randoms = np.random.uniform(low=c[2], high=c[1], size=100)
            randoms = np.sort(randoms)
            #arrays = [x for _ in range(100)]
            arrays = np.repeat(x, 100, axis=0)
            arrays[:,c[3]] = randoms
            pred_arrays = self.apply_bb_predict(np.array(arrays))
            if abs(randoms[0]-c[2]) > abs(randoms[-1]-c[2]):
                pred_arrays = reversed(pred_arrays)
                arrays = reversed(arrays)
            #todo from dir (0 o -1) cerca il primo che cambia valore rispetto al valore di x predetto
            print('pred arrays is ', pred_arrays, self.apply_bb_predict(x))
            #first_counter = arrays[pred_arrays.index(self.apply_bb_predict(x))]
            #print('this is the first counter found ',self.apply_bb_predict(x), first_counter, pred_arrays.index(self.apply_bb_predict(x)))
            #new_c.append(first_counter)
            ind = np.where(pred_arrays != self.apply_bb_predict(x))
            if len(ind[0]) == 0:
                break
            else:
                to_append = arrays[ind[0],:]
                print('first counter closests ', to_append[0])
                new_c.append(to_append[0])
            #print('INDICE IND ', ind)
            #print('ecco quello che viene appeso ', ind[0][0], arrays[0,:])
            #to_append = arrays[ind[0][0],:]
            #print('valore di to append ', to_append)
            #if len(to_append) == 0:
            #    new_c.append(to_append)'''
        print('ELEMENTI trovati ', len(new_c))
        return new_c

    def for_loop_counter(self, n_feat, n_iter, possibilities, x):
        counters = dict()
        for i in range(n_iter):
            feats = random.sample(list(possibilities.keys()), n_feat)
            inds = list()
            #seleziono degli indici di features a caso
            for f in range(n_feat):
                inds.append(random.choice(range(0,len(possibilities[feats[f]]))))
            to_analize = list()
            for f in range(0, len(inds)):
                to_analize.append(possibilities[feats[f]][inds[f]])
            prova = to_analize[0]
            for t in range(1, len(to_analize)):
                prova[0][inds[t]] = to_analize[t][0][inds[t]]
            if self.apply_bb_predict(prova) != self.apply_bb_predict(x):
                #todo crea una stringa con comma e poi dopo dividi per il comma
                inds_string = tuple(inds)
                if inds_string in counters.keys():
                    #trovato un controfattuale
                    counters[inds_string].append(prova)
                else:
                    counters[inds_string] = list()
                    counters[inds_string].append(prova)
        print('counters in possibilities ', len(counters), counters.keys())
        return counters

    #procedura nel caso in cui nessun controfattuale con 1 solo cambiamento sia stato trovato
    def find_in_possibilities(self, possibilities, x, n_iter):
        print('la len di possibilities ', len(possibilities.keys()))
        #possibilities dict con chiave index delle features, dentro records che avvicinano la pred a counter
        for i in range(2, len(possibilities.keys())):
            counters = self.for_loop_counter(i, 1000, possibilities, x)
            if counters:
                print('sono in break')
                break
        print('ecco il counter che ho trovato con ', len(counters), i, len(possibilities.keys()))
        if not counters:
            raise Exception(' Impossible to find a counterfactual')
        else:
            generate_on = self.find_closest_counter(counters, x)
        return generate_on

    #genero con algoritmo sedc
    #ho bisogno di predict proba
    def generate(self, x, num_samples=1000):
        counter_list = dict()
        possibilities = dict()
        counter_found = False
        try:
            pred_proba_x = self.apply_bb_predict_proba(x)
            pred_x = self.apply_bb_predict(x)
            to_explain = copy.deepcopy(x)
        except:
            raise ValueError('Predict proba has to be defined for the Counter Generation')
        feature_bins = self.create_bins()
        #ciclo sulle features, tutte quelle di feature bins
        #f index della colonna
        for f in feature_bins.keys():
            #colonna da usare adesso
            actual = feature_bins[f]
            for val in actual['avg']:
                #to_explain[:, f] = val
                temp = copy.deepcopy(to_explain)
                temp[:,f] = val
                pred_actual = self.apply_bb_predict(temp)
                if pred_actual != pred_x:
                    if f in counter_list.keys():
                        counter_list[f].append(temp)
                    else:
                        counter_list[f] = list()
                        counter_list[f].append(temp)
                        counter_found = True
                else:
                    pred_proba_actual = self.apply_bb_predict_proba(temp)
                    if pred_proba_actual[0][pred_x] < pred_proba_x[0][pred_x]:
                        if f in possibilities.keys():
                            possibilities[f].append(temp)
                        else:
                            possibilities[f] = list()
                            possibilities[f].append(temp)
        if counter_found:
            generate_on = self.find_closest_counter(counter_list, x)
        #ho finito di ciclare, adesso guardo se counter list ha qualche elemento dentro
        #se counter list vuota, allora analizzo possibilities
        elif possibilities.keys():
            generate_on = self.find_in_possibilities(possibilities, x, 20)
        else:
            raise Exception('no counterfactual found')
        # se counter list ha qualche record, uso questi come record controfattuali
        # da questi record deve partire una generazione random+genetica
        randgen = RandomGeneticGenerator(self.bb_predict, self.feature_values, self.features_map,
                                        self.nbr_features, self.nbr_real_features, self.numeric_columns_index,
                                        ocr=self.ocr, alpha1=self.alpha1, alpha2=self.alpha2,
                                        metric=self.metric, ngen=self.ngen, mutpb=self.mutpb, cxpb=self.cxpb,
                                        tournsize=self.tournsize, halloffame_ratio=self.halloffame_ratio,
                                        random_seed=self.random_seed, encdec=self.encdec, verbose=False)
        #todo aggiungi un range in cui generare
        cont = 0
        #prima genero sul dato originale
        Z = randgen.generate(x, num_samples=1000)
        #max_counter dice quanti counterfactual records considerare, scelto dall'utente
        for ind in range(0, self.max_counter):
            rec_index = random.choice([0,len(generate_on)-1])
            rec = generate_on[rec_index]
            Z_temp = (randgen.generate(rec, num_samples=500))
            Z = np.concatenate((Z, Z_temp), axis=0)
            print('finito con cont ', cont, Z.shape)
        return Z
