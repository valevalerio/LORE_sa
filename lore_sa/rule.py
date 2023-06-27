import json
from collections import defaultdict
import copy
import numpy as np

from .dataset import Dataset
from .encoder_decoder.one_hot_enc import OneHotEnc
from .encoder_decoder.enc_dec import EncDec
from .surrogate.surrogate import Surrogate
from .util import vector2dict, multilabel2str

"""
This module provides methods and classes to extract the Rule from the surrogate model and to convert rules into json
"""
def get_rule(x, y, dt: Surrogate, dataset: Dataset, encdec: EncDec = None, multi_label: bool =False):
    """
    Extract the rule.

    :param x:
    :param y:
    :param dt:
    :param dataset:
    :param encdec:
    :param multi_label:
    :return:
    """
    x = x.reshape(1, -1)
    feature = dt.get_features()
    threshold = dt.get_threshold()

    leave_id = dt.apply(x)
    node_index = dt.decision_path(x).indices
    premises = list()
    for node_id in node_index:
        if leave_id[0] == node_id:
            break
        else:
            if encdec is not None:
                if isinstance(encdec, OneHotEnc):
                    att = dataset.features_map[feature[node_id]]
                    if att not in dataset.numeric_columns:
                        thr = 'no' if x[0][feature[node_id]] <= threshold[node_id] else 'yes'
                        op = '='
                    else:
                        op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
                        thr = threshold[node_id]
                    iscont = att in dataset.numeric_columns
                else:
                    raise Exception('unknown encoder instance ')
            else:
                op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
                att = dataset.features_map[feature[node_id]]
                thr = threshold[node_id]
                iscont = att in dataset.numeric_columns
            premises.append(Condition(att, op, thr, iscont))

    dt_outcome = dt.predict(x)[0]
    cons = dataset.class_values[int(dt_outcome)] if not multi_label else multilabel2str(dt_outcome,
                                                                                        dataset.class_values)
    premises = compact_premises(premises)
    return Rule(premises, cons, dataset.class_name)

def compact_premises(plist):
    att_list = defaultdict(list)
    for p in plist:
        att_list[p.att].append(p)

    compact_plist = list()
    for att, alist in att_list.items():
        if len(alist) > 1:
            min_thr = None
            max_thr = None
            for av in alist:
                if av.op == '<=':
                    max_thr = min(av.thr, max_thr) if max_thr else av.thr
                elif av.op == '>':
                    min_thr = max(av.thr, min_thr) if min_thr else av.thr

            if max_thr:
                compact_plist.append(Condition(att, '<=', max_thr))

            if min_thr:
                compact_plist.append(Condition(att, '>', min_thr))
        else:
            compact_plist.append(alist[0])
    return compact_plist


def get_counterfactual_rules(x, y, dt: Surrogate, Z, Y, feature_names, class_name, class_values, numeric_columns, features_map,
                             features_map_inv, multi_label=False, encdec: EncDec = None, filter_crules=None, constraints=None,
                             unadmittible_features=None):
    clen = np.inf
    crule_list = list()
    delta_list = list()
    Z1 = Z[np.where(Y != y)[0]]
    xd = vector2dict(x, feature_names)
    for z in Z1:
        crule = get_rule(z, y, dt, feature_names, class_name, class_values, numeric_columns, encdec, multi_label)
        delta, qlen = get_falsified_conditions(xd, crule)
        if unadmittible_features != None:
            is_feasible = check_feasibility_of_falsified_conditions(delta, unadmittible_features)
            if not is_feasible:
                continue
        if constraints is not None:
            to_remove = list()
            for p in crule.premises:
                if p.att in constraints.keys():
                    if p.op == constraints[p.att]['op']:
                        if p.thr > constraints[p.att]['thr']:
                            break
                            # caso corretto
                        else:
                            to_remove.append()

        if filter_crules is not None:
            xc = apply_counterfactual(x, delta, feature_names, features_map, features_map_inv, numeric_columns)
            bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
            bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                       class_values)
            dt_outcomec = crule.cons

            if bb_outcomec == dt_outcomec:
                if qlen < clen:
                    clen = qlen
                    crule_list = [crule]
                    delta_list = [delta]
                elif qlen == clen:
                    if delta not in delta_list:
                        crule_list.append(crule)
                        delta_list.append(delta)
        else:
            if qlen < clen:
                clen = qlen
                crule_list = [crule]
                delta_list = [delta]
            elif qlen == clen:
                if delta not in delta_list:
                    crule_list.append(crule)
                    delta_list.append(delta)

    return crule_list, delta_list


def apply_counterfactual(x, delta, feature_names, features_map=None, features_map_inv=None, numeric_columns=None):
    xd = vector2dict(x, feature_names)
    xcd = copy.deepcopy(xd)
    for p in delta:
        if p.att in numeric_columns:
            if p.thr == int(p.thr):
                gap = 1.0
            else:
                decimals = list(str(p.thr).split('.')[1])
                for idx, e in enumerate(decimals):
                    if e != '0':
                        break
                gap = 1 / (10**(idx+1))
            if p.op == '>':
                xcd[p.att] = p.thr + gap
            else:
                xcd[p.att] = p.thr
        else:
            fn = p.att.split('=')[0]
            if p.op == '>':
                if features_map is not None:
                    fi = list(feature_names).index(p.att)
                    fi = features_map_inv[fi]
                    for fv in features_map[fi]:
                        xcd['%s=%s' % (fn, fv)] = 0.0
                xcd[p.att] = 1.0

            else:
                if features_map is not None:
                    fi = list(feature_names).index(p.att)
                    fi = features_map_inv[fi]
                    for fv in features_map[fi]:
                        xcd['%s=%s' % (fn, fv)] = 1.0
                xcd[p.att] = 0.0

    xc = np.zeros(len(xd))
    for i, fn in enumerate(feature_names):
        xc[i] = xcd[fn]

    return xc

def get_falsified_conditions(xd, crule):
    delta = list()
    nbr_falsified_conditions = 0
    for p in crule.premises:
        try:
            if p.op == '<=' and xd[p.att] > p.thr:
                delta.append(p)
                nbr_falsified_conditions += 1
            elif p.op == '>' and xd[p.att] <= p.thr:
                delta.append(p)
                nbr_falsified_conditions += 1
        except:
            print('pop', p.op, 'xd', xd, 'xd di p ', p.att, 'hthrr', p.thr)
            continue
    return delta, nbr_falsified_conditions


def check_feasibility_of_falsified_conditions(delta, unadmittible_features):
    for p in delta:
        p_key = p.att if p.is_continuous else p.att.split('=')[0]
        if p_key in unadmittible_features:
            if unadmittible_features[p_key] is None:
                return False
            else:
                if unadmittible_features[p_key] == p.op:
                    return False
    return True


class Condition(object):

    def __init__(self, att, op, thr, is_continuous=True):
        self.att = att
        self.op = op
        self.thr = thr
        self.is_continuous = is_continuous

    def __str__(self):
        if self.is_continuous:
            if type(self.thr) is tuple:
                thr = str(self.thr[0])+' '+str(self.thr[1])
                return '%s %s %s' % (self.att, self.op, thr)
            if type(self.thr) is list:
                thr = '['
                for i in self.thr:
                    thr += str(i)
                thr += ']'
                return '%s %s %s' % (self.att, self.op, thr)
            return '%s %s %.2f' % (self.att, self.op, self.thr)
        else:
            if type(self.thr) is tuple:
                thr = '['+str(self.thr[0])+';'+str(self.thr[1])+']'
                return '%s %s %s' % (self.att, self.op, thr)
            if type(self.thr) is list:
                thr = '['
                for i in self.thr:
                    thr+=i+' ; '
                return '%s %s %s' % (self.att, self.op, thr)
            #print('alla fine, ', self.att, 'spazio ',  self.op, 'spazo ', self.thr)
            return '%s %s %.2f' % (self.att, self.op, self.thr)

    def __eq__(self, other):
        return self.att == other.att and self.op == other.op and self.thr == other.thr

    def __hash__(self):
        return hash(str(self))


class Rule(object):

    def __init__(self, premises, cons, class_name):
        self.premises = premises
        self.cons = cons
        self.class_name = class_name

    def _pstr(self):
        return '{ %s }' % (', '.join([str(p) for p in self.premises]))

    def _cstr(self):
        if not isinstance(self.class_name, list):
            return '{ %s: %s }' % (self.class_name, self.cons)
        else:
            return '{ %s }' % self.cons

    def __str__(self):
        return '%s --> %s' % (self._pstr(), self._cstr())

    def __eq__(self, other):
        return self.premises == other.premises and self.cons == other.cons

    def __len__(self):
        return len(self.premises)

    def __hash__(self):
        return hash(str(self))

    def is_covered(self, x, feature_names):
        xd = vector2dict(x, feature_names)
        for p in self.premises:
            if p.op == '<=' and xd[p.att] > p.thr:
                return False
            elif p.op == '>' and xd[p.att] <= p.thr:
                return False
        return True


def json2cond(obj):
    return Condition(obj['att'], obj['op'], obj['thr'], obj['is_continuous'])


def json2rule(obj):
    premises = [json2cond(p) for p in obj['premise']]
    cons = obj['cons']
    class_name = obj['class_name']
    return Rule(premises, cons, class_name)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ConditionEncoder(json.JSONEncoder):
    """ Special json encoder for Condition types """
    def default(self, obj):
        if isinstance(obj, Condition):
            json_obj = {
                'att': obj.att,
                'op': obj.op,
                'thr': obj.thr,
                'is_continuous': obj.is_continuous,
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


class RuleEncoder(json.JSONEncoder):
    """ Special json encoder for Rule types """
    def default(self, obj):
        if isinstance(obj, Rule):
            ce = ConditionEncoder()
            json_obj = {
                'premise': [ce.default(p) for p in obj.premises],
                'cons': obj.cons,
                'class_name': obj.class_name
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)
