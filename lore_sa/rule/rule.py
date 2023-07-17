import json
import numpy as np

from lore_sa.util import vector2dict
from typing import Callable
import operator 


__all__ = ["Rule"]

def json2cond(obj):
    return Condition(obj['att'], obj['op'], obj['thr'], obj['is_continuous'])

def json2rule(obj):
    premises = [json2cond(p) for p in obj['premise']]
    cons = obj['cons']
    class_name = obj['class_name']
    return Rule(premises, cons, class_name)


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
            #print('alla fine arriva polly, ', self.att, 'spazio ',  self.op, 'spazo ', self.thr)
            return '%s %s %.2f' % (self.att, self.op, self.thr)

    def __eq__(self, other):
        return self.att == other.att and self.op == other.op and self.thr == other.thr

    def __hash__(self):
        return hash(str(self))


class Expression(object):
    """
    Utility object to define a logical expression. It is used to define the premises of a Rule emitted from a surrogate model.
    """
    def __init__(self,variable:str, operator:Callable, value):
        """
        :param[str] variable: name of the variable that defines the rule
        :param[Callable] operator: logical operator involved in the rule
        :param value: numerical value to define the rule. E.g. variable > value 
        """

        self.variable = variable
        self.operator = operator 
        self.value= value




class Rule(object):

    def __init__(self, premises:list, cons:Expression, class_name:str):
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
