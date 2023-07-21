import copy
from collections import defaultdict

import numpy as np
import operator

from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import EncDec, OneHotEnc
from lore_sa.rule.rule import Rule, Expression
from lore_sa.rule.emitter import Emitter
from lore_sa.surrogate.decision_tree import DecisionTreeSurrogate
from lore_sa.util import multilabel2str, vector2dict


__all__ = ["Emitter", "DecisionTreeRuleEmitter"]

class DecisionTreeRuleEmitter(Emitter):

    def get_rule(self, x: np.array, dt: DecisionTreeSurrogate, dataset: TabularDataset, encdec: EncDec = None, multi_label: bool = False):
        """
        Extract the rules as the promises and consequences {p -> y}, starting from a Decision Tree

            >>> {( income > 90) -> grant),
                ( job = employer) -> grant)
            }

        :param [Numpy Array] x:
        :param [DecisionTreeSurrogate] dt:
        :param [TabularDataset] dataset:
        :param [EncDec] encdec:
        :param [boolean] multi_label:
        :return [Rule]: Rule objects
        """
        x = x.reshape(1, -1)
        feature = dt.tree_.feature
        threshold = dt.tree_.threshold

        leave_id = dt.apply(x)
        node_index = dt.decision_path(x).indices

        feature_names = dataset.get_features_names()
        numeric_columns = dataset.get_numeric_columns()
        class_values = dataset.get_class_values()

        premises = list()
        for node_id in node_index:
            if leave_id[0] == node_id:
                break
            else:
                if encdec is not None:
                    if isinstance(encdec, OneHotEnc):
                        attribute = feature_names[feature[node_id]]
                        if attribute not in numeric_columns:
                            thr = False if x[0][feature[node_id]] <= threshold[node_id] else True
                            op = operator.eq
                        else:
                            thr = threshold[node_id]
                            op = operator.le if x[0][feature[node_id]] <= threshold[node_id] else operator.gt
                    else:
                        raise Exception('unknown encoder instance ')
                else:
                    op = operator.le if x[0][feature[node_id]] <= threshold[node_id] else operator.gt
                    attribute = feature_names[feature[node_id]]
                    thr = threshold[node_id]
                premises.append(Expression(attribute, op, thr))

        dt_outcome = dt.predict(x)[0]
        consequences = class_values[int(dt_outcome)] if not multi_label else multilabel2str(dt_outcome, class_values)
        premises = self.compact_premises(premises)
        return Rule(premises, consequences, dataset.class_name)

    def compact_premises(self, premises_list):
        att_list = defaultdict(list)
        for premise in premises_list:
            att_list[premise.att].append(premise)

        compact_plist = list()
        for att, alist in att_list.items():
            if len(alist) > 1:
                min_thr = None
                max_thr = None
                for av in alist:
                    if av.op == operator.le:
                        max_thr = min(av.thr, max_thr) if max_thr else av.thr
                    elif av.op == operator.gt:
                        min_thr = max(av.thr, min_thr) if min_thr else av.thr

                if max_thr:
                    compact_plist.append(Expression(att, operator.le, max_thr))

                if min_thr:
                    compact_plist.append(Expression(att, operator.gt, min_thr))
            else:
                compact_plist.append(alist[0])
        return compact_plist

    def get_counterfactual_rules(self, x, y, dt: DecisionTreeSurrogate, Z, Y, dataset: TabularDataset,features_map,
                                 features_map_inv, multi_label=False, encdec: EncDec = None, filter_crules=None,
                                 constraints=None,unadmittible_features=None):

        feature_names = dataset.get_features_names()
        numeric_columns = dataset.get_numeric_columns()
        class_values = dataset.get_class_values()
        class_name = dataset.class_name

        clen = np.inf
        crule_list = list()
        delta_list = list()
        Z1 = Z[np.where(Y != y)[0]]
        xd = vector2dict(x, feature_names)
        for z in Z1:
            crule = self.get_rule(z, y, dt, dataset, encdec, multi_label)
            delta, qlen = self.get_falsified_conditions(xd, crule)
            if unadmittible_features != None:
                is_feasible = self.check_feasibility_of_falsified_conditions(delta, unadmittible_features)
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
                xc = self.apply_counterfactual(x, delta, feature_names, features_map, features_map_inv, numeric_columns)
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

    def apply_counterfactual(self, x, delta, feature_names, features_map=None, features_map_inv=None, numeric_columns=None):
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
                    gap = 1 / (10 ** (idx + 1))
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

    def get_falsified_conditions(self,xd, crule):
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

    def check_feasibility_of_falsified_conditions(self, delta, unadmittible_features):
        for p in delta:
            p_key = p.att if p.is_continuous else p.att.split('=')[0]
            if p_key in unadmittible_features:
                if unadmittible_features[p_key] is None:
                    return False
                else:
                    if unadmittible_features[p_key] == p.op:
                        return False
        return True