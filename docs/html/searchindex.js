Search.setIndex({"docnames": ["examples/tabular_explanations_example", "index", "source/generated/lore_sa.bbox.AbstractBBox", "source/generated/lore_sa.dataset.DataSet", "source/generated/lore_sa.dataset.utils", "source/generated/lore_sa.decision_tree.is_leaf", "source/generated/lore_sa.decision_tree.learn_local_decision_tree", "source/generated/lore_sa.decision_tree.prune_duplicate_leaves", "source/generated/lore_sa.decision_tree.prune_index", "source/generated/lore_sa.discretizer.Discretizer", "source/generated/lore_sa.discretizer.RMEPDiscretizer", "source/generated/lore_sa.encoder_decoder.EncDec", "source/generated/lore_sa.encoder_decoder.MyTargetEnc", "source/generated/lore_sa.encoder_decoder.OneHotEnc", "source/generated/lore_sa.explanation.Explanation", "source/generated/lore_sa.explanation.ExplanationEncoder", "source/generated/lore_sa.explanation.ImageExplanation", "source/generated/lore_sa.explanation.MultilabelExplanation", "source/generated/lore_sa.explanation.TextExplanation", "source/generated/lore_sa.explanation.json2explanation", "source/generated/lore_sa.lorem.LOREM", "source/generated/lore_sa.neighgen.CFSGenerator", "source/generated/lore_sa.neighgen.ClosestInstancesGenerator", "source/generated/lore_sa.neighgen.CounterGenerator", "source/generated/lore_sa.neighgen.GeneticGenerator", "source/generated/lore_sa.neighgen.GeneticProbaGenerator", "source/generated/lore_sa.neighgen.NeighborhoodGenerator", "source/generated/lore_sa.neighgen.RandomGenerator", "source/generated/lore_sa.neighgen.RandomGeneticGenerator", "source/generated/lore_sa.neighgen.RandomGeneticProbaGenerator", "source/generated/lore_sa.rule.Condition", "source/generated/lore_sa.rule.Rule", "source/generated/lore_sa.surrogate.DecTree", "source/generated/lore_sa.surrogate.SuperTree", "source/generated/lore_sa.surrogate.Surrogate", "source/generated/lore_sa.util", "source/modules"], "filenames": ["examples\\tabular_explanations_example.rst", "index.rst", "source\\generated\\lore_sa.bbox.AbstractBBox.rst", "source\\generated\\lore_sa.dataset.DataSet.rst", "source\\generated\\lore_sa.dataset.utils.rst", "source\\generated\\lore_sa.decision_tree.is_leaf.rst", "source\\generated\\lore_sa.decision_tree.learn_local_decision_tree.rst", "source\\generated\\lore_sa.decision_tree.prune_duplicate_leaves.rst", "source\\generated\\lore_sa.decision_tree.prune_index.rst", "source\\generated\\lore_sa.discretizer.Discretizer.rst", "source\\generated\\lore_sa.discretizer.RMEPDiscretizer.rst", "source\\generated\\lore_sa.encoder_decoder.EncDec.rst", "source\\generated\\lore_sa.encoder_decoder.MyTargetEnc.rst", "source\\generated\\lore_sa.encoder_decoder.OneHotEnc.rst", "source\\generated\\lore_sa.explanation.Explanation.rst", "source\\generated\\lore_sa.explanation.ExplanationEncoder.rst", "source\\generated\\lore_sa.explanation.ImageExplanation.rst", "source\\generated\\lore_sa.explanation.MultilabelExplanation.rst", "source\\generated\\lore_sa.explanation.TextExplanation.rst", "source\\generated\\lore_sa.explanation.json2explanation.rst", "source\\generated\\lore_sa.lorem.LOREM.rst", "source\\generated\\lore_sa.neighgen.CFSGenerator.rst", "source\\generated\\lore_sa.neighgen.ClosestInstancesGenerator.rst", "source\\generated\\lore_sa.neighgen.CounterGenerator.rst", "source\\generated\\lore_sa.neighgen.GeneticGenerator.rst", "source\\generated\\lore_sa.neighgen.GeneticProbaGenerator.rst", "source\\generated\\lore_sa.neighgen.NeighborhoodGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGeneticGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGeneticProbaGenerator.rst", "source\\generated\\lore_sa.rule.Condition.rst", "source\\generated\\lore_sa.rule.Rule.rst", "source\\generated\\lore_sa.surrogate.DecTree.rst", "source\\generated\\lore_sa.surrogate.SuperTree.rst", "source\\generated\\lore_sa.surrogate.Surrogate.rst", "source\\generated\\lore_sa.util.rst", "source\\modules.rst"], "titles": ["Tabular explanations example", "lore_sa", "lore_sa.bbox.AbstractBBox", "lore_sa.dataset.DataSet", "lore_sa.dataset.utils", "lore_sa.decision_tree.is_leaf", "lore_sa.decision_tree.learn_local_decision_tree", "lore_sa.decision_tree.prune_duplicate_leaves", "lore_sa.decision_tree.prune_index", "lore_sa.discretizer.Discretizer", "lore_sa.discretizer.RMEPDiscretizer", "lore_sa.encoder_decoder.EncDec", "lore_sa.encoder_decoder.MyTargetEnc", "lore_sa.encoder_decoder.OneHotEnc", "lore_sa.explanation.Explanation", "lore_sa.explanation.ExplanationEncoder", "lore_sa.explanation.ImageExplanation", "lore_sa.explanation.MultilabelExplanation", "lore_sa.explanation.TextExplanation", "lore_sa.explanation.json2explanation", "lore_sa.lorem.LOREM", "lore_sa.neighgen.CFSGenerator", "lore_sa.neighgen.ClosestInstancesGenerator", "lore_sa.neighgen.CounterGenerator", "lore_sa.neighgen.GeneticGenerator", "lore_sa.neighgen.GeneticProbaGenerator", "lore_sa.neighgen.NeighborhoodGenerator", "lore_sa.neighgen.RandomGenerator", "lore_sa.neighgen.RandomGeneticGenerator", "lore_sa.neighgen.RandomGeneticProbaGenerator", "lore_sa.rule.Condition", "lore_sa.rule.Rule", "lore_sa.surrogate.DecTree", "lore_sa.surrogate.SuperTree", "lore_sa.surrogate.Surrogate", "lore_sa.util", "Modules"], "terms": {"import": [0, 15], "panda": 0, "pd": 0, "numpi": [0, 20], "np": 0, "from": [0, 3, 4, 8, 12, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29], "sklearn": [0, 2], "preprocess": [0, 32, 33, 34], "ensembl": 0, "randomforestclassifi": 0, "model_select": 0, "train_test_split": 0, "linear_model": 0, "logisticregress": 0, "xailib": 0, "data_load": 0, "dataframe_load": 0, "prepare_datafram": [0, 3], "lime_explain": 0, "limexaitabularexplain": 0, "lore_explain": 0, "loretabularexplain": 0, "shap_explainer_tab": 0, "shapxaitabularexplain": 0, "sklearn_classifier_wrapp": 0, "we": [0, 8, 23], "start": [0, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29], "read": [0, 3], "csv": [0, 3, 4], "file": [0, 3], "analyz": 0, "The": [0, 3, 15], "tabl": 0, "i": [0, 5, 11, 12, 13, 15, 20, 23, 35], "mean": [0, 23], "datafram": [0, 3], "class": [0, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "librari": 0, "among": 0, "all": [0, 15], "attribut": [0, 15], "select": [0, 13], "class_field": 0, "column": 0, "contain": [0, 15], "observ": 0, "correspond": 0, "row": 0, "source_fil": 0, "german_credit": 0, "default": [0, 15, 20], "transform": [0, 35], "df": 0, "read_csv": 0, "skipinitialspac": 0, "true": [0, 15, 20, 21, 22, 23, 30], "na_valu": 0, "keep_default_na": 0, "after": 0, "memori": 0, "need": [0, 20], "extract": [0, 3, 23], "metadata": 0, "inform": [0, 3, 15], "automat": 0, "handl": [0, 3, 11], "content": 0, "withint": 0, "method": [0, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "scan": [0, 3], "follow": [0, 3], "trasform": 0, "version": [0, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29], "origin": [0, 21], "where": 0, "discret": [0, 20], "ar": [0, 11, 15, 21], "numer": 0, "us": [0, 8, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29], "one": 0, "hot": 0, "encod": [0, 3, 11, 12, 13, 15], "strategi": 0, "feature_nam": 0, "list": [0, 2, 15, 20], "containint": 0, "name": [0, 12, 20], "featur": [0, 23], "class_valu": [0, 6], "possibl": 0, "valu": [0, 3, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35], "numeric_column": 0, "e": [0, 23], "continu": 0, "rdf": 0, "befor": 0, "real_feature_nam": 0, "features_map": [0, 21, 22, 23, 24, 25, 26, 27, 28, 29], "dictionari": [0, 15, 20], "point": [0, 21], "each": [0, 11, 15, 21], "train": [0, 4, 12], "rf": 0, "classifi": 0, "split": 0, "test": [0, 2, 15], "subset": 0, "test_siz": 0, "0": [0, 8, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35], "3": [0, 20, 23, 24, 25, 28, 29], "random_st": [0, 20], "42": 0, "x_train": 0, "x_test": 0, "y_train": 0, "y_test": 0, "stratifi": 0, "Then": 0, "set": 0, "onc": 0, "ha": 0, "been": 0, "wrapper": 0, "get": [0, 15], "access": 0, "xai": 0, "lib": 0, "bb": [0, 20], "n_estim": 0, "20": 0, "fit": [0, 13, 35], "bbox": 0, "new": [0, 21], "instanc": [0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "classfi": 0, "print": [0, 15], "inst": 0, "iloc": 0, "147": 0, "8": 0, "reshap": 0, "1": [0, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35], "15": 0, "975": 0, "2": [0, 23, 24, 25, 28, 29], "25": 0, "provid": [0, 2, 3, 20], "an": [0, 3, 11, 15, 20], "explant": 0, "everi": [0, 23], "take": [0, 15, 23], "input": 0, "blackbox": [0, 20], "configur": 0, "object": [0, 3, 15, 20], "initi": [0, 20], "config": [0, 20], "tree": 0, "100": [0, 20, 23, 24, 25, 28, 29], "exp": 0, "plot_features_import": 0, "neigh_typ": 0, "rndgen": 0, "size": [0, 21, 22, 23, 24, 25, 26, 27, 28, 29], "1000": [0, 21, 22, 23, 24, 25, 26, 27, 28, 29], "ocr": [0, 21, 22, 23, 24, 25, 26, 27, 28, 29], "ngen": [0, 23, 24, 25, 28, 29], "10": [0, 35], "plotrul": 0, "plotcounterfactualrul": 0, "limeexplain": 0, "feature_select": 0, "lasso_path": 0, "lime_exp": 0, "as_list": 0, "account_check_statu": 0, "check": [0, 5, 15], "account": 0, "03792512128083548": 0, "duration_in_month": 0, "03701527256562679": 0, "dm": 0, "03144299031649348": 0, "save": 0, "020051934530021572": 0, "ag": 0, "019751080001761446": 0, "credit_histori": 0, "critic": 0, "other": 0, "exist": 0, "thi": [0, 3, 8, 12, 15, 20], "bank": [0, 4], "018970043296280513": 0, "other_installment_plan": 0, "none": [0, 3, 10, 11, 12, 13, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35], "018869997928840695": 0, "017658677626390982": 0, "hous": 0, "own": 0, "014948467979451343": 0, "delai": 0, "pai": 0, "off": 0, "past": 0, "012221985897781883": 0, "plot_lime_valu": 0, "5": [0, 6, 20, 21, 22, 23, 24, 25, 28, 29, 35], "regress": [0, 15], "scaler": 0, "normal": 0, "standardscal": 0, "x_scale": 0, "c": 0, "penalti": 0, "l2": 0, "pass": [0, 2], "record": [0, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "182": 0, "27797454": 0, "35504085": 0, "94540357": 0, "07634233": 0, "04854891": 0, "72456474": 0, "43411405": 0, "65027399": 0, "61477862": 0, "25898489": 0, "80681063": 0, "4": [0, 21], "17385345": 0, "6435382": 0, "32533856": 0, "03489416": 0, "20412415": 0, "22941573": 0, "33068147": 0, "75885396": 0, "34899122": 0, "60155441": 0, "15294382": 0, "09298136": 0, "46852129": 0, "12038585": 0, "08481889": 0, "23623492": 0, "21387736": 0, "36174054": 0, "24943031": 0, "15526362": 0, "59715086": 0, "45485883": 0, "73610476": 0, "43875307": 0, "23307441": 0, "65242771": 0, "23958675": 0, "90192655": 0, "72581563": 0, "2259448": 0, "15238005": 0, "54212562": 0, "70181003": 0, "63024248": 0, "30354212": 0, "40586384": 0, "49329429": 0, "88675135": 0, "59227935": 0, "46170508": 0, "46388049": 0, "33747696": 0, "13206764": 0, "same": 0, "previou": 0, "In": 0, "case": 0, "few": 0, "adjust": 0, "necessari": 0, "For": [0, 15, 23], "specif": [0, 15], "linear": 0, "feature_pert": 0, "intervent": 0, "shapxaitabularexplan": 0, "0x12a72dac8": 0, "geneticp": 0, "loretabularexplan": 0, "0x12bc41a90": 0, "why": 0, "becaus": 0, "condit": 0, "happen": 0, "726173400878906credit": 0, "amount": 0, "439": 0, "6443485021591purpos": 0, "retrain": 0, "11524588242173195durat": 0, "month": 0, "9407005310058594purpos": 0, "furnitur": 0, "equip": 0, "18370826542377472foreign": 0, "worker": 0, "7168410122394562purpos": 0, "domest": 0, "applianc": 0, "015466570854187save": 0, "7176859378814697purpos": 0, "vacat": 0, "doe": 0, "4622504562139511credit": 0, "histori": 0, "9085964262485504": 0, "It": [0, 11, 15], "would": [0, 15], "have": [0, 21], "hold": 0, "6443485021591": 0, "26": 0, "468921303749084durat": 0, "795059680938721instal": 0, "incom": [0, 15], "perc": 0, "603440999984741": 0, "other_debtor": 0, "co": 0, "applic": 0, "3046177878918616e": 0, "09": 0, "paid": 0, "back": 0, "duli": 0, "0114574629252053e": 0, "present_emp_sinc": 0, "unemploi": 0, "87554096296626e": 0, "7": 0, "43754044231906e": 0, "free": 0, "4157786564097103e": 0, "properti": 0, "unknown": 0, "275710719845092e": 0, "credit_amount": 0, "271233788564153e": 0, "job": 0, "manag": 0, "self": [0, 4, 12, 13], "emploi": [0, 20], "highli": 0, "qualifi": 0, "employe": 0, "offic": 0, "164190703926506e": 0, "8902027822084106e": 0, "604277452741881e": 0, "skill": 0, "offici": 0, "3808188198617575e": 0, "foreign_work": 0, "ye": 0, "365347360238489e": 0, "telephon": 0, "2048259721367863e": 0, "171945479826713e": 0, "1116662177987812e": 0, "credits_this_bank": 0, "9999632029038067e": 0, "till": 0, "now": 0, "9243622007776865e": 0, "people_under_mainten": 0, "902008911572941e": 0, "purpos": 0, "car": 0, "7104663723358493e": 0, "6584313433238958e": 0, "200": [0, 35], "639544710042764e": 0, "317487567892989e": 0, "unskil": 0, "resid": 0, "307761159896724e": 0, "store": 0, "2347569776391545e": 0, "1825353902253505e": 0, "year": 0, "1478921168922655e": 0, "a121": 0, "a122": 0, "6": 0, "1222769011436428e": 0, "personal_status_sex": 0, "femal": 0, "divorc": 0, "separ": [0, 3, 15], "marri": 0, "1002871894681165e": 0, "500": [0, 21], "0982251402773794e": 0, "0567984890752028e": 0, "present_res_sinc": 0, "9": 0, "869484730455045e": 0, "11": 0, "salari": 0, "assign": 0, "least": 0, "721716212812873e": 0, "327030468700815e": 0, "installment_as_income_perc": 0, "192261925231111e": 0, "real": 0, "estat": 0, "180043418264463e": 0, "974505020571898e": 0, "848004118893571e": 0, "80910843922895e": 0, "educ": 0, "803520453193465e": 0, "busi": 0, "330599059469541e": 0, "rent": 0, "975475868460632e": 0, "build": 0, "societi": 0, "agreement": 0, "life": 0, "insur": 0, "826524390749874e": 0, "guarantor": 0, "385760952840171e": 0, "338094381227495e": 0, "689756440260244e": 0, "582965568284186e": 0, "non": [0, 15], "473736018584135e": 0, "230002403518189e": 0, "974714318917145e": 0, "radio": 0, "televis": 0, "909852887925919e": 0, "620862803354922e": 0, "582941358078461e": 0, "501318386790144e": 0, "male": 0, "widow": 0, "500125372750834e": 0, "regist": 0, "under": 0, "custom": [0, 15], "495252929908006e": 0, "repair": 0, "2177896575440796e": 0, "0557757647139625e": 0, "627184253632623e": 0, "singl": [0, 20], "9862189862658355e": 0, "taken": 0, "8131802175589855e": 0, "9548368945624186e": 0, "modul": 1, "exampl": [1, 15], "tabular": [1, 3], "explan": [1, 20], "index": [1, 5, 8], "search": 1, "page": 1, "sourc": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "gener": [2, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34], "black": [2, 20], "box": [2, 20], "witch": 2, "two": 2, "like": 2, "__init__": [2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "abstract": [2, 26], "predict": 2, "sample_matrix": 2, "wrap": 2, "label": 2, "data": [2, 3, 15, 20, 21, 35], "paramet": [2, 3, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35], "arrai": [2, 3, 15, 20, 21], "spars": 2, "matrix": 2, "shape": [2, 21, 35], "n_queri": 2, "n_featur": 2, "sampl": [2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "return": [2, 3, 4, 12, 13, 15, 20, 21, 23, 35], "ndarrai": 2, "n_class": 2, "n_output": 2, "predict_proba": 2, "probabl": [2, 21], "estim": 2, "filenam": [3, 4], "str": [3, 15], "class_nam": [11, 12, 13, 20, 31], "interfac": 3, "imag": 3, "etc": 3, "incapsul": [3, 20], "expos": 3, "prepar": 3, "prepare_dataset": 3, "encdec": [3, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "option": [3, 20, 26], "inner_tre": [5, 8], "whether": 5, "node": [5, 8], "leaf": 5, "z": 6, "yb": 6, "weight": 6, "multi_label": [6, 20], "fals": [6, 15, 20, 21, 22, 23, 24, 25, 28, 29], "one_vs_rest": 6, "cv": 6, "prune_tre": [6, 20], "dt": 7, "remov": 7, "leav": [7, 8], "both": 7, "decis": 8, "prune": 8, "bottom": 8, "top": 8, "might": 8, "miss": 8, "becom": 8, "dure": [8, 15, 20], "do": 8, "directli": 8, "prune_duplicate_leav": 8, "instead": 8, "to_discret": 10, "proto_fn": 10, "dataset": [11, 12, 13, 20], "implement": 11, "decod": [11, 15], "differ": 11, "which": [11, 15, 20], "must": 11, "function": [4, 11, 12, 15, 20, 22, 23, 24, 25, 28, 29, 35], "enc": 11, "dec": 11, "enc_fit_transform": [11, 12, 13], "idea": 11, "user": 11, "send": 11, "complet": 11, "here": [11, 23], "onli": [11, 15], "categor": [11, 12, 13], "variabl": [11, 12, 13], "extend": 12, "targetencod": 12, "category_encod": 12, "given": [12, 15], "appli": [12, 13], "target": 12, "dataset_enc": [12, 13], "kwarg": [12, 13, 20, 21], "onehot": 13, "them": 13, "alreadi": [13, 23], "skipkei": 15, "ensure_ascii": 15, "check_circular": 15, "allow_nan": 15, "sort_kei": 15, "indent": 15, "special": 15, "json": 15, "rule": [15, 20], "type": [15, 21], "constructor": 15, "jsonencod": 15, "sensibl": 15, "If": 15, "typeerror": 15, "attempt": 15, "kei": 15, "int": [15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "float": [15, 21], "item": 15, "simpli": 15, "skip": 15, "output": 15, "guarante": 15, "ascii": 15, "charact": 15, "escap": 15, "can": 15, "dict": [3, 15], "circular": 15, "refer": 15, "prevent": 15, "infinit": 15, "recurs": 15, "caus": 15, "overflowerror": 15, "otherwis": 15, "place": 15, "nan": 15, "infin": 15, "behavior": 15, "compliant": 15, "consist": 15, "most": 15, "javascript": 15, "base": [15, 20, 23], "valueerror": 15, "sort": 15, "ensur": 15, "serial": 15, "compar": 15, "dai": 15, "basi": 15, "neg": 15, "integ": 15, "element": 15, "member": 15, "pretti": 15, "level": 15, "insert": 15, "newlin": 15, "compact": 15, "represent": 15, "specifi": 15, "should": 15, "item_separ": 15, "key_separ": 15, "tupl": 15, "To": 15, "you": 15, "elimin": 15, "whitespac": 15, "call": 15, "t": 15, "rais": 15, "obj": [15, 19], "report": 15, "about": 15, "objgect": 15, "param": [3, 4, 15], "o": 15, "string": [15, 20], "python": 15, "structur": 15, "foo": 15, "bar": 15, "baz": 15, "iterencod": 15, "_one_shot": 15, "yield": 15, "avail": 15, "chunk": 15, "bigobject": 15, "mysocket": 15, "write": 15, "img": 16, "segment": 16, "text": 18, "indexed_text": 18, "abstractbbox": 20, "neigh_gen": 20, "neighborhoodgener": 20, "surrog": 20, "k_transform": 20, "filter_crul": 20, "kernel_width": 20, "kernel": 20, "binari": 20, "extreme_fidel": 20, "bool": 20, "constraint": 20, "verbos": [20, 21, 22, 23, 24, 25, 28, 29], "local": 20, "datamanag": 20, "explain_instance_st": 20, "x": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35], "use_weight": 20, "metric": [20, 23, 24, 25, 28, 29], "neuclidean": [20, 22, 23, 24, 25, 28, 29], "run": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "exemplar_num": 20, "n_job": 20, "explain": 20, "stabl": 20, "number": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "neighbourhood": [20, 23], "measur": 20, "distanc": 20, "between": 20, "time": 20, "done": 20, "examplar": 20, "retriev": 20, "add": 20, "cf": 20, "bb_predict": [21, 22, 23, 24, 25, 26, 27, 28, 29], "feature_valu": [21, 22, 23, 24, 25, 26, 27, 28, 29], "nbr_featur": [21, 22, 23, 24, 25, 26, 27, 28, 29], "nbr_real_featur": [21, 22, 23, 24, 25, 26, 27, 28, 29], "numeric_columns_index": [21, 22, 23, 24, 25, 26, 27, 28, 29], "n_search": 21, "10000": 21, "n_batch": 21, "lower_threshold": 21, "upper_threshold": 21, "kind": [21, 32, 33, 34], "gaussian_match": 21, "sampling_kind": 21, "stopping_ratio": 21, "01": 21, "check_upper_threshold": 21, "final_counterfactual_search": 21, "custom_sampling_threshold": 21, "custom_closest_counterfactu": 21, "n": 21, "balanc": 21, "cut_radiu": 21, "forced_balance_ratio": 21, "downward_onli": 21, "num_sampl": [21, 22, 23, 24, 25, 26, 27, 28, 29], "synthet": [21, 22, 23, 24, 25, 26, 27, 28, 29], "orgin": [21, 22, 23, 24, 25, 26, 27, 28, 29], "ani": [21, 22, 23, 24, 25, 26, 27, 28, 29], "seed": [21, 22, 23, 24, 25, 26, 27, 28, 29], "neighborhood": [21, 22, 23, 24, 25, 26, 27, 28, 29], "multi_gener": [21, 22, 23, 24, 25, 26, 27, 28, 29], "multi": [21, 22, 23, 24, 25, 26, 27, 28, 29], "thread": [21, 22, 23, 24, 25, 26, 27, 28, 29], "translat": 21, "center": 21, "axi": 21, "uniform_sphere_origin": 21, "d": 21, "r": 21, "num_point": 21, "random": 21, "dimens": 21, "uniform": 21, "over": 21, "unit": 21, "ball": 21, "scale": 21, "radiu": 21, "length": 21, "rang": 21, "dimension": 21, "sphere": 21, "k": [22, 35], "rk": 22, "core_neigh_typ": 22, "unifi": 22, "alphaf": 22, "alphal": 22, "metric_featur": 22, "metric_label": 22, "ham": 22, "categorical_use_prob": 22, "continuous_fun_estim": 22, "bb_predict_proba": [23, 25, 26, 29], "original_data": [23, 26], "alpha1": [23, 24, 25, 28, 29], "alpha2": [23, 24, 25, 28, 29], "mutpb": [23, 24, 25, 28, 29], "random_se": [23, 24, 25, 28, 29], "cxpb": [23, 24, 25, 28, 29], "tournsiz": [23, 24, 25, 28, 29], "halloffame_ratio": [23, 24, 25, 28, 29], "closest": 23, "max_count": 23, "counterfactu": 23, "code": 23, "henc": 23, "latent": 23, "space": 23, "create_bin": 23, "defin": 23, "bin": [23, 35], "feature_bin": 23, "find_closest_count": 23, "counter_list": 23, "inserisco": 23, "un": 23, "per": 23, "ogni": 23, "combinazion": 23, "di": 23, "il": 23, "maggiorment": 23, "vicino": 23, "clost": 23, "ho": 23, "con": [23, 31], "le": 23, "che": 23, "sono": 23, "state": 23, "cambiat": 23, "quando": 23, "trovato": 23, "cerco": 23, "piu": 23, "generando": 23, "caso": 23, "tra": 23, "l": [23, 35], "original": 23, "nuovo": 23, "att": 30, "op": 30, "thr": 30, "is_continu": 30, "premis": 31, "best_fit_distribut": 35, "ax": 35, "model": [4, 35], "find": 35, "best": 35, "distribut": 35, "sigmoid": 35, "x0": 35, "A": 35, "logist": 35, "curv": 35, "common": 35, "": 35, "midpoint": 35, "maximum": 35, "steep": 35, "prepare_bank_dataset": 4, "http": 4, "www": 4, "kaggl": 4, "com": 4, "aniruddhachoudhuri": 4, "credit": 4, "risk": 4, "home": 4, "riccardo": 4, "scaricati": 4, "classmethod": 3, "from_csv": 3, "comma": 3, "from_dict": 3, "seri": 3}, "objects": {"lore_sa.bbox": [[2, 0, 1, "", "AbstractBBox"]], "lore_sa.bbox.AbstractBBox": [[2, 1, 1, "", "__init__"], [2, 1, 1, "", "predict"], [2, 1, 1, "", "predict_proba"]], "lore_sa.dataset": [[3, 0, 1, "", "DataSet"], [4, 2, 0, "-", "utils"]], "lore_sa.dataset.DataSet": [[3, 1, 1, "", "__init__"], [3, 1, 1, "", "from_csv"], [3, 1, 1, "", "from_dict"], [3, 1, 1, "", "prepare_dataset"]], "lore_sa.dataset.utils": [[4, 3, 1, "", "prepare_bank_dataset"]], "lore_sa.decision_tree": [[5, 3, 1, "", "is_leaf"], [6, 3, 1, "", "learn_local_decision_tree"], [7, 3, 1, "", "prune_duplicate_leaves"], [8, 3, 1, "", "prune_index"]], "lore_sa.discretizer": [[9, 0, 1, "", "Discretizer"], [10, 0, 1, "", "RMEPDiscretizer"]], "lore_sa.discretizer.Discretizer": [[9, 1, 1, "", "__init__"]], "lore_sa.discretizer.RMEPDiscretizer": [[10, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder": [[11, 0, 1, "", "EncDec"], [12, 0, 1, "", "MyTargetEnc"], [13, 0, 1, "", "OneHotEnc"]], "lore_sa.encoder_decoder.EncDec": [[11, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder.MyTargetEnc": [[12, 1, 1, "", "__init__"], [12, 1, 1, "", "enc_fit_transform"]], "lore_sa.encoder_decoder.OneHotEnc": [[13, 1, 1, "", "__init__"], [13, 1, 1, "", "enc_fit_transform"]], "lore_sa.explanation": [[14, 0, 1, "", "Explanation"], [15, 0, 1, "", "ExplanationEncoder"], [16, 0, 1, "", "ImageExplanation"], [17, 0, 1, "", "MultilabelExplanation"], [18, 0, 1, "", "TextExplanation"], [19, 3, 1, "", "json2explanation"]], "lore_sa.explanation.Explanation": [[14, 1, 1, "", "__init__"]], "lore_sa.explanation.ExplanationEncoder": [[15, 1, 1, "", "__init__"], [15, 1, 1, "", "default"], [15, 1, 1, "", "encode"], [15, 1, 1, "", "iterencode"]], "lore_sa.explanation.ImageExplanation": [[16, 1, 1, "", "__init__"]], "lore_sa.explanation.MultilabelExplanation": [[17, 1, 1, "", "__init__"]], "lore_sa.explanation.TextExplanation": [[18, 1, 1, "", "__init__"]], "lore_sa.lorem": [[20, 0, 1, "", "LOREM"]], "lore_sa.lorem.LOREM": [[20, 1, 1, "", "__init__"], [20, 1, 1, "", "explain_instance_stable"]], "lore_sa.neighgen": [[21, 0, 1, "", "CFSGenerator"], [22, 0, 1, "", "ClosestInstancesGenerator"], [23, 0, 1, "", "CounterGenerator"], [24, 0, 1, "", "GeneticGenerator"], [25, 0, 1, "", "GeneticProbaGenerator"], [26, 0, 1, "", "NeighborhoodGenerator"], [27, 0, 1, "", "RandomGenerator"], [28, 0, 1, "", "RandomGeneticGenerator"], [29, 0, 1, "", "RandomGeneticProbaGenerator"]], "lore_sa.neighgen.CFSGenerator": [[21, 1, 1, "", "__init__"], [21, 1, 1, "", "generate"], [21, 1, 1, "", "multi_generate"], [21, 1, 1, "", "translate"], [21, 1, 1, "", "uniform_sphere_origin"]], "lore_sa.neighgen.ClosestInstancesGenerator": [[22, 1, 1, "", "__init__"], [22, 1, 1, "", "generate"], [22, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.CounterGenerator": [[23, 1, 1, "", "__init__"], [23, 1, 1, "", "create_bins"], [23, 1, 1, "", "find_closest_counter"], [23, 1, 1, "", "generate"], [23, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.GeneticGenerator": [[24, 1, 1, "", "__init__"], [24, 1, 1, "", "generate"], [24, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.GeneticProbaGenerator": [[25, 1, 1, "", "__init__"], [25, 1, 1, "", "generate"], [25, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.NeighborhoodGenerator": [[26, 1, 1, "", "__init__"], [26, 1, 1, "", "generate"], [26, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.RandomGenerator": [[27, 1, 1, "", "__init__"], [27, 1, 1, "", "generate"], [27, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.RandomGeneticGenerator": [[28, 1, 1, "", "__init__"], [28, 1, 1, "", "generate"], [28, 1, 1, "", "multi_generate"]], "lore_sa.neighgen.RandomGeneticProbaGenerator": [[29, 1, 1, "", "__init__"], [29, 1, 1, "", "generate"], [29, 1, 1, "", "multi_generate"]], "lore_sa.rule": [[30, 0, 1, "", "Condition"], [31, 0, 1, "", "Rule"]], "lore_sa.rule.Condition": [[30, 1, 1, "", "__init__"]], "lore_sa.rule.Rule": [[31, 1, 1, "", "__init__"]], "lore_sa.surrogate": [[32, 0, 1, "", "DecTree"], [33, 0, 1, "", "SuperTree"], [34, 0, 1, "", "Surrogate"]], "lore_sa.surrogate.DecTree": [[32, 1, 1, "", "__init__"]], "lore_sa.surrogate.SuperTree": [[33, 1, 1, "", "__init__"]], "lore_sa.surrogate.Surrogate": [[34, 1, 1, "", "__init__"]], "lore_sa": [[35, 2, 0, "-", "util"]], "lore_sa.util": [[35, 3, 1, "", "best_fit_distribution"], [35, 3, 1, "", "sigmoid"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:module", "3": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "module", "Python module"], "3": ["py", "function", "Python function"]}, "titleterms": {"tabular": 0, "explan": [0, 14, 15, 16, 17, 18, 19, 36], "exampl": 0, "learn": 0, "explain": 0, "german": 0, "credit": 0, "dataset": [0, 3, 4, 36], "load": 0, "prepar": 0, "data": 0, "random": 0, "forest": 0, "classfier": 0, "predict": 0, "shap": 0, "lore": 0, "lime": 0, "differ": 0, "model": 0, "logist": 0, "regressor": 0, "lore_sa": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], "indic": 1, "tabl": 1, "bbox": [2, 36], "abstractbbox": 2, "decision_tre": [5, 6, 7, 8, 36], "is_leaf": 5, "learn_local_decision_tre": 6, "prune_duplicate_leav": 7, "prune_index": 8, "discret": [9, 10, 36], "rmepdiscret": 10, "encoder_decod": [11, 12, 13, 36], "encdec": 11, "mytargetenc": 12, "onehotenc": 13, "explanationencod": 15, "imageexplan": 16, "multilabelexplan": 17, "textexplan": 18, "json2explan": 19, "lorem": [20, 36], "neighgen": [21, 22, 23, 24, 25, 26, 27, 28, 29, 36], "cfsgener": 21, "closestinstancesgener": 22, "countergener": 23, "geneticgener": 24, "geneticprobagener": 25, "neighborhoodgener": 26, "randomgener": 27, "randomgeneticgener": 28, "randomgeneticprobagener": 29, "rule": [30, 31, 36], "condit": 30, "surrog": [32, 33, 34, 36], "dectre": 32, "supertre": 33, "util": [4, 35, 36], "modul": 36, "class": 36, "blackbox": 36, "abstract": 36, "neighborhood": 36, "gener": 36, "decis": 36, "tree": 36, "function": 36, "encod": 36, "decod": 36}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})