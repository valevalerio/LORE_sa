Search.setIndex({"docnames": ["examples/tabular_explanations_example", "index", "source/generated/lore_sa.bbox.AbstractBBox", "source/generated/lore_sa.dataset.Dataset", "source/generated/lore_sa.dataset.TabularDataset", "source/generated/lore_sa.dataset.utils", "source/generated/lore_sa.discretizer.Discretizer", "source/generated/lore_sa.discretizer.RMEPDiscretizer", "source/generated/lore_sa.encoder_decoder.EncDec", "source/generated/lore_sa.encoder_decoder.LabelEnc", "source/generated/lore_sa.encoder_decoder.OneHotEnc", "source/generated/lore_sa.encoder_decoder.TabularEnc", "source/generated/lore_sa.explanation.Explanation", "source/generated/lore_sa.explanation.ExplanationEncoder", "source/generated/lore_sa.explanation.ImageExplanation", "source/generated/lore_sa.explanation.MultilabelExplanation", "source/generated/lore_sa.explanation.TextExplanation", "source/generated/lore_sa.explanation.json2explanation", "source/generated/lore_sa.lorem.LOREM", "source/generated/lore_sa.neighgen.RandomGenerator", "source/generated/lore_sa.rule.Expression", "source/generated/lore_sa.rule.Rule", "source/generated/lore_sa.rule.RuleEncoder", "source/generated/lore_sa.rule.json2expression", "source/generated/lore_sa.rule.json2rule", "source/generated/lore_sa.surrogate.DecisionTreeSurrogate", "source/generated/lore_sa.surrogate.Surrogate", "source/generated/lore_sa.util", "source/modules"], "filenames": ["examples\\tabular_explanations_example.rst", "index.rst", "source\\generated\\lore_sa.bbox.AbstractBBox.rst", "source\\generated\\lore_sa.dataset.Dataset.rst", "source\\generated\\lore_sa.dataset.TabularDataset.rst", "source\\generated\\lore_sa.dataset.utils.rst", "source\\generated\\lore_sa.discretizer.Discretizer.rst", "source\\generated\\lore_sa.discretizer.RMEPDiscretizer.rst", "source\\generated\\lore_sa.encoder_decoder.EncDec.rst", "source\\generated\\lore_sa.encoder_decoder.LabelEnc.rst", "source\\generated\\lore_sa.encoder_decoder.OneHotEnc.rst", "source\\generated\\lore_sa.encoder_decoder.TabularEnc.rst", "source\\generated\\lore_sa.explanation.Explanation.rst", "source\\generated\\lore_sa.explanation.ExplanationEncoder.rst", "source\\generated\\lore_sa.explanation.ImageExplanation.rst", "source\\generated\\lore_sa.explanation.MultilabelExplanation.rst", "source\\generated\\lore_sa.explanation.TextExplanation.rst", "source\\generated\\lore_sa.explanation.json2explanation.rst", "source\\generated\\lore_sa.lorem.LOREM.rst", "source\\generated\\lore_sa.neighgen.RandomGenerator.rst", "source\\generated\\lore_sa.rule.Expression.rst", "source\\generated\\lore_sa.rule.Rule.rst", "source\\generated\\lore_sa.rule.RuleEncoder.rst", "source\\generated\\lore_sa.rule.json2expression.rst", "source\\generated\\lore_sa.rule.json2rule.rst", "source\\generated\\lore_sa.surrogate.DecisionTreeSurrogate.rst", "source\\generated\\lore_sa.surrogate.Surrogate.rst", "source\\generated\\lore_sa.util.rst", "source\\modules.rst"], "titles": ["Tabular explanations example", "lore_sa", "lore_sa.bbox.AbstractBBox", "lore_sa.dataset.Dataset", "lore_sa.dataset.TabularDataset", "lore_sa.dataset.utils", "lore_sa.discretizer.Discretizer", "lore_sa.discretizer.RMEPDiscretizer", "lore_sa.encoder_decoder.EncDec", "lore_sa.encoder_decoder.LabelEnc", "lore_sa.encoder_decoder.OneHotEnc", "lore_sa.encoder_decoder.TabularEnc", "lore_sa.explanation.Explanation", "lore_sa.explanation.ExplanationEncoder", "lore_sa.explanation.ImageExplanation", "lore_sa.explanation.MultilabelExplanation", "lore_sa.explanation.TextExplanation", "lore_sa.explanation.json2explanation", "lore_sa.lorem.LOREM", "lore_sa.neighgen.RandomGenerator", "lore_sa.rule.Expression", "lore_sa.rule.Rule", "lore_sa.rule.RuleEncoder", "lore_sa.rule.json2expression", "lore_sa.rule.json2rule", "lore_sa.surrogate.DecisionTreeSurrogate", "lore_sa.surrogate.Surrogate", "lore_sa.util", "Modules"], "terms": {"import": [0, 13, 22], "panda": [0, 4], "pd": 0, "numpi": [0, 8, 9, 10, 11, 18, 25], "np": 0, "from": [0, 4, 5, 9, 10, 13, 19, 20, 22, 25], "sklearn": [0, 2, 10], "preprocess": [0, 25, 26], "ensembl": 0, "randomforestclassifi": 0, "model_select": 0, "train_test_split": 0, "linear_model": 0, "logisticregress": 0, "xailib": 0, "data_load": 0, "dataframe_load": 0, "prepare_datafram": 0, "lime_explain": 0, "limexaitabularexplain": 0, "lore_explain": 0, "loretabularexplain": 0, "shap_explainer_tab": 0, "shapxaitabularexplain": 0, "sklearn_classifier_wrapp": 0, "we": [0, 25], "start": [0, 19, 25], "read": [0, 4], "csv": [0, 4, 5], "file": [0, 4], "analyz": 0, "The": [0, 13, 19, 22, 25], "tabl": [0, 11], "i": [0, 8, 13, 18, 19, 20, 22, 25, 27], "mean": [0, 4], "datafram": [0, 4, 8], "class": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26], "librari": 0, "among": 0, "all": [0, 13, 22, 25], "attribut": [0, 13, 22], "select": 0, "class_field": 0, "column": [0, 4, 8], "contain": [0, 4, 8, 13, 19, 22], "observ": 0, "correspond": 0, "row": 0, "source_fil": 0, "german_credit": 0, "default": [0, 13, 18, 22], "transform": [0, 27], "df": [0, 4, 8], "read_csv": 0, "skipinitialspac": 0, "true": [0, 13, 18, 22, 25], "na_valu": 0, "keep_default_na": 0, "after": 0, "memori": 0, "need": [0, 18], "extract": [0, 25], "metadata": 0, "inform": [0, 4], "automat": 0, "handl": [0, 3, 4, 8], "content": 0, "withint": 0, "method": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26], "scan": 0, "follow": 0, "trasform": 0, "version": [0, 13, 22], "origin": [0, 9, 10], "where": 0, "discret": [0, 18], "ar": [0, 8, 13, 22], "numer": [0, 4, 19, 20], "us": [0, 13, 19, 20, 22, 25], "one": [0, 11], "hot": [0, 10, 11], "encod": [0, 8, 9, 10, 11, 13, 19, 22, 25], "strategi": 0, "feature_nam": [0, 27], "list": [0, 2, 8, 13, 18, 19, 21, 22, 25, 27], "containint": 0, "name": [0, 4, 8, 9, 10, 11, 18, 20, 27], "featur": [0, 4, 8, 9, 10, 11, 19, 25, 27], "class_valu": [0, 25], "possibl": [0, 25], "valu": [0, 4, 19, 20, 25, 27], "numeric_column": 0, "e": [0, 20], "continu": 0, "rdf": 0, "befor": 0, "real_feature_nam": 0, "features_map": 0, "dictionari": [0, 3, 4, 8, 9, 10, 11, 13, 18, 22, 27], "point": 0, "each": [0, 4, 8, 13, 19, 22], "train": [0, 5, 25], "rf": 0, "classifi": [0, 2], "split": 0, "test": [0, 2, 13, 22], "subset": 0, "test_siz": 0, "0": [0, 13, 22, 25, 27], "3": [0, 18], "random_st": [0, 18], "42": 0, "x_train": 0, "x_test": 0, "y_train": 0, "y_test": 0, "stratifi": 0, "Then": 0, "set": [0, 4], "onc": 0, "ha": 0, "been": 0, "wrapper": 0, "get": [0, 13, 22], "access": [0, 9, 10], "xai": 0, "lib": 0, "bb": [0, 18], "n_estim": 0, "20": 0, "fit": [0, 27], "bbox": 0, "new": [0, 8, 9, 10, 11, 19], "instanc": [0, 18, 19, 25, 27], "classfi": 0, "print": [0, 13, 22], "inst": 0, "iloc": 0, "147": 0, "8": 0, "reshap": 0, "1": [0, 2, 18, 27], "15": 0, "975": 0, "2": 0, "25": 0, "provid": [0, 2, 4, 8, 9, 10, 11, 18, 25], "an": [0, 4, 8, 9, 10, 13, 18, 19, 22, 25], "explant": 0, "everi": 0, "take": [0, 13, 22], "input": [0, 8, 9, 10, 25], "blackbox": [0, 18], "configur": 0, "object": [0, 4, 13, 18, 19, 20, 21, 22, 25], "initi": [0, 18], "config": [0, 18], "tree": [0, 25], "100": [0, 18], "exp": 0, "plot_features_import": 0, "neigh_typ": 0, "rndgen": 0, "size": 0, "1000": 0, "ocr": 0, "ngen": 0, "10": [0, 27], "plotrul": 0, "plotcounterfactualrul": 0, "limeexplain": 0, "feature_select": 0, "lasso_path": 0, "lime_exp": 0, "as_list": 0, "account_check_statu": 0, "check": [0, 13, 19, 22, 25], "account": 0, "03792512128083548": 0, "duration_in_month": 0, "03701527256562679": 0, "dm": 0, "03144299031649348": 0, "save": 0, "020051934530021572": 0, "ag": 0, "019751080001761446": 0, "credit_histori": 0, "critic": 0, "other": 0, "exist": 0, "thi": [0, 13, 18, 22, 25], "bank": [0, 5], "018970043296280513": 0, "other_installment_plan": 0, "none": [0, 4, 7, 11, 13, 18, 19, 22, 25, 26, 27], "018869997928840695": 0, "017658677626390982": 0, "hous": 0, "own": 0, "014948467979451343": 0, "delai": 0, "pai": 0, "off": 0, "past": 0, "012221985897781883": 0, "plot_lime_valu": 0, "5": [0, 18, 25, 27], "regress": [0, 13, 22], "scaler": 0, "normal": 0, "standardscal": 0, "x_scale": 0, "c": 0, "penalti": 0, "l2": 0, "pass": [0, 2, 13, 22], "record": [0, 8, 18], "182": 0, "27797454": 0, "35504085": 0, "94540357": 0, "07634233": 0, "04854891": 0, "72456474": 0, "43411405": 0, "65027399": 0, "61477862": 0, "25898489": 0, "80681063": 0, "4": 0, "17385345": 0, "6435382": 0, "32533856": 0, "03489416": 0, "20412415": 0, "22941573": 0, "33068147": 0, "75885396": 0, "34899122": 0, "60155441": 0, "15294382": 0, "09298136": 0, "46852129": 0, "12038585": 0, "08481889": 0, "23623492": 0, "21387736": 0, "36174054": 0, "24943031": 0, "15526362": 0, "59715086": 0, "45485883": 0, "73610476": 0, "43875307": 0, "23307441": 0, "65242771": 0, "23958675": 0, "90192655": 0, "72581563": 0, "2259448": 0, "15238005": 0, "54212562": 0, "70181003": 0, "63024248": 0, "30354212": 0, "40586384": 0, "49329429": 0, "88675135": 0, "59227935": 0, "46170508": 0, "46388049": 0, "33747696": 0, "13206764": 0, "same": [0, 25], "previou": 0, "In": 0, "case": 0, "few": 0, "adjust": 0, "necessari": 0, "For": [0, 13, 22], "specif": [0, 13, 22], "linear": 0, "feature_pert": 0, "intervent": 0, "shapxaitabularexplan": 0, "0x12a72dac8": 0, "geneticp": 0, "loretabularexplan": 0, "0x12bc41a90": 0, "why": 0, "becaus": 0, "condit": [0, 25], "happen": 0, "726173400878906credit": 0, "amount": 0, "439": 0, "6443485021591purpos": 0, "retrain": 0, "11524588242173195durat": 0, "month": 0, "9407005310058594purpos": 0, "furnitur": 0, "equip": 0, "18370826542377472foreign": 0, "worker": 0, "7168410122394562purpos": 0, "domest": 0, "applianc": 0, "015466570854187save": 0, "7176859378814697purpos": 0, "vacat": 0, "doe": 0, "4622504562139511credit": 0, "histori": 0, "9085964262485504": 0, "It": [0, 4, 8, 9, 10, 11, 13, 19, 20, 22], "would": [0, 13, 22], "have": 0, "hold": 0, "6443485021591": 0, "26": 0, "468921303749084durat": 0, "795059680938721instal": 0, "incom": [0, 13, 22, 25], "perc": 0, "603440999984741": 0, "other_debtor": 0, "co": 0, "applic": 0, "3046177878918616e": 0, "09": 0, "paid": 0, "back": 0, "duli": 0, "0114574629252053e": 0, "present_emp_sinc": 0, "unemploi": 0, "87554096296626e": 0, "7": 0, "43754044231906e": 0, "free": 0, "4157786564097103e": 0, "properti": 0, "unknown": 0, "275710719845092e": 0, "credit_amount": 0, "271233788564153e": 0, "job": [0, 25], "manag": 0, "self": [0, 5, 13, 22], "emploi": [0, 18], "highli": 0, "qualifi": 0, "employe": 0, "offic": 0, "164190703926506e": 0, "8902027822084106e": 0, "604277452741881e": 0, "skill": 0, "offici": 0, "3808188198617575e": 0, "foreign_work": 0, "ye": 0, "365347360238489e": 0, "telephon": 0, "2048259721367863e": 0, "171945479826713e": 0, "1116662177987812e": 0, "credits_this_bank": 0, "9999632029038067e": 0, "till": 0, "now": 0, "9243622007776865e": 0, "people_under_mainten": 0, "902008911572941e": 0, "purpos": 0, "car": 0, "7104663723358493e": 0, "6584313433238958e": 0, "200": [0, 27], "639544710042764e": 0, "317487567892989e": 0, "unskil": 0, "resid": 0, "307761159896724e": 0, "store": 0, "2347569776391545e": 0, "1825353902253505e": 0, "year": 0, "1478921168922655e": 0, "a121": 0, "a122": 0, "6": 0, "1222769011436428e": 0, "personal_status_sex": 0, "femal": 0, "divorc": 0, "separ": [0, 4, 13, 22], "marri": 0, "1002871894681165e": 0, "500": 0, "0982251402773794e": 0, "0567984890752028e": 0, "present_res_sinc": 0, "9": 0, "869484730455045e": 0, "11": 0, "salari": 0, "assign": 0, "least": 0, "721716212812873e": 0, "327030468700815e": 0, "installment_as_income_perc": 0, "192261925231111e": 0, "real": [0, 19], "estat": 0, "180043418264463e": 0, "974505020571898e": 0, "848004118893571e": 0, "80910843922895e": 0, "educ": 0, "803520453193465e": 0, "busi": 0, "330599059469541e": 0, "rent": 0, "975475868460632e": 0, "build": 0, "societi": 0, "agreement": 0, "life": 0, "insur": 0, "826524390749874e": 0, "guarantor": 0, "385760952840171e": 0, "338094381227495e": 0, "689756440260244e": 0, "582965568284186e": 0, "non": [0, 13, 22], "473736018584135e": 0, "230002403518189e": 0, "974714318917145e": 0, "radio": 0, "televis": 0, "909852887925919e": 0, "620862803354922e": 0, "582941358078461e": 0, "501318386790144e": 0, "male": 0, "widow": 0, "500125372750834e": 0, "regist": 0, "under": 0, "custom": [0, 13, 22], "495252929908006e": 0, "repair": 0, "2177896575440796e": 0, "0557757647139625e": 0, "627184253632623e": 0, "singl": [0, 18], "9862189862658355e": 0, "taken": 0, "8131802175589855e": 0, "9548368945624186e": 0, "modul": 1, "exampl": [1, 13, 22], "tabular": [1, 19], "explan": [1, 18], "index": [1, 4, 8, 9, 10, 11, 25], "search": 1, "page": 1, "sourc": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "gener": [2, 3, 8, 18, 19, 26], "black": [2, 18], "box": [2, 18], "witch": 2, "two": 2, "like": [2, 13, 22], "__init__": [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26], "model": [2, 5, 20, 27], "abstract": [2, 3, 8, 19], "predict": 2, "sample_matrix": 2, "wrap": 2, "label": [2, 9, 11, 25], "data": [2, 4, 13, 18, 19, 22, 27], "paramet": [2, 4, 9, 10, 18, 19, 20, 25, 27], "arrai": [2, 4, 8, 9, 10, 11, 13, 18, 22, 25], "spars": 2, "matrix": 2, "shape": [2, 27], "n_queri": 2, "n_featur": 2, "sampl": [2, 18, 25], "return": [2, 4, 5, 8, 9, 10, 11, 13, 18, 19, 22, 25, 27], "ndarrai": 2, "n_class": 2, "n_output": 2, "predict_proba": 2, "probabl": 2, "estim": 2, "update_descriptor": [3, 4], "creat": [3, 4, 19], "descriptor": [3, 4, 9, 10, 11, 19], "class_nam": [4, 18], "option": [4, 11, 18, 25], "str": [4, 11, 13, 18, 20, 22, 25], "interfac": [4, 9, 10], "includ": 4, "some": 4, "essenti": 4, "structur": [4, 13, 22], "semant": 4, "whole": 4, "type": [4, 13, 19, 22], "informationregard": 4, "format": 4, "min": 4, "max": 4, "std": 4, "standard": 4, "deviat": 4, "median": 4, "q1": 4, "first": [4, 11], "quartil": 4, "distribut": [4, 27], "q3": 4, "third": 4, "categor": [4, 8], "distinct_valu": 4, "distinct": 4, "value_count": 4, "element": [4, 13, 19, 22], "count": 4, "dict": [4, 9, 10, 11, 13, 19, 22, 25], "classmethod": 4, "from_csv": 4, "filenam": [4, 5], "comma": 4, "param": [4, 5, 8, 11, 20, 21, 25, 27], "from_dict": 4, "seri": 4, "get_class_valu": 4, "set_class_nam": 4, "onli": [4, 8, 13, 19, 22], "string": [4, 13, 18, 20, 22, 25], "set_target_label": 4, "target": [4, 25], "function": [5, 8, 9, 10, 13, 18, 22, 27], "prepare_bank_dataset": 5, "http": [5, 10], "www": 5, "kaggl": 5, "com": 5, "aniruddhachoudhuri": 5, "credit": 5, "risk": 5, "home": 5, "riccardo": 5, "scaricati": 5, "to_discret": 7, "proto_fn": 7, "dataset_descriptor": 8, "implement": [8, 13, 22], "decod": [8, 9, 10, 11, 13, 22], "differ": [8, 11, 25], "which": [8, 13, 18, 22], "must": 8, "enc": 8, "dec": 8, "enc_fit_transform": 8, "idea": 8, "user": 8, "send": 8, "complet": 8, "here": 8, "variabl": [8, 20], "x": [8, 9, 10, 11, 18, 19, 25, 27], "appli": [8, 9, 10], "dataset": [8, 11, 18, 19, 25], "features_to_encod": 8, "get_encoded_featur": [8, 9, 10, 11], "encond": [9, 10], "stare": [9, 10], "One": 10, "en": 10, "wikipedia": 10, "org": 10, "wiki": 10, "reli": 10, "onehotencod": [10, 19], "target_class": 11, "combin": 11, "over": 11, "skipkei": [13, 22], "fals": [13, 18, 22, 25], "ensure_ascii": [13, 22], "check_circular": [13, 22], "allow_nan": [13, 22], "sort_kei": [13, 22], "indent": [13, 22], "special": [13, 22], "json": [13, 22], "rule": [13, 18, 25], "constructor": [13, 22], "jsonencod": [13, 22], "sensibl": [13, 22], "If": [13, 22], "typeerror": [13, 22], "attempt": [13, 22], "kei": [13, 19, 22], "int": [13, 18, 19, 22, 25], "float": [13, 22], "item": [13, 22], "simpli": [13, 22], "skip": [13, 22], "output": [13, 22], "guarante": [13, 22], "ascii": [13, 22], "charact": [13, 22], "escap": [13, 22], "can": [13, 22], "circular": [13, 22], "refer": [13, 22], "dure": [13, 18, 22, 25], "prevent": [13, 22], "infinit": [13, 22], "recurs": [13, 22], "caus": [13, 22], "overflowerror": [13, 22], "otherwis": [13, 22], "place": [13, 22], "nan": [13, 22], "infin": [13, 22], "behavior": [13, 22], "compliant": [13, 22], "consist": [13, 22], "most": [13, 22], "javascript": [13, 22], "base": [13, 18, 22], "valueerror": [13, 22], "sort": [13, 22], "ensur": [13, 22], "serial": [13, 22], "compar": [13, 22], "dai": [13, 22], "basi": [13, 22], "neg": [13, 22], "integ": [13, 22, 25], "member": [13, 22], "pretti": [13, 22], "level": [13, 22], "insert": [13, 22], "newlin": [13, 22], "compact": [13, 22], "represent": [13, 20, 22], "specifi": [13, 22], "should": [13, 22], "item_separ": [13, 22], "key_separ": [13, 22], "tupl": [13, 22], "To": [13, 22], "you": [13, 22], "elimin": [13, 22], "whitespac": [13, 22], "call": [13, 22], "t": [13, 22], "rais": [13, 22], "obj": [13, 17, 22, 23, 24], "subclass": [13, 22], "serializ": [13, 22], "o": [13, 22], "support": [13, 22], "arbitrari": [13, 22], "iter": [13, 22], "could": [13, 22], "def": [13, 22], "try": [13, 22], "except": [13, 22], "els": [13, 22], "let": [13, 22], "python": [13, 22], "foo": [13, 22], "bar": [13, 22], "baz": [13, 22], "iterencod": [13, 22], "_one_shot": [13, 22], "given": [13, 22], "yield": [13, 22], "avail": [13, 22], "chunk": [13, 22], "bigobject": [13, 22], "mysocket": [13, 22], "write": [13, 22], "img": 14, "segment": 14, "text": 16, "indexed_text": 16, "tabulardataset": [18, 19, 25], "abstractbbox": 18, "encdec": [18, 19, 25], "neigh_gen": 18, "neighborhoodgener": 18, "surrog": [18, 20], "k_transform": 18, "multi_label": [18, 25], "filter_crul": [18, 25], "kernel_width": 18, "kernel": 18, "binari": 18, "extreme_fidel": 18, "bool": [18, 25], "constraint": [18, 25], "verbos": 18, "kwarg": 18, "local": 18, "incapsul": 18, "datamanag": 18, "explain_instance_st": 18, "use_weight": 18, "metric": 18, "neuclidean": 18, "run": 18, "exemplar_num": 18, "n_job": 18, "prune_tre": [18, 25], "explain": [18, 19], "stabl": 18, "number": [18, 19], "neighbourhood": 18, "measur": 18, "distanc": 18, "between": 18, "time": 18, "done": 18, "examplar": 18, "retriev": 18, "add": 18, "cf": 18, "random": 19, "neighbor": 19, "check_gener": 19, "filter_funct": 19, "check_fuct": 19, "logic": [19, 20], "requir": 19, "num_inst": 19, "detect": 19, "order": [19, 27], "eventu": 19, "rang": 19, "associ": 19, "randomli": 19, "choic": 19, "within": 19, "oper": 20, "callabl": 20, "util": 20, "defin": [20, 25], "premis": [20, 21, 25], "emit": 20, "involv": 20, "g": 20, "operator2str": 20, "convert": 20, "gt": 20, "consequ": [21, 25], "express": [21, 25], "repres": 21, "con": 21, "kind": [25, 26], "check_feasibility_of_falsified_condit": 25, "delta": 25, "unadmittible_featur": 25, "falsifield": 25, "confit": 25, "unadmitt": 25, "compact_premis": 25, "premises_list": 25, "remov": 25, "threashold": 25, "get_counterfactual_rul": 25, "y": 25, "z": 25, "features_map_inv": 25, "neighborhood": 25, "get_falsified_condit": 25, "x_dict": 25, "crule": 25, "wrong": 25, "falsifi": 25, "get_rul": 25, "promis": 25, "p": 25, "decis": 25, "90": 25, "grant": 25, "employ": 25, "is_leaf": 25, "inner_tre": 25, "whether": 25, "node": 25, "leaf": 25, "prune_duplicate_leav": 25, "dt": 25, "leav": 25, "both": 25, "prune_index": 25, "prune": 25, "bottom": 25, "top": 25, "might": 25, "miss": 25, "becom": 25, "do": 25, "directli": 25, "instead": 25, "yb": 25, "weight": 25, "one_vs_rest": 25, "cv": 25, "best_fit_distribut": 27, "bin": 27, "ax": 27, "find": 27, "best": 27, "sigmoid": 27, "x0": 27, "k": 27, "l": 27, "A": 27, "logist": 27, "curv": 27, "common": 27, "": 27, "midpoint": 27, "maximum": 27, "steep": 27, "vector2dict": 27}, "objects": {"lore_sa.bbox": [[2, 0, 1, "", "AbstractBBox"]], "lore_sa.bbox.AbstractBBox": [[2, 1, 1, "", "__init__"], [2, 1, 1, "", "model"], [2, 1, 1, "", "predict"], [2, 1, 1, "", "predict_proba"]], "lore_sa.dataset": [[3, 0, 1, "", "Dataset"], [4, 0, 1, "", "TabularDataset"], [5, 3, 0, "-", "utils"]], "lore_sa.dataset.Dataset": [[3, 1, 1, "", "__init__"], [3, 1, 1, "", "update_descriptor"]], "lore_sa.dataset.TabularDataset": [[4, 1, 1, "", "__init__"], [4, 2, 1, "", "descriptor"], [4, 2, 1, "", "df"], [4, 1, 1, "", "from_csv"], [4, 1, 1, "", "from_dict"], [4, 1, 1, "", "get_class_values"], [4, 1, 1, "", "set_class_name"], [4, 1, 1, "", "set_target_label"], [4, 1, 1, "", "update_descriptor"]], "lore_sa.dataset.utils": [[5, 4, 1, "", "prepare_bank_dataset"]], "lore_sa.discretizer": [[6, 0, 1, "", "Discretizer"], [7, 0, 1, "", "RMEPDiscretizer"]], "lore_sa.discretizer.Discretizer": [[6, 1, 1, "", "__init__"]], "lore_sa.discretizer.RMEPDiscretizer": [[7, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder": [[8, 0, 1, "", "EncDec"], [9, 0, 1, "", "LabelEnc"], [10, 0, 1, "", "OneHotEnc"], [11, 0, 1, "", "TabularEnc"]], "lore_sa.encoder_decoder.EncDec": [[8, 1, 1, "", "__init__"], [8, 1, 1, "", "encode"], [8, 1, 1, "", "get_encoded_features"]], "lore_sa.encoder_decoder.LabelEnc": [[9, 1, 1, "", "__init__"], [9, 1, 1, "", "decode"], [9, 1, 1, "", "encode"], [9, 1, 1, "", "get_encoded_features"]], "lore_sa.encoder_decoder.OneHotEnc": [[10, 1, 1, "", "__init__"], [10, 1, 1, "", "decode"], [10, 1, 1, "", "encode"], [10, 1, 1, "", "get_encoded_features"]], "lore_sa.encoder_decoder.TabularEnc": [[11, 1, 1, "", "__init__"], [11, 1, 1, "", "decode"], [11, 1, 1, "", "encode"], [11, 1, 1, "", "get_encoded_features"]], "lore_sa.explanation": [[12, 0, 1, "", "Explanation"], [13, 0, 1, "", "ExplanationEncoder"], [14, 0, 1, "", "ImageExplanation"], [15, 0, 1, "", "MultilabelExplanation"], [16, 0, 1, "", "TextExplanation"], [17, 4, 1, "", "json2explanation"]], "lore_sa.explanation.Explanation": [[12, 1, 1, "", "__init__"]], "lore_sa.explanation.ExplanationEncoder": [[13, 1, 1, "", "__init__"], [13, 1, 1, "", "default"], [13, 1, 1, "", "encode"], [13, 1, 1, "", "iterencode"]], "lore_sa.explanation.ImageExplanation": [[14, 1, 1, "", "__init__"]], "lore_sa.explanation.MultilabelExplanation": [[15, 1, 1, "", "__init__"]], "lore_sa.explanation.TextExplanation": [[16, 1, 1, "", "__init__"]], "lore_sa.lorem": [[18, 0, 1, "", "LOREM"]], "lore_sa.lorem.LOREM": [[18, 1, 1, "", "__init__"], [18, 1, 1, "", "explain_instance_stable"]], "lore_sa.neighgen": [[19, 0, 1, "", "RandomGenerator"]], "lore_sa.neighgen.RandomGenerator": [[19, 1, 1, "", "__init__"], [19, 1, 1, "", "check_generated"], [19, 1, 1, "", "generate"]], "lore_sa.rule": [[20, 0, 1, "", "Expression"], [21, 0, 1, "", "Rule"], [22, 0, 1, "", "RuleEncoder"], [23, 4, 1, "", "json2expression"], [24, 4, 1, "", "json2rule"]], "lore_sa.rule.Expression": [[20, 1, 1, "", "__init__"], [20, 1, 1, "", "operator2string"]], "lore_sa.rule.Rule": [[21, 1, 1, "", "__init__"]], "lore_sa.rule.RuleEncoder": [[22, 1, 1, "", "__init__"], [22, 1, 1, "", "default"], [22, 1, 1, "", "encode"], [22, 1, 1, "", "iterencode"]], "lore_sa.surrogate": [[25, 0, 1, "", "DecisionTreeSurrogate"], [26, 0, 1, "", "Surrogate"]], "lore_sa.surrogate.DecisionTreeSurrogate": [[25, 1, 1, "", "__init__"], [25, 1, 1, "", "check_feasibility_of_falsified_conditions"], [25, 1, 1, "", "compact_premises"], [25, 1, 1, "", "get_counterfactual_rules"], [25, 1, 1, "", "get_falsified_conditions"], [25, 1, 1, "", "get_rule"], [25, 1, 1, "", "is_leaf"], [25, 1, 1, "", "prune_duplicate_leaves"], [25, 1, 1, "", "prune_index"], [25, 1, 1, "", "train"]], "lore_sa.surrogate.Surrogate": [[26, 1, 1, "", "__init__"]], "lore_sa": [[27, 3, 0, "-", "util"]], "lore_sa.util": [[27, 4, 1, "", "best_fit_distribution"], [27, 4, 1, "", "sigmoid"], [27, 4, 1, "", "vector2dict"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:attribute", "3": "py:module", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "module", "Python module"], "4": ["py", "function", "Python function"]}, "titleterms": {"tabular": 0, "explan": [0, 12, 13, 14, 15, 16, 17, 28], "exampl": 0, "learn": 0, "explain": 0, "german": 0, "credit": 0, "dataset": [0, 3, 4, 5, 28], "load": 0, "prepar": 0, "data": 0, "random": 0, "forest": 0, "classfier": 0, "predict": 0, "shap": 0, "lore": 0, "lime": 0, "differ": 0, "model": 0, "logist": 0, "regressor": 0, "lore_sa": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "indic": 1, "tabl": 1, "bbox": [2, 28], "abstractbbox": 2, "tabulardataset": 4, "util": [5, 27, 28], "discret": [6, 7, 28], "rmepdiscret": 7, "encoder_decod": [8, 9, 10, 11, 28], "encdec": 8, "labelenc": 9, "onehotenc": 10, "tabularenc": 11, "explanationencod": 13, "imageexplan": 14, "multilabelexplan": 15, "textexplan": 16, "json2explan": 17, "lorem": [18, 28], "neighgen": [19, 28], "randomgener": 19, "rule": [20, 21, 22, 23, 24, 28], "express": 20, "ruleencod": 22, "json2express": 23, "json2rul": 24, "surrog": [25, 26, 28], "decisiontreesurrog": 25, "modul": 28, "class": 28, "blackbox": 28, "abstract": 28, "neighborhood": 28, "gener": 28, "function": 28, "encod": 28, "decod": 28}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})