# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import constants
import time
import re
import operator
import requests
import os
import threading
import uuid
import sys
import meaningcloud
import pickle

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = constants.GOOGLE_CREDS
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

class NerExtractor:

    def __init__(self, logger, dicts_dir, model_dir, lp):
        self.logger = logger
        self.lp = lp
        # minimum level of similarity for grouping
        self.min_similarity = 0.85
        self.supported_cloud_lang = {'en' : 1, 'es' : 1, 'fr' : 1, 'pt' : 1, 'it' : 1, 'ca' : 1, 'da' : 1, 'sv' : 1, 'no' : 1, 'fi' : 1, 'nl' : 1}
        # languages with additional ML features
        self.lang_with_add_feature = {'de' : 1, 'es' : 1, 'et' : 1, 'eu' : 1, 'ja' : 1, 'lv' : 1, 'nl' : 1, 'pt' : 1, 'sl' : 1, 'th' : 1}
        # maximum time for ner execution
        self.max_execution_time = 500
        self.ner_models = {}
        self.rules = []
        self.filters = {}
        self.dicts = {}
        loader = threading.Thread(target=self.load_models, args=[model_dir])
        loader.start()
        # load dictionaries
        dict_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dicts_dir) for f in filenames if os.path.splitext(f)[1] == '.dict']
        for d in dict_files:
            fn = os.path.basename(d).split('.')[0]
            fin = open(d, "r")
            if fn not in self.dicts:
                self.dicts[fn] = {}
            for line in fin:
                line = re.sub(r'[\r\n]+', '', line)
                self.dicts[fn][line.decode('utf8', 'ignore')] = 1
            fin.close()
        # specific ner detection thresholds for better accuracy
        self.thresholds = {'persons' : [1.0, 2], 'locations' : [1.1, 1.1], 'organizations' : [1.1, 1.1]}
        # languages to apply thresholds
        self.lang_thersholds = ['en', 'es', 'fr', 'it', 'pt', 'de']
        # ner patterns for features
        self.patterns = {'streets-icase' : re.compile(ur"(^| )(%s)( |$)" % "|".join(map(re.escape, self.dicts['streets-marks'].keys())), re.IGNORECASE | re.U), 'streets' : re.compile(ur"(^(av.|ave|avenida|avenue|blvd|c/|calle|iela|parkway|pkwy|plaza|road|street|ul.) )|( (av.|ave|avenida|avenue|blvd|c/|calle|court|ct|drive|iela|lane|ln|parkway|pkwy|plaza|rd|road|st|st.|street|ul.)$)", re.IGNORECASE | re.U), 'shortening' : re.compile(ur"^[A-Z]([A-Za-z]{1,2})?\.$", re.IGNORECASE | re.U), 'companies' : re.compile(ur" (%s)( |$)" % "|".join(map(re.escape, self.dicts['companies-suffixes'].keys())), re.IGNORECASE | re.U), 'special_symbols' : re.compile(ur"^[“‟”•—–―․‥…‧ʺ«»❝˵˶＂!\"\#$%&'‛‘()’*+❛,\-—./:;<=>?@\[\\\]^_`{|}~]$", re.IGNORECASE | re.U), 'dates' : re.compile(ur"^\d+/\d+/\d{2,4}$", re.IGNORECASE | re.U), 'currency_ends' : re.compile(ur"[€¢£¥ƀƒ؋฿៛₡₦₨₩₪₫€₭₮₱₴₼₽﷼$]$", re.IGNORECASE | re.U), 'idn_context' : re.compile(ur"[A-Z][A-Z]\d\d \d{4} \d{4} \d{4}", re.IGNORECASE | re.U), 'currency' : re.compile(ur"^\d+(\.|\,)\d+((\,|\.)\d+)?((\,|\.)\d+)?$", re.IGNORECASE | re.U), 'idn' : re.compile(ur"^([A-Z]-?\d{6,}$)|(\d{6,}-?[A-Z])|(\d{15,})|([A-Z]-\d+(\.|/)\d{2,})|(\d+\.\d+\.\d+-[A-Z]$)|([A-Z][A-Z][A-Z]?\d{4,}[A-Z]?)", re.IGNORECASE | re.U)}
        # custom ner labels
        self.c_labels = {'nePer' : 'persons', 'neLoc' : 'locations', 'nePos' : 'positions', 'neOrg' : 'organizations', 'neStr' : 'streets', 'neEml' : 'emails', 'nePhn' : 'phone numbers', 'neIdn' : 'id numbers', 'neCrd' : 'cards', 'neCmp' : 'companies', 'neCur' : 'currencies', 'neEvn' : 'events', 'neDte': 'dates'}
        # cloud ner labels
        self.cloud_labels = {'Organization' : 'organizations', 'Location' : 'locations', 'Person' : 'persons', 'Email' : 'emails', 'PhoneNumber' : 'phone numbers', 'PHONE_NUMBER' : 'phone numbers', 'PERSON' : 'persons', 'LOCATION' : 'locations', 'ORGANIZATION' : 'organizations', 'Address' : 'streets', 'Company' : 'companies', 'ADDRESS' : 'streets', 'PRICE' : 'currencies', 'EVENT' : 'events', 'WORK_OF_ART' : 'artworks', 'CONSUMER_GOOD' : 'products', 'DATE' : 'dates'}
        # extra weight for specific entity types
        self.extra_weights = {'streets' : 2, 'companies' : 2}
        # high-level entity groups
        self.entity_groups = {'streets' : 'locations', 'companies' : 'organizations'}

    def load_models(self, model_dir):
        # load custom NER models
        ml_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(model_dir) for f in filenames if os.path.splitext(f)[1] == '.ml']
        for file in ml_files:
            # get basename as index of language model
            base = os.path.basename(file).split('.')[0]
            # load machine learning model
            self.logger.info("Load NER model " + base)
            vectorizer, model = pickle.load(open(file, 'rb'))
            self.ner_models[base] = (vectorizer, model)
        # load custom NER rules
        frules = open(model_dir + '/ner_rules.re')
        for line in frules:
            line = re.sub(r'[\r\n]+', '', line)
            line = line.decode('utf8', 'ignore')
            tag, rule = line.split('\t')
            self.rules.append((re.compile(rule, re.I | re.U), tag))
        frules.close()
        # load custom NER filters
        fin = open(model_dir + '/ner_filters.re')
        for line in fin:
            line = re.sub(r'[\r\n]+', '', line)
            line = line.decode('utf8', 'ignore')
            id, rule = line.split('\t')
            if id in self.filters:
                self.filters[id].append(re.compile(rule, re.I | re.U))
            else:
                self.filters[id] = [re.compile(rule, re.I | re.U)]
        fin.close()
        return 1
    
    def chunkstring(self, string, length):
        chunks = list(string[0+i:length+i] for i in range(0, len(string), length))
        for i in range(0, len(chunks) - 1):
            chunk = chunks[i]
            ind = chunk.find("\n", len(chunk) - 200)
            if ind != -1:
                chunks[i] = chunk[:ind+1]
                chunks[i+1] = chunk[ind+1:] + chunks[i+1]
        return chunks

    def check_in_dict(self, entity, dict_name):
        result = False
        if dict_name not in self.dicts:
            return False, 0
        if dict_name == "persons":
            entity = entity.lower()
        weight = 0
        words = entity.split(" ")
        for w in words:
            if w.isupper():
                w = w.capitalize()
            if w in self.dicts[dict_name]:
                result = True
                weight += 0.5
        final_weight = weight / len(words)
        return result, final_weight
    
    def check_pattern(self, text, pattern):
        res = pattern.search(text)
        if res is not None:
            return True
        else:
            return False
    
    def group_similar_ners(self, results):
        ners = {}
        for t in results:
            for n in results[t]['ners']:
                if n not in ners:
                    ners[n] = results[t]['ners'][n]
                else:
                    for e in results[t]['ners'][n]:
                        if e not in ners[n]:
                            ners[n][e] = results[t]['ners'][n][e]
                        else:
                            ners[n][e]['w'] += results[t]['ners'][n][e]['w']
                            ners[n][e]['model'] += '-' + results[t]['ners'][n][e]['model']
                            offsets = ners[n][e]['offsets']
                            for o in results[t]['ners'][n][e]['offsets']:
                                if o not in ners[n][e]['offsets']:
                                    ners[n][e]['offsets'].append(o)
        for n in ners:
            ners_keys = ners[n].keys()
            sz = len(ners_keys)
            i = 0
            replaced_indexes = []
            while i < sz:
                if i in replaced_indexes:
                    i += 1
                    continue
                offsets = ners[n][ners_keys[i]]["offsets"]
                for j in range(i + 1, sz):
                    if j in replaced_indexes:
                        continue
                    n_offsets = ners[n][ners_keys[j]]["offsets"]
                    save_index = -1
                    replace_index = -1
                    for o in offsets:
                        for k in n_offsets:
                            if o["start"] >= k["start"] and o["end"] <= k["end"]:
                                save_index = j
                                replace_index = i
                                break
                            if k["start"] >= o["start"] and k["end"] <= o["end"]:
                                save_index = i
                                replace_index = j
                                break
                        if replace_index != -1:
                            break
                    if replace_index != -1:
                        r_offsets = ners[n][ners_keys[replace_index]]["offsets"]
                        if ners[n][ners_keys[save_index]]["model"] != ners[n][ners_keys[replace_index]]["model"]:
                            ners[n][ners_keys[save_index]]["w"] += ners[n][ners_keys[replace_index]]["w"]
                        for m in r_offsets:
                            i_offsets = ners[n][ners_keys[save_index]]["offsets"]
                            exists = False
                            for i_o in i_offsets:
                                if m['start'] >= i_o['start'] and m['end'] <= i_o['end']:
                                    exists = True
                                    break
                            if exists == False:
                                ners[n][ners_keys[save_index]]["offsets"].append(m)
                            #if m not in ners[n][ners_keys[save_index]]["offsets"]:
                            #    ners[n][ners_keys[save_index]]["offsets"].append(m)
                        replaced_indexes.append(replace_index)
                        #self.logger.info("Save - " + ners_keys[save_index])
                        #self.logger.info("Delete - " + ners_keys[replace_index])
                        del ners[n][ners_keys[replace_index]]
                        break
                i += 1
        # group ners with different types
        ner_types = ners.keys()
        ners_to_delete = []
        for i in range(0, len(ner_types)):
            if i < len(ner_types) - 1:
                for e in ners[ner_types[i]]:
                    for j in range(i+1, len(ner_types)):
                        isFound = 0
                        for n in ners[ner_types[j]]:
                            for o in ners[ner_types[i]][e]["offsets"]:
                                for k in ners[ner_types[j]][n]["offsets"]:
                                    if o["start"] >= k["start"] and o["end"] <= k["end"]:
                                        #self.logger.info("Delete " + e + ' - ' + n)
                                        if ner_types[i] in self.entity_groups or ner_types[j] in self.entity_groups:
                                            if ner_types[i] in self.entity_groups and ner_types[j] == self.entity_groups[ner_types[i]]:
                                                ners_to_delete.append((ner_types[j], n))
                                                ners[ner_types[i]][e]['w'] += 1
                                                isFound = 1
                                                break
                                            if ner_types[j] in self.entity_groups and ner_types[i] == self.entity_groups[ner_types[j]]:
                                                ners_to_delete.append((ner_types[i], e))
                                                ners[ner_types[j]][n]['w'] += 1
                                                isFound = 1
                                                break
                                        if ners[ner_types[i]][e]["w"] > ners[ner_types[j]][n]["w"]:
                                            ners_to_delete.append((ner_types[j], n))
                                        else:
                                            ners_to_delete.append((ner_types[i], e))
                                        isFound = 1
                                        break
                                    if k["start"] >= o["start"] and k["end"] <= o["end"]:
                                        #self.logger.info("Delete " + e + ' - ' + n)
                                        if ner_types[i] in self.entity_groups or ner_types[j] in self.entity_groups:
                                            if ner_types[i] in self.entity_groups and ner_types[j] == self.entity_groups[ner_types[i]]:
                                                ners_to_delete.append((ner_types[j], n))
                                                ners[ner_types[i]][e]['w'] += 1
                                                isFound = 1
                                                break
                                            if ner_types[j] in self.entity_groups and ner_types[i] == self.entity_groups[ner_types[j]]:
                                                ners_to_delete.append((ner_types[i], e))
                                                ners[ner_types[j]][n]['w'] += 1
                                                isFound = 1
                                                break
                                        if ners[ner_types[i]][e]["w"] > ners[ner_types[j]][n]["w"]:
                                            ners_to_delete.append((ner_types[j], n))
                                        else:
                                            if ners[ner_types[i]][e]["w"] < ners[ner_types[j]][n]["w"]:
                                                ners_to_delete.append((ner_types[i], e))
                                            else:
                                                if len(e) > len(n):
                                                    ners_to_delete.append((ner_types[j], n))
                                                elif len(e) < len(n):
                                                    ners_to_delete.append((ner_types[i], e))
                                        isFound = 1
                                        break
                                if isFound == 1:
                                    break
        for d in ners_to_delete:
            try:
                del ners[d[0]][d[1]]
            except:
                pass
        return ners
    
    
    def insert_entities(self, entities, model_name):
        # regex for special chars removing
        regex = re.compile(r'[\r\n\s\t]+')
        ners = {}
        for e in entities:
            ent_text = e['text']
            ent_text = ent_text.strip()
            ent_text = regex.sub(' ', ent_text)
            if ent_text == "":
                continue
            isFiltered = False
            if str(e['count words']) in self.filters:
                for r in self.filters[str(e['count words'])]:
                    if r.match(e['pos']):
                        isFiltered = True
                        break
            if isFiltered == True:
                continue
            if e['type'] in self.c_labels:
                ent_type = self.c_labels[e['type']]
                # check entity type in patterns
                for p in ['streets', 'companies']:
                    if self.check_pattern(ent_text, self.patterns[p]) == True:
                        ent_type = p
                        break
                if ent_type not in ners:
                    ners[ent_type] = {}
                if ent_text in ners[ent_type]:
                    offsets = {"start" : e['start offset'], "end" : e['end offset']}
                    if offsets not in ners[ent_type][ent_text]["offsets"]:
                        ners[ent_type][ent_text]["offsets"].append(offsets)
                else:
                    ners[ent_type][ent_text] = {"w" : 1, "model": model_name, "offsets" : [{"start" : e['start offset'], "end" : e['end offset']}]}
                    isDict, w = self.check_in_dict(ent_text, ent_type)
                    if isDict == True:
                        ners[ent_type][ent_text]["w"] += w
                    if ent_type in self.extra_weights:
                        ners[ent_type][ent_text]["w"] += self.extra_weights[ent_type]
        return ners
    
    def get_features(self, tokens, index, words, g_patterns, lang):
        word = word_cp = tokens[index]["word"]
        lemma = tokens[index]["lemma"]
        tag = tokens[index]["tag"]
        word_lc = word.lower()
        context_string = ""
        left_context_string = " ".join(words[index-3:index+1])
        right_context_string = " ".join(words[index:index+6])
        if index >= 3:
            context_string = " ".join(words[index-3:index+4])
        else:
            context_string = " ".join(words[index:index+4])
        prevword = '' if index == 0 else tokens[index - 1]["word"]
        nextword = '' if index == len(tokens) - 1 else tokens[index + 1]["word"]
        prevprevword = '' if index < 2 else tokens[index - 2]["word"]
        prevprevword_lc = prevprevword.lower()
        nextnextword = '' if index + 2 > len(tokens)-1 else tokens[index + 2]["word"]
        nextnextword_lc = nextnextword.lower()
        nextword_lc = nextword.lower()
        prevword_lc = prevword.lower()
        prevallcaps = prevword == prevword.capitalize()
        nextallcaps = nextword == nextword.capitalize()
        is_double_name = False
        is_double_surname = False
        if '-' in word_lc:
            wrds = word_lc.split('-')
            if wrds[0] in self.dicts['persons'] and wrds[1] in self.dicts['persons']:
                is_double_name = True
            if wrds[0] in self.dicts['surnames'] and wrds[1] in self.dicts['surnames']:
                is_double_surname = True
        if word.isupper():
            word_cp = word.capitalize()
        try:
            features = {
                # word-based features
                'word': word,
                'lemma' : lemma,
                'tag' : tag,
                'prefix-4': '' if len(word) < 5 else word[:4],
                'suffix-4': '' if len(word) < 5 else word[-4:],
                'prefix-5': '' if len(word) < 6 else word[:5],
                'lenght' :  len(word),
                'is_letter' : True if len(word) == 1 and word[0].isalpha() else False,
                'is_digit' : word.isdigit(),
                'has_digit' : bool(re.search(r'\d', word)),
                'contains-dot' : '.' in word,
                'contains-comma' : ',' in word,
                'is_comma' : True if word == ',' else False,
                'is_special_symbol' :  True if self.patterns['special_symbols'].search(word) else False,
                'is_all_caps': word.upper() == word,
                'is_first': index == 0,
                'is_last': index == len(tokens) - 1,
                'is_capitalized': word[0].upper() == word[0],
                'is_all_lower' : word.lower() == word,
                'is_shortening' : True if self.patterns['shortening'].search(word) else False,
                'capitals_inside': word[1:].lower() != word[1:],
                'contains_email_sign' : '@' in word,
                'is_next_@' : True if nextword == '@' else False,
                'is_prev_@' : True if prevword == '@' else False,
                
                # pattern-based features
                'is_idn_pattern' : True if self.patterns['idn'].search(word) or self.patterns['idn_context'].search(context_string) else False,
                'is_currency_pattern' : True if self.patterns['currency'].search(word) else False,
                'is_next_currency' : True if word_lc in self.dicts['currencies'] and self.patterns['currency'].search(nextword) else False,
                'is_prev_currency' : True if word_lc in self.dicts['currencies'] and self.patterns['currency'].search(prevword) else False,
                'is_currency_ends' : True if self.patterns['currency_ends'].search(word) else False,
                'is_date' : True if self.patterns['dates'].search(word) else False,
                'has_street' : True if g_patterns['streets'] == True and self.patterns['streets-icase'].search(left_context_string) else False,
                'has_company' : True if g_patterns['companies'] == True and self.patterns['companies'].search(right_context_string) else False,
            
                # # context-based features
                'prev_word': prevword,
                'next_word': nextword,
                'prev_lemma' : '' if prevword == "" else tokens[index-1]["lemma"],
                'next_lemma' : '' if nextword == "" else tokens[index+1]["lemma"],
                'prev_pos' : '' if prevword == "" else tokens[index-1]["tag"],
                'next_pos' : '' if nextword == "" else tokens[index+1]["tag"],
                'next_has_digit' : bool(re.search(r'\d', nextword)),
                'prev_has_digit' : bool(re.search(r'\d', prevword)),
                'next_is_digit' : True if nextword.isdigit() else False,
                'prev-all-caps': prevallcaps,
                'next-all-caps': nextallcaps,
                'prev-capitalized': False if index == 0 else prevword[0].upper() == prevword[0],
                'next-capitalized': False if index == len(tokens) - 1 else nextword[0].upper() == nextword[0],
                'prev-prev-word': prevprevword,
                'next-next-word': nextnextword,
                'prev_prev_lemma' : '' if prevprevword == "" else tokens[index-2]["lemma"],
                'next_next_lemma' : '' if nextnextword == "" else tokens[index+2]["lemma"],
            
            # dict-based features
                'is_in_currencies': True if word_lc in self.dicts['currencies'] or lemma in self.dicts['currencies'] else False,
                'is_prev_or_next_in_currencies': True if nextword_lc in self.dicts['currencies'] or prevword_lc in self.dicts['currencies'] else False,
                'is_in_positions' : True if word_lc in self.dicts['positions'] or lemma in self.dicts['positions'] else False,
                'is_prev_in_positions' : True if prevword_lc in self.dicts['positions'] else False,
                'is_next_in_positions' : True if nextword_lc in self.dicts['positions'] else False,
                'is_in_names' : True if word_lc in self.dicts['persons'] or lemma in self.dicts['persons'] else False,
                'is_double_name' : is_double_name,
                'is_double_surname' : is_double_surname,
                'is_prev_in_names' : True if prevword_lc in self.dicts['persons'] else False,
                'is_next_in_names' : True if nextword_lc in self.dicts['persons'] else False,
                'is_in_surnames' : True if word_lc in self.dicts['surnames'] or lemma in self.dicts['surnames'] else False,
                'is_next_in_surnames' : True if nextword_lc in self.dicts['surnames'] else False,
                'is_in_titles' : True if word_lc in self.dicts['titles'] or lemma in self.dicts['titles'] else False,
                'is_in_locations' : True if word_cp in self.dicts['locations'] else False,
                'is_next_in_locations' : True if nextword in self.dicts['locations'] else False,
                'is_prev_in_locations' : True if prevword in self.dicts['locations'] else False,
                'is_in_loc_main_word' : True if word_lc in self.dicts['loc-main-word'] or lemma in self.dicts['loc-main-word'] else False,
                'is_prev_in_loc_main_word' : True if prevword_lc in self.dicts['loc-main-word'] else False,
                'is_prev_prev_in_loc_main_word' : True if prevprevword_lc in self.dicts['loc-main-word'] else False,
                'is_full_location' : True if (prevword_lc + ' ' + word_lc + ' ' + nextword_lc) in self.dicts['locations-full'] or (prevprevword_lc + ' ' + prevword_lc + ' ' + word_lc) in self.dicts['locations-full'] or (word_lc + ' ' + nextword_lc + ' ' + nextnextword_lc) in self.dicts['locations-full'] or (prevword_lc + ' ' + word_lc) in self.dicts['locations-full'] or (word_lc + ' ' + nextword_lc) in self.dicts['locations-full'] else False,
                'is_in_company_suffixes' : True if word in self.dicts['companies-suffixes'] else False,
                'is_prev_in_company_suffixes' : True if prevword_lc in self.dicts['companies-suffixes'] else False,
                'is_next_in_company_suffixes' : True if nextword_lc in self.dicts['companies-suffixes'] else False,
                'is_next_next_in_company_suffixes' : True if nextnextword_lc in self.dicts['companies-suffixes'] else False,
                'is_in_companies' : True if word in self.dicts['companies'] else False,
                'is_full_company' : True if (prevword_lc + ' ' + word_lc + ' ' + nextword_lc) in self.dicts['companies-full'] or (prevprevword_lc + ' ' + prevword_lc + ' ' + word_lc) in self.dicts['companies-full'] or (word_lc + ' ' + nextword_lc + ' ' + nextnextword_lc) in self.dicts['companies-full'] or (prevword_lc + ' ' + word_lc) in self.dicts['companies-full'] or (word_lc + ' ' + nextword_lc) in self.dicts['companies-full'] else False,
                'is_in_streets' : True if word_lc in self.dicts['streets-marks'] else False,
                'is_prev_in_streets' : True if prevword_lc in self.dicts['streets-marks'] else False,
                'is_prev_prev_in_streets' : True if prevprevword_lc in self.dicts['streets-marks'] else False,
                'is_next_in_streets' : True if nextword_lc in self.dicts['streets-marks'] else False,
                'is_full_street' : True if (prevword_lc + ' ' + word_lc + ' ' + nextword_lc) in self.dicts['streets-full'] or (prevprevword_lc + ' ' + prevword_lc + ' ' + word_lc) in self.dicts['streets-full'] or (word_lc + ' ' + nextword_lc + ' ' + nextnextword_lc) in self.dicts['streets-full'] or (prevword_lc + ' ' + word_lc) in self.dicts['streets-full'] or (word_lc + ' ' + nextword_lc) in self.dicts['streets-full'] else False,
                'is_in_org_main_word' :  True if word_cp in self.dicts['org-main-word'] else False,
                'is_in_org_full' : True if (prevword_lc + ' ' + word_lc + ' ' + nextword_lc) in self.dicts['org-full'] or (prevprevword_lc + ' ' + prevword_lc + ' ' + word_lc) in self.dicts['org-full'] or (word_lc + ' ' + nextword_lc + ' ' + nextnextword_lc) in self.dicts['org-full'] or (prevword_lc + ' ' + word_lc) in self.dicts['org-full'] or (word_lc + ' ' + nextword_lc) in self.dicts['org-full'] or word_lc in self.dicts['org-full'] else False,
                'is_next_in_org_main_word' : True if nextword in self.dicts['org-main-word'] else False,
                'is_next_next_in_org_main_word' : True if nextnextword in self.dicts['org-main-word'] else False,
                'is_prev_in_org_main_word' : True if prevword in self.dicts['org-main-word'] else False,
                'is_prev_prev_in_org_main_word' : True if prevprevword in self.dicts['org-main-word'] else False,
                'is_in_events' : True if word in self.dicts['events'] else False,
                'is_in_events_main_word' : True if word in self.dicts['events-main-word'] else False,
                'is_next_in_events' : True if nextword in self.dicts['events-main-word'] else False,
                'is_next_next_in_events' : True if nextnextword in self.dicts['events-main-word'] else False,
                'is_prev_in_events' : True if prevword in self.dicts['events-main-word'] else False,
                'is_prev_prev_in_events' : True if prevprevword in self.dicts['events-main-word'] else False
            }
            if lang in self.lang_with_add_feature:
                features['sentence lenght'] = len(words)
            return features
        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(exc_tb.tb_lineno)
            self.logger.info("Error " + str(e))
    
    def extract_custom_ners(self, text, language, results, sent_tokenize_list, sent_offsets):
        thread_id = uuid.uuid4().hex
        results[thread_id] = {}
        custom_entities = []
        model_name = language
        try:
            if language in self.ner_models:
                # extract ners via custom model
                start_time = time.time()
                for s in range(len(sent_tokenize_list)):
                    curr_time = time.time() - start_time
                    if curr_time >= self.max_execution_time:
                        self.logger.info("Maximum processing time has reached")
                        break
                    words, offsets = self.lp.tokenize_words(sent_tokenize_list[s], language)
                    tokens, chunks, relations = self.lp.parse_words(words, offsets, language, chunks = False, lemma = True, relations = False, tags_normalization = True)
                    input_features = []
                    vectorizer = self.ner_models[language][0]
                    model = self.ner_models[language][1]
                    untagged_str = " ".join(words)
                    # match global patterns to improve the speed
                    global_patterns = {'streets' : False, 'companies' : False}
                    if self.patterns['streets-icase'].search(untagged_str):
                        global_patterns['streets'] = True
                    if self.patterns['companies'].search(untagged_str):
                        global_patterns['companies'] = True
                    for i in range(len(tokens)):
                        input_features.append(self.get_features(tokens, i, words, global_patterns, language))
                    # self.logger.info(s)
                    features = vectorizer.transform(input_features)
                    ner_tags = model.predict(features)
                    # self.logger.info(ner_tags)
                    entity = {}
                    for i in range(len(tokens)):
                        if ner_tags[i] != "neNone":
                            if 'type' in entity and entity['type'] == ner_tags[i]:
                                entity['end offset'] = tokens[i]['end offset'] + sent_offsets[s][0]
                                entity['count words'] += 1
                                entity['pos'] += " " + tokens[i]['tag']
                            if 'type' not in entity or entity['type'] != ner_tags[i]:
                                if 'type' in entity:
                                    entity['text'] = text[entity['start offset']:entity['end offset']]
                                    custom_entities.append(entity)
                                # skip determinant as first word and create new entity
                                if tokens[i]['tag'] == "DT":
                                    entity = {'start offset' : tokens[i]['end offset'] + sent_offsets[s][0], 'end offset' : tokens[i]['end offset'] + sent_offsets[s][0], 'count words' : 1, 'pos' : tokens[i]['tag'], 'type' : ner_tags[i]}
                                else:
                                    entity = {'start offset' : tokens[i]['start offset'] + sent_offsets[s][0], 'end offset' : tokens[i]['end offset'] + sent_offsets[s][0], 'count words' : 1, 'pos' : tokens[i]['tag'], 'type' : ner_tags[i]}
                        else:
                            if 'type' in entity:
                                entity['text'] = text[entity['start offset']:entity['end offset']]
                                custom_entities.append(entity)
                                entity = {}
        except Exception, e:
            self.logger.error('Error extracting ners from custom model ' + str(e))
            results[thread_id]['error'] = True
            results[thread_id]['ners'] = {}
            return results
        try:
            for r in self.rules:
                for m in r[0].finditer(text):
                    entity = {'start offset' : m.start(), 'end offset' : m.start() + len(m.group(0)), 'count words' : 1, 'pos' : "NN", 'type' : r[1]}
                    entity['text'] = text[entity['start offset'] : entity['end offset']]
                    custom_entities.append(entity)
        except Exception, e:
            self.logger.error('Error applying rules ' + str(e))
        ners = self.insert_entities(custom_entities, model_name)
        results[thread_id]['error'] = False
        results[thread_id]['ners'] = ners
        self.logger.info("Return custom entities")
        return results
    
    def extract_ners(self, text, language):
        results = {}
        clean_text = re.sub(r'[\r\n][\r\n]', '. ', text)
        sent_tokenize_list, sent_offsets = self.lp.tokenize_sentences(clean_text, language, preprocessing=False)
        self.logger.info("Extract ners via custom model")
        custom = threading.Thread(target=self.extract_custom_ners, args=[clean_text, language, results, sent_tokenize_list, sent_offsets])
        custom.start()
        custom.join()
        #self.logger.info("Group Ners " + str(results))
        self.logger.info("Group Ners")
        extracted_ners = self.group_similar_ners(results)
        self.logger.info("Prepare ners")
        #self.logger.info("After grouping: " + str(extracted_ners))
        final_result = {}
        offsets = []
        for o in sent_offsets:
            offsets.append({"s" : o[0], 'e' : o[1]})
        for t in extracted_ners:
            if t not in final_result:
                final_result[t] = []
            for p in extracted_ners[t]:
                if t not in self.thresholds:
                    threshold_weight = 1.0
                elif language not in self.lang_thersholds:
                    threshold_weight = 1.0
                else:
                    threshold_weight = self.thresholds[t][0]
                    if p.count(" ") == 0:
                        threshold_weight = self.thresholds[t][1]
                if extracted_ners[t][p]["w"] >= threshold_weight:
                    if p in self.dicts["garbage"]:
                        continue
                    extracted_ners[t][p]["entity"] = p
                    del extracted_ners[t][p]["w"]
                    del extracted_ners[t][p]["model"]
                    final_result[t].append(extracted_ners[t][p])
        #self.logger.info("Final " + str(final_result))
        self.logger.info("Return results")
        del results
        del extracted_ners
        return final_result, offsets