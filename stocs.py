#!/usr/bin/python2.7
import argparse
import csv
import numpy as np
import scipy as sp
import sys
sys.path.append("./HPT/")
sys.path.append("./lilliefors/")
from HPT import *
from lilliefors import *
from scipy import stats

def rdt_open(rdt_file):
    f = file(rdt_file, 'r')
    reader = csv.DictReader(f)
    return reader

def get_dataset(reader, field):
    dsl = []
    for row in reader:
        dsl.append(float(row[field]))
    return dsl

def get_dataset_list(ds, field):
    dsl = []
    for i in ds:
        reader = rdt_open(i)
        dsl.append(get_dataset(reader, field))
    return dsl

def calculate_ci(length, mean, std, conf):
    if length >= 30:
        ci = stats.norm.interval(alpha = 1 - np.float(conf)/100,
                                 loc = mean, scale = std)
    else:
        ci = stats.t.interval(1 - np.float(conf)/100, length - 1,
                              loc = mean, scale = std)
    return mean - ci[0]

def get_means_list(ds_list):
    ml = []
    for i in ds_list:
        i = np.array(i, dtype = float)
        ml.append((i.mean(), stats.gmean(i), np.median(i), i.var(), i.std(),
                   len(i), i.max(), i.min()))
    return ml

def find_replace(string, old_value, new_value):
    if string.find(old_value) >= 0:
        return string.replace(old_value, new_value)
    else:
        return string

def parse_symbols_ds(text, sym, ds, ds_list, conf):
    string_list = text.split(sym)
    for item in string_list[1:]:
        st_list = item.split('-')
        num = st_list[0]
        if num.isalnum():
            dsN = ds[int(num)]
            pref = sym + num + '-'
            text = find_replace(text, pref + 'av', str(dsN[0]))
            text = find_replace(text, pref + 'gm', str(dsN[1]))
            text = find_replace(text, pref + 'med', str(dsN[2]))
            text = find_replace(text, pref + 'va', str(dsN[3]))
            text = find_replace(text, pref + 'std', str(dsN[4]))
            text = find_replace(text, pref + 'ci', str(calculate_ci(dsN[5],
                                                       dsN[0], dsN[4], conf)))
            text = find_replace(text, pref + 'max', str(dsN[6]))
            text = find_replace(text, pref + 'min', str(dsN[7]))
            if text.find(pref + 'lillie-conf') >= 0:
                lillie_conf = 1 - lilliefors(ds_list[int(num)])[1]
                text = text.replace(pref + 'lillie-conf', str(lillie_conf))
    return text

def parse_symbols_avgm(text, sym, ds1, ds2, conf):
    string_list = text.split(sym)
    if sym == "%av-":
        choice = 0
    elif sym == "%gm-":
        choice = 1
    for item in string_list[1:]:
        st_list = item.split('-')
        num = st_list[0]
        if num.isalnum():
            ds1N = ds1[int(num)]
            ds2N = ds2[int(num)]
            pref = sym + num + '-'
            text = find_replace(text, pref + 'diff',
                                str(ds1N[choice] - ds2N[choice]))
            text = find_replace(text, pref + 'ratio',
                                str(ds1N[choice] / ds2N[choice]))
    return text

def format_ds(output, conf, ds_list, num):
    ds = get_means_list(ds_list)
    ds = np.array(ds, dtype = float)
    m = ds.mean(axis = 0)[0]
    mx = ds.max(axis = 0)[6]
    mn = ds.min(axis = 0)[7]
    ss = stats.gmean(ds, axis = 0)[2]
    output = find_replace(output, '%ds'+str(num)+'-av', str(m))
    output = find_replace(output, '%ds'+str(num)+'-max', str(mx))
    output = find_replace(output, '%ds'+str(num)+'-min', str(mn))
    output = find_replace(output, '%ds'+str(num)+'-spec', str(ss))
    if output.find('%ds'+str(num)+'-lillie-conf') >= 0:
        lillie_conf = 1 - lilliefors(ds[0])[1]
        output = output.replace('%ds'+str(num)+'-lillie-conf',  \
                                str(lillie_conf))
    if output.find('%ds'+str(num)+'-') >= 0:
	    output = parse_symbols_ds(output, '%ds'+str(num)+'-', ds, ds_list, conf)
    return output, m, ss

def format_avgm(output, ds1, ds2, conf):
    if output.find('%av-') >= 0:
        output = parse_symbols_avgm(output, '%av-',ds1, ds2, conf)
    if output.find('%gm-') >= 0:
        output = parse_symbols_avgm(output, '%gm-',ds1, ds2, conf)
    return output

def format_output(string, ds1_list, ds2_list, conf, md, acc):
    output = str(string)
    output = find_replace(output, '\\n', '\n')
    output = find_replace(output, '\\t', '\t')
    output = find_replace(output, '%conf-lvl', str(float(conf)/100))
    if ds1_list:
        output, m1, gm1 = format_ds(output, conf, ds1_list, 1)
    if ds2_list:
        output, m2, gm2 = format_ds(output, conf, ds2_list, 2)
    if ds1_list and ds2_list:
    	if md:
    		m1, gm1, m2, gm2 = m2, gm2, m1, gm1
        output = find_replace(output, '%av-ratio', str(m1/float(m2)))
        output = find_replace(output, '%av-diff', str(m1 - m2))
        output = find_replace(output, '%spec-spd', str(gm1/float(gm2)))
        output = find_replace(output, '%spec-diff', str(gm1 - gm2))
        output = format_avgm(output, ds1_list, ds2_list, conf)
	if output.find('%hpt-conf-av') >= 0:
            conf_av = 1 - HPT(ds1_list, ds2_list, speedup = m1/float(m2), \
                            mode = md)[0]
            output = output.replace("%hpt-conf-av",  str(conf_av))
        if output.find('%hpt-conf-spec') >= 0:
            conf_gm = 1 - HPT(ds1_list, ds2_list, speedup = gm1/float(gm2),\
                            mode = md)[0]
            output = output.replace("%hpt-conf-spec", str(conf_gm))
        if output.find('%hpt-spd') >= 0:
            hpt_su = HPT_max_speedup(ds1_list, ds2_list,
                                        rel = np.float(conf)/100,
                                        mode = md, acc = acc)
            output = output.replace("%hpt-spd",  str(("{:."+str(acc)+"f}").format(hpt_su)))
            if output.find('%hpt-conf') >= 0:
                hpt_conf = 1 - HPT(ds1_list, ds2_list, speedup = hpt_su, \
	    		            mode = md)[0]
                output = output.replace("%hpt-conf", str(hpt_conf))
    return output    

parser = argparse.ArgumentParser(description =
                                 'Compares two datasets statistcally',
                                 epilog = "The speedups and differences are\
                                 calculated considering first ds1 then ds2",
                                 formatter_class= 
                                 argparse.ArgumentDefaultsHelpFormatter)
ds = parser.add_argument_group('datasets')
ds.add_argument('-ds1', help = 'names of the first dataset files', type = str,
                nargs = '+', required = True)
ds.add_argument('-ds2', help = 'names of the second dataset files', type = str,
                nargs = '+')
parser.add_argument('-cf', help = 'field to be compared', type = str,
                    required = True)
parser.add_argument('--smw', dest = 'mode', help = 'change to the HPT mode \
                    where smaller values are better', action = 'store_true',
                    default = False)
parser.add_argument('-cl', dest = 'confidence',
                    help = 'confidence level', type = float, default = 95.0)
parser.add_argument('-of', dest = 'format', help = 'output format',
					type = str, default = "SPEC rate: %ds1-spec\n" + \
					"Overall mean: %ds1-av", required = False)
parser.add_argument('-hpt-acc', dest = 'acc', 
                    help = 'number of decimal digits in HPT output',
                    type = int, default = 3)
args = parser.parse_args()
if args.ds1:
    ds1 = get_dataset_list(args.ds1, args.cf)
else:
    ds1 = None
if args.ds2:
    ds2 = get_dataset_list(args.ds2, args.cf)
else:
    ds2 = None
output = format_output(args.format, ds1, ds2, args.confidence, args.mode, args.acc)
if output != "None":
    print(output)
