#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import collections
import csv
import json
import operator
import math

Rec = collections.namedtuple('Rec', ['file', 'log_like', 'df', 'n', 'aic',
                                     'bic'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('json_file',
                   nargs=argparse.ONE_OR_MORE)
    p.add_argument('-o', '--outfile', default='-')
    a = p.parse_args()

    results = []
    for f in a.json_file:
        with argparse.FileType('r')(f) as fp:
            d = json.load(fp)
            log_like = d['logLikelihood']
            df = d['degreesOfFreedom']
            n = d['alignedBases']
            r = Rec(f, log_like, df, n,
                    aic=2 * df - 2 * log_like,
                    bic=-2 * log_like + df * math.log(n))
            results.append(r)

    results.sort(key=operator.attrgetter('aic'))
    with argparse.FileType('w')(a.outfile) as ofp:
        w = csv.writer(ofp, lineterminator='\n')
        w.writerow(Rec._fields)
        w.writerows(results)


if __name__ == '__main__':
    main()
