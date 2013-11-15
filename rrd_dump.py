#!/usr/bin/python
from lxml import etree
import sys
import re
import subprocess


def parse_rrddump_output(stream):
    root = etree.parse(stream)
    step_size = float(root.find('step').text)

    for rra in root.iter('rra'):
        pdp_per_row = float(rra.find('pdp_per_row').text)
        duration = step_size * pdp_per_row

        database = rra.find('database')
        for row in database.iter('row'):
            timestamp_comment = row.getprevious().text
            (ts,) = re.match(r'.* / (\d+)\b', timestamp_comment).groups()
            ts = float(ts)

            value = float(row.find('v').text)
            yield (ts, value, duration)


def run_rrddump(filename):
    pipe = subprocess.Popen(["rrdtool", "dump", filename], stdout=subprocess.PIPE)
    return pipe.stdout


if __name__ == '__main__':
    if len(sys.argv) > 1:
        stream = open(sys.argv[1])
    else:
        stream = sys.stdin

    parse_rrddump_output(stream)
