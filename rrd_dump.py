#!/usr/bin/python
from lxml import etree
import sys
import re
import subprocess
import json

def parse_rrddump_output(stream):
    root = etree.parse(stream)
    step_size = float(root.find('step').text)

    col_names = [ds.find('name').text.strip() for ds in root.findall('ds')]

    def yield_points():
        for rra in root.findall('rra'):
            pdp_per_row = float(rra.find('pdp_per_row').text)
            duration = step_size * pdp_per_row

            database = rra.find('database')
            seen_real_value = False
            for row in database.findall('row'):
                timestamp_comment = row.getprevious().text
                (ts,) = re.match(r'.* / (\d+)\b', timestamp_comment).groups()
                ts = float(ts)

                values = {}
                for i, v in enumerate(row.findall('v')):
                    seen_real_value = seen_real_value or v.text != "NaN"
                    values[col_names[i]] = float(v.text)
                if seen_real_value:
                    yield (ts, values, duration)

    return step_size, yield_points()

def run_rrddump(filename):
    pipe = subprocess.Popen(["rrdtool", "dump", filename], stdout=subprocess.PIPE)
    return pipe.stdout


if __name__ == '__main__':

    step_size, points = parse_rrddump_output(run_rrddump(sys.argv[1]))
    for point in points:
        print json.dumps(point)

