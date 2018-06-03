#!/usr/bin/env python
import re, io
import pandas as pd
from datetime import datetime
from IPython import embed


# Conversione del dataset da file .txt a file .csv
def sensor_data_to_csv(path):
    date_regex = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[ ]?')
    with open(path, 'r') as f:
        next(f); next(f);
        lines = [x.strip() for x in f.readlines()]
        text = "\n".join(lines)
        text = re.sub(r'[ \t]*\t+', ',', text)
        text = re.sub(date_regex, date_to_timestamp, text)

    return text


def date_to_timestamp(m):
    return str(
        datetime.strptime(m.group().strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    )[:-2] # Perché non mi piaceva vedere .0


def main():
    # Conversione txt -> csv e conversione date in timestamp
    files = [
        'OrdonezA_ADLs',
        'OrdonezB_ADLs',
        'OrdonezA_Sensors',
        'OrdonezB_Sensors',
    ]
    fieldnames_ADL = ['start_time', 'end_time', 'activity']
    fieldnames_sensors = ['start_time', 'end_time', 'location','type', 'place']

    for f in files:
        out = io.StringIO()
        if f.find('ADL') > 0:
            out.write(','.join(fieldnames_ADL))
        else:
            out.write(','.join(fieldnames_sensors))
        out.write('\n')
        out.write(sensor_data_to_csv(f'dataset/{f}.txt'))
        out.seek(0)

        # Controllo consistenza delle rilevazioni e sort degli eventi
        df = pd.read_csv(out, sep=',')
        df.sort_values(by=['start_time'], inplace=True)
        df.drop(df[df['start_time'] > df['end_time']].index, inplace=True)
        df.to_csv(f'dataset_csv/{f}.csv', index=False)


if __name__ == '__main__':
    main()
