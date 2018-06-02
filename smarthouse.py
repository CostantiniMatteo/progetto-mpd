#!/usr/bin/env python
import re, csv

def sensor_data_to_csv(path):
    with open(path, 'r') as f:
        next(f); next(f);
        lines = [x.strip() for x in f.readlines()]
        text = "\n".join(lines)
        text = re.sub(r'\s\s+', ',', text)
        text = re.sub(r'\t', ',', text)
    return text

def date_to_timestamp(dataset):
    pass



if __name__ == '__main__':
    # Convertire txt -> csv
    files = ['OrdonezA_ADLs', 'OrdonezB_ADLs',
        'OrdonezA_Sensors', 'OrdonezB_Sensors']

    for f in files:
        with open(f'dataset_csv/{f}.csv', 'w') as out:
            out.write(sensor_data_to_csv(f'dataset/{f}.txt'))

    # Convertire date in timestamp
    # Controllare la consistenza delle rilevazioni
    # Ordinare in ordine cronologico
    print("Work in progress...")
