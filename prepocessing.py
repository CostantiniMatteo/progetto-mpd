#!/usr/bin/env python
import re, io, os
import pandas as pd
import numpy as np
from datetime import datetime


# Per stampare le colonne del dataframe senza che vada a capo
pd.set_option('display.expand_frame_repr', False)

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


# Parsa la data
def date_to_timestamp(m):
    return str(
        datetime.strptime(m.group().strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    )[:-2] # Perché non mi piaceva vedere .0


# Trova lo stato più "vicino" alla rilevazione del sensore al tempo timestamp
def find_state(timestamp, states):
    l = np.asarray(states)
    idx = (np.abs(l - timestamp)).argmin()
    return states[idx]


def main():
    files = [
        'OrdonezA_ADLs',
        'OrdonezA_Sensors',
        'OrdonezB_ADLs',
        'OrdonezB_Sensors',
    ]
    fieldnames_ADL = ['start_time', 'end_time', 'activity']
    fieldnames_sensors = ['start_time', 'end_time', 'location','type', 'place']
    dfs = {}

    os.system('rm dataset_csv/*')
    for f in files:
        # Conversione txt -> csv e conversione date in timestamp
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
        df.reset_index(inplace=True)
        dfs[f] = df

    for i in range(2):
        adl = dfs[files[2 * i]]
        obs = dfs[files[2 * i + 1]]

        # Calcolo timestamp medio di osservazioni e stati
        adl['mean_timestamp'] = (adl['start_time'] + adl['end_time']) // 2
        obs['mean_timestamp'] = (obs['start_time'] + obs['end_time']) // 2

        # Ad ogni osservazione di un sensore viene associato l'evento più vicino
        obs['state_timestamp'] = 0
        for idx, row in obs.iterrows():
            state_timestamp = find_state(row[6], adl['mean_timestamp'])
            obs.at[idx, 'state_timestamp'] = state_timestamp

        merged = pd.merge(obs, adl,
            left_on='state_timestamp',right_on='mean_timestamp'
        )

        # Drop colonne inutili e rename
        merged.drop(axis=1, inplace=True,
            columns=['state_timestamp', 'index_x', 'index_y', 'type', 'place'])
        merged.columns = ['start_time_sensor', 'end_time_sensor', 'location',
            'mean_sensor', 'start_time_activity', 'end_time_activity',
            'activity', 'mean_activity']

        # Conversione dei valori categorici in interi
        cols = ['location', 'activity']
        merged[cols] = merged[cols].apply(lambda x: x.astype('category'))
        merged[cols] = merged[cols].apply(lambda x: x.cat.codes)

        merged.to_csv(f'dataset_csv/Ordonez{"A" if i == 0 else "B"}.csv',
            sep=',', index=False)


if __name__ == '__main__':
    main()
