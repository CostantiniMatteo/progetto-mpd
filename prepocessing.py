import io, os
import pandas as pd
import numpy as np
from datetime import datetime


# Per stampare le colonne del dataframe senza che vada a capo
pd.set_option('display.expand_frame_repr', False)

# Legge il txt e lo converte in csv trasformando le date in timestamp
def txt_to_csv(path):
    df = pd.read_csv(path, sep='\t+', engine='python')
    df.drop(0, inplace=True)
    df.columns = [x.strip().lower().replace(' ', '_') for x in df.columns]
    df['start_time'] = df['start_time'].apply(date_to_timestamp)
    df['end_time'] = df['end_time'].apply(date_to_timestamp)

    return df


# Parsa la data
def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


# Trova lo stato più "vicino" alla rilevazione del sensore al tempo timestamp
def find_state(timestamp, states):
    l = np.asarray(states)
    idx = (np.abs(l - timestamp)).argmin()
    return states[idx]


def main():
    os.system('rm dataset_csv/*')
    files = [
        'OrdonezA_ADLs',
        'OrdonezA_Sensors',
        'OrdonezB_ADLs',
        'OrdonezB_Sensors',
    ]

    dfs = {}
    for f in files:
        df = txt_to_csv(f'dataset/{f}.txt')
        df.sort_values(by=['start_time'], inplace=True)
        df.drop(df[df['start_time'] > df['end_time']].index, inplace=True)
        df.to_csv(f'dataset_csv/{f}.csv', index=False)
        df.reset_index(inplace=True)
        dfs[f] = df

    for f in range(2):
        adl = dfs[files[2 * f]]
        obs = dfs[files[2 * f + 1]]

        # Join tra le attività e sensore se i due eventi si accavallano
        start_times = []; end_times = []; activities = []; sensors = []
        for i in range(adl.shape[0]):
            start = adl.loc[i, 'start_time']
            end = adl.loc[i, 'end_time']
            state = adl.loc[i, 'activity']

            # I due eventi si accavallano
            q =  obs.query('@end >= start_time and end_time >= @start')
            for j, row in q.iterrows():
                start_times.append(row['start_time'])
                end_times.append(row['end_time'])
                activities.append(state)
                sensors.append(row['location'])

        merged = pd.DataFrame(
            columns=['start_time', 'end_time', 'activity', 'sensor'],
            data = {
                'start_time': start_times,
                'end_time': end_times,
                'activity': activities,
                'sensor': sensors,
            }
        )

        # Conversione dei valori categorici in interi
        cols = ['sensor', 'activity']
        merged[cols] = merged[cols].apply(lambda x: x.astype('category'))
        merged[cols] = merged[cols].apply(lambda x: x.cat.codes)

        merged.to_csv(f'dataset_csv/Ordonez{"A" if f == 0 else "B"}.csv',
            sep=',', index=False)


if __name__ == '__main__':
    main()
