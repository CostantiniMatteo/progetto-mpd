import io, os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


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


# Restituisce il minuto del giorno
def day_minute(timestamp):
    return ((timestamp // 60) % (24*60))


# Suddivide la giornata in 4 slice
# Restituisce la frazione del giorno a cui appartiene il timestamp
def day_period(timestamp):
    h = ((timestamp // (60*60)) % 24)
    if h < 6: return 0
    elif h < 12: return 1
    elif h < 18: return 2
    else: return 3


# TODO: Prendi quella che ne ha di più
# Discretizza il tempo e unisce i due dataset di attività ed osservazioni
def merge_dataset(adl, obs, start_date, end_date, length=60, on_att='id'):
    first_minute = date_to_timestamp(start_date)
    last_minute = date_to_timestamp(end_date)
    n_sens = max(obs[on_att]) + 1

    timestamps = []; activities = []; sensors = []; periods = []
    for s in tqdm(range(first_minute, last_minute + 1, length)):
        e = s + length - 1

        # Trova i sensori attivi al tempo s
        q = obs.query('@e >= start_time and end_time >= @s')
        sl = q[on_att].tolist()
        active_sensors = "".join('1' if x in sl else '0' for x in range(n_sens))

        # Trova l'attività al tempo s
        q = adl.query('@e >= start_time and end_time >= @s')
        if q.shape[0] == 0:
            # Se la configurazione precedente è uguale a quella attuale
            # probabilmente l'attività è la stessa
            if len(sensors) > 0 and active_sensors == sensors[-1]:
                activity = activities[-1]
            # Stato che indica 'nessuna attività'
            else:
                activity = max(adl['activity']) + 1
        else: activity = q.iloc[0]['activity']

        # Calcola il periodo della giornata
        period = day_period(s)

        timestamps.append(s)
        activities.append(activity)
        sensors.append(active_sensors)
        periods.append(period)

    result = pd.DataFrame(
        columns=['timestamp', 'activity', 'sensors', 'period'],
        data = {
            'timestamp': timestamps,
            'activity': activities,
            'sensors': sensors,
            'period': periods,
        }
    )

    return result


def main(length=60):
    if not os.path.exists('dataset_csv'): os.makedirs('dataset_csv')
    files = [
        'OrdonezA_ADLs',
        'OrdonezA_Sensors',
        'OrdonezB_ADLs',
        'OrdonezB_Sensors',
    ]

    # Lettura dei file txt con attività e sensori e conversione in csv
    dfs = {}
    for f in files:
        df = txt_to_csv(f'dataset/{f}.txt')
        df.sort_values(by=['end_time'], inplace=True)
        if f.find('ADL') == -1:
            df['id'] = df['location'] + df['place'] + df['type']

        # Elimina le righe inconsistenti (finiscono prima di iniziare)
        df.drop(df[df['start_time'] > df['end_time']].index, inplace=True)

        # Conversione dei valori categorici in interi
        if f.find('ADL') >= 0: cols = ['activity']
        else: cols = ['location', 'type', 'place', 'id']
        df[cols] = df[cols].apply(lambda x: x.astype('category'))
        df[cols] = df[cols].apply(lambda x: x.cat.codes)

        # Salva il csv. Just in case
        df.to_csv(f'dataset_csv/{f}.csv', index=False)
        df.reset_index(inplace=True)
        dfs[f] = df

    # Creazione dei due dataset A e B a partire dai csv di attività e sensori
    for f in range(2):
        adl = dfs[files[2 * f]]
        obs = dfs[files[2 * f + 1]]

        # Associa l'attività di ogni sensore ad ogni evento che si è verificato
        # durante l'attività del sensore.pa
        start_date = "2011-11-28 00:00:00" if f == 0 else "2012-11-11 00:00:00"
        end_date = "2011-12-11 23:59:59" if f == 0 else "2012-12-02 23:59:59"
        merged = merge_dataset(adl, obs, start_date, end_date, length=length)

        merged.to_csv(
            f'dataset_csv/Ordonez{"A" if f == 0 else "B"}.csv',
            sep=',', index=False
        )


if __name__ == '__main__':
    main()
