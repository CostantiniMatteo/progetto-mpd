import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)

def find_state(timestamp, states):
    l = np.asarray(states)
    idx = (np.abs(l - timestamp)).argmin()
    return states[idx]


if __name__ == '__main__':
    datasets_state = ['OrdonezA_ADLs.csv', 'OrdonezB_ADLs.csv']
    datasets_obs = ['OrdonezA_Sensors.csv', 'OrdonezB_Sensors.csv']

    states = pd.read_csv(f'dataset_csv/{datasets_state[0]}')
    obs = pd.read_csv(f'dataset_csv/{datasets_obs[0]}')

    states['mean_timestamp'] = (states['start_time'] + states['end_time']) // 2
    obs['mean_timestamp'] = (obs['start_time'] + obs['end_time']) // 2

    obs['state_timestamp'] = 0
    for i, row in obs.iterrows():
        obs.set_value(i, 'state_timestamp',
            find_state(row[5], states['mean_timestamp'])
        )

    merged = pd.merge(obs, states,
        left_on='state_timestamp',right_on='mean_timestamp'
    )
    merged.drop(columns=['state_timestamp'], axis=1, inplace=True)
    merged.columns = ['start_time_sensor', 'end_time_sensor', 'location',
        'type', 'place', 'mean_sensor', 'start_time_activity',
        'end_time_activity', 'activity', 'mean_activity'
    ]

    merged.to_csv('dataset_csv/full.csv', sep=',', index=False)
