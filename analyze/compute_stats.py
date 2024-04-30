import pandas as pd
from icecream import ic
import numpy as np

df = pd.read_csv('results.csv')
methods = df.Model_name.unique()
ic(methods)
# methods = methods[0:3]
# white_BG_names = ['white_BG_bigClamp', 'white_BG_paint', 'white_BG_jelloRed', 'white_BG_yellow2']
# lab_BG_names = ['lab_BG_bigClamp', 'lab_BG_paint', 'lab_BG_jelloRed', 'lab_BG_yellow2']
# peg_names = ['square_peg']
# wipe_names = ['wipe_plate_16']

white_BG_names = ['white_BG_bigClamp_slow', 'white_BG_paint_slow', 'white_BG_jelloRed_slow', 'white_BG_yellow3_slow2']
lab_BG_names = ['lab_BG_bigClamp_slow', 'lab_BG_paint_slow', 'lab_BG_jelloRed_slow', 'lab_BG_yellow3_slow']
peg_names = ['square_peg_smooth2']
wipe_names = ['wipe_plate_16']

counter = 0

white_seeds = []
lab_seeds = []
peg_seeds = []
wipe_seeds = []
white_seeds_1 = []
lab_seeds_1 = []
peg_seeds_1 = []
wipe_seeds_1 = []
white_seeds_2 = []
lab_seeds_2 = []
peg_seeds_2 = []
wipe_seeds_2 = []
white_seeds_2plus = []
lab_seeds_2plus = []
peg_seeds_2plus = []
wipe_seeds_2plus = []
for shift in [0]: #[1, 2, 3, 4, 5, 6]:
    for method in methods:
        print(method)
        white_BG_total = 0
        lab_BG_total = 0
        peg_total = 0
        wipe_total = 0
        white_BG_total_1 = 0
        lab_BG_total_1 = 0
        peg_total_1 = 0
        wipe_total_1 = 0
        white_BG_total_2, white_BG_total_N_2 = 0,0
        lab_BG_total_2, lab_BG_total_N_2 = 0,0
        peg_total_2, peg_total_N_2 = 0,0
        wipe_total_2, wipe_total_N_2 = 0,0
        white_BG_total_2plus, white_BG_total_N_2plus = 0,0
        lab_BG_total_2plus, lab_BG_total_N_2plus = 0,0
        peg_total_2plus, peg_total_N_2plus = 0,0
        wipe_total_2plus, wipe_total_N_2plus = 0,0

        for name in white_BG_names:
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE'].iloc[0]
            
            white_BG_total += value
            white_BG_total_1 += df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_1'].iloc[0]
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2'].iloc[0]
            white_BG_total_2 += value
            if value > 0:
                white_BG_total_N_2 += 1
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2plus'].iloc[0]
            white_BG_total_2plus += value
            if value > 0:
                white_BG_total_N_2plus += 1

        for name in lab_BG_names:
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE'].iloc[0]
            lab_BG_total += value
            lab_BG_total_1 += df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_1'].iloc[0]
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2'].iloc[0]
            lab_BG_total_2 += value
            if value > 0:
                lab_BG_total_N_2 += 1
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2plus'].iloc[0]
            lab_BG_total_2plus += value
            if value > 0:
                lab_BG_total_N_2plus += 1
        for name in peg_names:
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE'].iloc[0]
            peg_total += value
            peg_total_1 += df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_1'].iloc[0]
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2'].iloc[0]
            peg_total_2 += value
            if value > 0:
                peg_total_N_2 += 1
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2plus'].iloc[0]
            peg_total_2plus += value
            if value > 0:
                peg_total_N_2plus += 1
        for name in wipe_names:
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE'].iloc[0]
            wipe_total += value
            wipe_total_1 += df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_1'].iloc[0]
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2'].iloc[0]
            wipe_total_2 += value
            if value > 0:
                wipe_total_N_2 += 1
            value = df[df['Model_name'].isin([method]) & df['Problem'].isin([name])& df['shift'].isin([shift])]['avg_MAE_2plus'].iloc[0]
            wipe_total_2plus += value
            if value > 0:
                wipe_total_N_2plus += 1


        white_BG_avg = white_BG_total/len(white_BG_names)
        lab_BG_avg = lab_BG_total/len(lab_BG_names)
        peg_avg = peg_total/len(peg_names)
        wipe_avg =wipe_total/len(wipe_names)

        white_BG_avg_1 = white_BG_total_1/len(white_BG_names)
        lab_BG_avg_1 = lab_BG_total_1/len(lab_BG_names)
        peg_avg_1 = peg_total_1/len(peg_names)
        wipe_avg_1 =wipe_total_1/len(wipe_names)

        white_BG_avg_2= white_BG_total_2/white_BG_total_N_2
        lab_BG_avg_2 = lab_BG_total_2/lab_BG_total_N_2
        if peg_total_N_2 > 0:
            peg_avg_2 = peg_total_2/peg_total_N_2
        else:
            peg_avg_2 = None
        wipe_avg_2 =wipe_total_2/wipe_total_N_2

        white_BG_avg_2plus = white_BG_total_2plus/white_BG_total_N_2plus
        lab_BG_avg_2plus = lab_BG_total_2plus/lab_BG_total_N_2plus
        if peg_total_N_2plus > 0:
            peg_avg_2plus = peg_total_2plus/peg_total_N_2plus
        else:
            peg_avg_2plus = None
        try:
            wipe_avg_2plus =wipe_total_2plus/wipe_total_N_2plus
        except:
            wipe_avg_2plus = None

        white_seeds.append(white_BG_avg)
        lab_seeds.append(lab_BG_avg)
        peg_seeds.append(peg_avg)
        wipe_seeds.append(wipe_avg)

        white_seeds_1.append(white_BG_avg_1)
        lab_seeds_1.append(lab_BG_avg_1)
        peg_seeds_1.append(peg_avg_1)
        wipe_seeds_1.append(wipe_avg_1)

        white_seeds_2.append(white_BG_avg_2)
        lab_seeds_2.append(lab_BG_avg_2)
        if peg_avg_2plus is not None:
            peg_seeds_2.append(peg_avg_2)
        wipe_seeds_2.append(wipe_avg_2)

        white_seeds_2plus.append(white_BG_avg_2plus)
        lab_seeds_2plus.append(lab_BG_avg_2plus)
        if peg_avg_2plus is not None:
            peg_seeds_2plus.append(peg_avg_2plus)
        if wipe_avg_2plus is not None:
            wipe_seeds_2plus.append(wipe_avg_2plus)

        if counter%3 == 2:
            ic(shift)
            # ic(white_seeds)
            ic(np.average(white_seeds), np.std(white_seeds))
            ic(np.average(lab_seeds), np.std(lab_seeds))
            ic(np.average(peg_seeds), np.std(peg_seeds))
            ic(np.average(wipe_seeds), np.std(wipe_seeds))

            ic(np.average(white_seeds_1))
            ic(np.average(lab_seeds_1))
            ic(np.average(peg_seeds_1))
            ic(np.average(wipe_seeds_1))
            
            ic(np.average(white_seeds_2))
            ic(np.average(lab_seeds_2))
            ic(np.average(peg_seeds_2))
            ic(np.average(wipe_seeds_2))

            ic(np.average(white_seeds_2plus))
            ic(np.average(lab_seeds_2plus))
            ic(np.average(peg_seeds_2plus))
            ic(np.average(wipe_seeds_2plus))

            white_seeds = []
            lab_seeds = []
            peg_seeds = []
            wipe_seeds = []
            
            white_seeds_1 = []
            lab_seeds_1 = []
            peg_seeds_1 = []
            wipe_seeds_1 = []
            white_seeds_2 = []
            lab_seeds_2 = []
            peg_seeds_2 = []
            wipe_seeds_2 = []
            white_seeds_2plus = []
            lab_seeds_2plus = []
            peg_seeds_2plus = []
            wipe_seeds_2plus = []

        counter += 1
