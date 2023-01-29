
import time
import datetime
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import random
import os

def pad_zero(data, max_length):
    if len(data) >= max_length:
        return data[-max_length:]
    else:
        return [0] * (max_length - len(data)) + data


def preprocess(dataset, purchase_file, cart_file, fav_file, click_file, num_interactions=10, max_purchase_length=10, max_cart_length=10, max_fav_length=10, max_click_length=10):
    
    dataset_path = "/data/junsu7463/" + dataset + "/"
    if os.path.exists(dataset_path + 'data_' + str(num_interactions) + '_' + str(max_purchase_length) + '_' + str(max_cart_length) + '_' + str(max_fav_length) + '_' + str(max_click_length) + '.pkl'):
        with open(dataset_path + 'data_' + str(num_interactions) + '_' + str(max_purchase_length) + '_' + str(max_cart_length) + '_' + str(max_fav_length) + '_' + str(max_click_length) + '.pkl', 'rb') as f:
            tr_purchase_labels, tr_purchase_sequences, tr_cart_sequences, tr_fav_sequences, tr_click_sequences, \
            va_purchase_labels, va_purchase_sequences, va_cart_sequences, va_fav_sequences, va_click_sequences, \
            te_purchase_labels, te_purchase_sequences, te_cart_sequences, te_fav_sequences, te_click_sequences, \
            num_users, num_items = pickle.load(f)
    else:
        purchase_data = pd.read_csv(purchase_file)
        cart_data = pd.read_csv(cart_file)
        fav_data = pd.read_csv(fav_file)
        click_data = pd.read_csv(click_file)

        user_key = 'userid' 
        item_key = 'itemid'
        time_key = 'timestamp'
        

        # Sort by time
        purchase_data.sort_values([user_key, time_key], ascending=[True, True], inplace=True)
        cart_data.sort_values([user_key, time_key], ascending=[True, True], inplace=True)
        fav_data.sort_values([user_key, time_key], ascending=[True, True], inplace=True)
        click_data.sort_values([user_key, time_key], ascending=[True, True], inplace=True)

        
        # 0 is for padding
        purchase_data['userid'] += 1
        purchase_data['itemid'] += 1
        cart_data['userid'] += 1
        cart_data['itemid'] += 1
        fav_data['userid'] += 1
        fav_data['itemid'] += 1
        click_data['userid'] += 1
        click_data['itemid'] += 1

        num_users = max(purchase_data[user_key])
        num_items = max(max(purchase_data[item_key]), max(click_data[item_key]), max(cart_data[item_key]))

        purchase_data = purchase_data.to_numpy().tolist()
        cart_data = cart_data.to_numpy().tolist()
        fav_data = fav_data.to_numpy().tolist()
        click_data = click_data.to_numpy().tolist()

        tr_purchase_labels = []
        tr_purchase_sequences = []
        tr_cart_sequences = []
        tr_fav_sequences = []
        tr_click_sequences = []

        va_purchase_labels = []
        va_purchase_sequences = []
        va_cart_sequences = []
        va_fav_sequences = []
        va_click_sequences = []

        te_purchase_labels = []
        te_purchase_sequences = []
        te_cart_sequences = []
        te_fav_sequences = []
        te_click_sequences = []
        

        num_purchase = len(purchase_data)
        num_cart = len(cart_data)
        num_fav = len(fav_data)
        num_click = len(click_data)

        user_purchase_labels = []
        user_purchase_data = []
        user_cart_data = []
        user_fav_data = []
        user_click_data = []

        user_purchase_start_i = 0
        user_cart_start_i = 0
        user_fav_start_i = 0
        user_click_start_i = 0

        purchase_i = 0
        cart_i = 0
        fav_i = 0
        click_i = 0

        last_purchase_i = 0
        last_cart_i = 0
        last_fav_i = 0
        last_click_i = 0

        uid = -1
        cnt = 0
        for i, (uid, iid, action, timestamp) in tqdm(enumerate(purchase_data), total=len(purchase_data)):
            assert action == 3 ## Purchase


            ## purchase
            while True:
                if purchase_i > len(purchase_data)-1:
                    last_purchase_i = purchase_i - 1
                    break
                if uid != purchase_data[purchase_i][0]:
                    last_purchase_i = purchase_i - 1
                    break
                purchase_time = purchase_data[purchase_i][3]
                if purchase_time >= timestamp:
                    last_purchase_i = purchase_i - 1
                    break 
                purchase_i += 1
            purchase_so_far = purchase_data[max(user_purchase_start_i, (last_purchase_i+1) - (max_purchase_length)) : last_purchase_i+1] # max_purchase_length+3
            purchase_so_far = list(map(lambda x: x[1], purchase_so_far))
        
            assert len(purchase_so_far) <= max_purchase_length


            ## Cart
            while True:
                if cart_i > len(cart_data)-1:
                    last_cart_i = cart_i - 1
                    break
                if uid != cart_data[cart_i][0]:
                    last_cart_i = cart_i - 1
                    break
                cart_time = cart_data[cart_i][3]
                if cart_time >= timestamp:
                    last_cart_i = cart_i - 1
                    break
                cart_i += 1
            cart_so_far = cart_data[max(user_cart_start_i, (last_cart_i+1) - (max_cart_length)) : last_cart_i+1] # max_cart_length+3
            cart_so_far = list(map(lambda x: x[1], cart_so_far))
            
            assert len(cart_so_far) <= max_cart_length


            ## Fav
            while True:
                if fav_i > len(fav_data)-1:
                    last_fav_i = fav_i - 1
                    break
                if uid != fav_data[fav_i][0]:
                    last_fav_i = fav_i - 1
                    break
                fav_time = fav_data[fav_i][3]
                if fav_time >= timestamp:
                    last_fav_i = fav_i - 1
                    break
                fav_i += 1
            fav_so_far = fav_data[max(user_fav_start_i, (last_fav_i+1) - (max_fav_length)) : last_fav_i+1] # max_fav_length+3
            fav_so_far = list(map(lambda x: x[1], fav_so_far))
            
            assert len(fav_so_far) <= max_fav_length


            ## click
            while True:
                if click_i > len(click_data)-1:
                    last_click_i = click_i - 1
                    break
                if uid != click_data[click_i][0]:
                    last_click_i = click_i - 1
                    break
                click_time = click_data[click_i][3]
                if click_time >= timestamp:
                    last_click_i = click_i - 1
                    break
                click_i += 1
            click_so_far = click_data[max(user_click_start_i, (last_click_i+1) - (max_click_length)) : last_click_i+1] # max_click_length+3
            click_so_far = list(map(lambda x: x[1], click_so_far))
            
            assert len(click_so_far) <= max_click_length

            user_purchase_labels.append(iid)
            user_purchase_data.append(pad_zero(list(purchase_so_far), max_purchase_length))
            user_click_data.append(pad_zero(list(click_so_far), max_click_length))
            user_cart_data.append(pad_zero(list(cart_so_far), max_cart_length))
            user_fav_data.append(pad_zero(list(fav_so_far), max_fav_length))
            

            if i == len(purchase_data)-1 or purchase_data[i+1][0] != uid: # next is new user

                if len(user_purchase_labels) > num_interactions+2:
                    user_purchase_labels = user_purchase_labels[-(num_interactions+2) :]
                    user_purchase_data = user_purchase_data[-(num_interactions+2) :]
                    user_cart_data = user_cart_data[-(num_interactions+2) :]
                    user_fav_data = user_fav_data[-(num_interactions+2) :]
                    user_click_data = user_click_data[-(num_interactions+2) :]

                start_ind = 0
                for k in range(len(user_purchase_labels)-2):
                    if sum(user_purchase_data[k]) == 0 and sum(user_cart_data[k]) == 0 and sum(user_fav_data[k]) == 0 and sum(user_click_data[k]) == 0:
                        start_ind += 1 
                    else: break


                tr_purchase_labels += list(user_purchase_labels[start_ind:-2]) # ?
                tr_purchase_sequences += list(user_purchase_data[start_ind:-2]) # ? * l
                tr_cart_sequences += list(user_cart_data[start_ind:-2])
                tr_fav_sequences += list(user_fav_data[start_ind:-2])
                tr_click_sequences += list(user_click_data[start_ind:-2])

                va_purchase_labels.append(user_purchase_labels[-2]) # 1
                va_purchase_sequences.append(list(user_purchase_data[-2])) # l
                va_cart_sequences.append(list(user_cart_data[-2]))
                va_fav_sequences.append(list(user_fav_data[-2]))
                va_click_sequences.append(list(user_click_data[-2]))

                te_purchase_labels.append(user_purchase_labels[-1]) # 1
                te_purchase_sequences.append(list(user_purchase_data[-1])) # l
                te_cart_sequences.append(list(user_cart_data[-1]))
                te_fav_sequences.append(list(user_fav_data[-1]))
                te_click_sequences.append(list(user_click_data[-1]))

                
                purchase_i = i + 1
                cart_i = last_cart_i + 1
                fav_i = last_fav_i + 1
                click_i = last_click_i + 1

                user_purchase_start_i = i+1
                last_purchase_i = i + 1
                user_cart_start_i = last_cart_i+1
                last_cart_i = last_cart_i + 1
                user_fav_start_i = last_fav_i+1
                last_fav_i = last_fav_i + 1
                user_click_start_i = last_click_i+1
                last_click_i = last_click_i + 1

                user_purchase_labels = []
                user_purchase_data = []
                user_click_data = []
                user_cart_data = []
                user_fav_data = []
            
        with open(dataset_path + 'data_' + str(num_interactions) + '_' + str(max_purchase_length) + '_' + str(max_cart_length) + '_' + str(max_fav_length) + '_' + str(max_click_length) + '.pkl', 'wb') as f:
            pickle.dump([tr_purchase_labels, tr_purchase_sequences, tr_cart_sequences, tr_fav_sequences, tr_click_sequences, 
                va_purchase_labels, va_purchase_sequences, va_cart_sequences, va_fav_sequences, va_click_sequences, 
                te_purchase_labels, te_purchase_sequences, te_cart_sequences, te_fav_sequences, te_click_sequences, 
                num_users, num_items], f)
        

    return tr_purchase_labels, tr_purchase_sequences, tr_cart_sequences, tr_fav_sequences, tr_click_sequences, \
            va_purchase_labels, va_purchase_sequences, va_cart_sequences, va_fav_sequences, va_click_sequences, \
            te_purchase_labels, te_purchase_sequences, te_cart_sequences, te_fav_sequences, te_click_sequences, \
            num_users, num_items

