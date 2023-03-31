import pandas as pd
import numpy as np
import time
# cascade 4: 128-146-128
# cas15122: 1168952:0 1168952/1168953:10 1168952/1168953:12 1168952/1168953:6 1168952/1168953:19 1168952/1168953:24 1168952/1168953:28 1168952/1168953:26 1168952/1168953:21 1168952/1168953:17 1168952/1168953:3 1168952/1168953:14
# '122605/122605/1152898:690' '122605/122605:58769' x
# cas78491-num996-hub node

def preprocess_weibo(data_name, cut_time):
    # cascadeID, user_id, publish_time, retweet_number, retweets
    cas_list, src_list, target_list, ts_list, label_list, size_list = [], [], [], [], [], []
    cas_num = 0
    f = open(data_name, 'r')
    while True:
        cas_list_tmp, src_list_tmp, target_list_tmp, ts_list_tmp, label_list_tmp, size_list_tmp = [], [], [], [], [], []
        line = f.readline()
        if not line:
            break
        entry = line.strip().split('\t')
        assert len(entry) == 5
        casid = int(entry[0])
        pub_uid = int(entry[1])
        pub_ts = int(entry[2])
        # 只选择8-18点之间发布的帖子
        hour = int(time.strftime("%H",time.localtime(pub_ts)))
        if hour <= 7 or hour >= 19 :
            continue
        label = int(entry[3])
        paths = entry[4].split(' ')
        paths = sorted(paths, key=lambda x:[int(x.split(':')[1])])
        # print(paths)
        observed_nodes = []
        count_path = 0
        max_len = 0
        # if casid==78491:
        #     print(paths)
        for p in paths:
            p_entry = p.split(':')
            edge_ts = int(p_entry[1])
            # choose retweet before observation time 
            if edge_ts >= cut_time:
                break
            else:
                count_path += 1
            unobserved_flag = False
            if '/' not in p_entry[0]:
                continue
            else:
                node_arr = [int(n) for n in p_entry[0].split('/')]
                for i in range(1, len(node_arr)):
                    if [node_arr[i - 1], node_arr[i]] not in observed_nodes:
                        # 128-146, 128-146-128
                        unobserved_flag = True
                    else:
                        # 1168952/1168953:10 1168952/1168953:12
                        if i == len(node_arr) - 1:
                            unobserved_flag = True
                    if unobserved_flag:
                        observed_nodes.append([node_arr[i - 1], node_arr[i]])
                        cas_list_tmp.append(casid)
                        src_list_tmp.append(node_arr[i - 1])
                        target_list_tmp.append(node_arr[i])
                        ts_list_tmp.append(edge_ts)
                        label_list_tmp.append(label) 
                        size_list_tmp.append(count_path)
                # if len(node_arr) > 2:
                #     print(casid)
        # 删除观测数量小于10或大于1000的级联
        if count_path < 10 or count_path > 1000:
            # print(casid, count_path)
            continue
        else:
            assert len(cas_list_tmp) >= count_path - 1
            cas_list.extend(cas_list_tmp)
            src_list.extend(src_list_tmp)
            target_list.extend(target_list_tmp)
            ts_list.extend(ts_list_tmp)
            label_list.extend(label_list_tmp)
            size_list.extend(size_list_tmp)
            cas_num += 1
    f.close()
    print('cas num: ', cas_num)
    return pd.DataFrame({'cas': cas_list,
                         'src': src_list, 
                         'target': target_list, 
                         'ts': ts_list, 
                         'label': label_list, 
                         'e_idx': size_list})



def run(data_name, cut_time):    
    PATH = '/root/shm/zzz_dataset/{}.txt'.format(data_name)
    OUT_DF = './processed/ml_{}_{}.csv'.format(data_name, cut_time)
    df = preprocess_weibo(PATH, cut_time)
    df.to_csv(OUT_DF)

run('weibo', 10800)