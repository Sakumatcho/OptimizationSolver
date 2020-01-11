"""
複数台の車に乗るメンバーの組み合わせ最適化計算ソルバー

指定可能な制約：
    同乗禁止：同じ車に乗ってはいけないペアの指定
    分散乗車：リストに指定した人が少なくとも一人ずつ全ての車に乗車
    乗車定員：車の台数および各車の乗車定員(リストの要素数がそのまま車の台数となる)

入力ファイル：
    member_list.xlsx：メンバーリスト。計算対象となるメンバー全員の名前が記載されたリスト。
    prohibit_list.xlsx：同乗禁止リスト。メンバーリスト内のメンバーのうち、同じ車に乗ってはいけないペアを記載。
    distribute_list.xlsx：分散乗車リスト。記載されたリストのうち少なくとも一人が全ての車に乗車。
    max_list.xlsx：乗車定員リスト。各車の乗車定員のリスト。リストの要素数がそのまま車の台数。
"""

import sys
from operator import itemgetter
import time
import itertools

import numpy as np
import pandas as pd

"""
前処理
入力ファイルの内容をソルバーに入力可能な形式に変換
"""

# インターフェース
# メンバーリスト
df_member = pd.read_excel('member_list.xlsx')
print('メンバーリスト')
print(df_member)
# 同乗禁止リスト
df_prohibit = pd.read_excel('prohibit_list.xlsx')
print('同乗禁止リスト')
print(df_prohibit)
# 分散乗車リスト
df_distribute = pd.read_excel('distribute_list.xlsx')
print('分散乗車リスト')
print(df_distribute)
# 乗車定員リスト
df_max = pd.read_excel('max_list.xlsx')
print('乗車定員リスト')
print(df_max)

# 入力チェック処理および前処理
if len(df_member) == 0:
    print('メンバーリストを入力してください。')
    sys.exit()

else:
    num_person = len(df_member)
    dict_idx2member = df_member.to_dict()['メンバー']
    dict_member2idx = {val: key for key, val in dict_idx2member.items()}


if len(df_max) == 0:
    print('乗車定員を入力してください。')
    sys.exit()

else:
    arr_car_max = df_max['乗車定員リスト'].values
    
for maximum in arr_car_max:
    if maximum == 0:
        print('乗車定員を1以上で入力してください。')
        sys.exit()    

list_pair_prohibit = list()
if len(df_prohibit.columns) != 2:
        print('同乗禁止リストは2人組みのペアで入力してください。')
        sys.exit()
    
for idx in range(len(df_prohibit)):
    pair_prohibit = df_prohibit.iloc[idx, :]
    if pair_prohibit.isnull().sum() != 0:
        print('同乗禁止リストに未入力があります。')
        sys.exit()
                    
    try: 
        list_pair_prohibit.append([dict_member2idx[name] for name in pair_prohibit.values.tolist()])
        
    except KeyError:
        print('同乗禁止リストにはメンバーリストに存在する名前を入力してください。')
        sys.exit()
        
    
if len(df_distribute) == 1:
    print('分散乗車リストは未入力か、2人以上入力してください。')
    sys.exit()
    
else:
    set_distribute = set([dict_member2idx[name] for name in df_distribute['分散乗車リスト'].values.tolist()])

print('ソルバーへの入力')
print(num_person)
print(dict_idx2member)
print(dict_member2idx)
print(arr_car_max)
print(list_pair_prohibit)
print(set_distribute)


"""
ソルバー本体
"""

def calc_energy(state):
        list_result = [state]
        list_result.append(np.dot(np.dot(state, J), state.T))
        
        return list_result 


start = time.time()
#変数・パラメータ
num_person = num_person
arr_car_max = arr_car_max # np.ndarrayである必要ある？
list_pair_prohibit = list_pair_prohibit
set_distribute = set_distribute

num_car = len(arr_car_max)
num_spin = num_person * num_car
arr_all_member = np.array(range(num_person))

# 同乗禁止リスト
list_pair_prohibit_sorted = list()
for pair in list_pair_prohibit:
    list_pair_prohibit_sorted.append(sorted(pair))
        
list_pair_prohibit_sorted.sort(key=itemgetter(0))

# 相互作用行列の初期化
J = (-1) * np.tri(num_spin, k=-1).T

# 分身禁止制約
for person in range(num_person):
    for car in range(num_car):
        for adj_car in range(num_car - 1 - car):
            J[person * num_car + car][person * num_car + car + adj_car + 1] = 1
            
# 同乗禁止制約
for pair_prohibit in list_pair_prohibit_sorted:
    person_1, person_2 = pair_prohibit
    for car in range(num_car):
        J[person_1 * num_car + car][person_2 * num_car + car] = 1
        

# 全状態生成
spin =  [-1,1]
states = np.array(list(itertools.product(spin, repeat=num_spin)))
print('interaction: \n' + str(J))
print('states: \n' + str(states))
print()

print('経過時間: ' + str(time.time() - start))
print()

list_H = list()
# エネルギー計算
for state in states:
    # 全員乗車制約
    # 改良の余地あり。本当に一人で二代乗車していないか確認すべき。
    if np.sum((state + 1) / 2) == num_person:            
        for car in range(num_car):
            # 乗車定員制約
            if np.sum(state[car::num_car]) > (2 * arr_car_max[car] - num_person):
                # 乗車定員オーバー
                break

        else:
            # 乗車定員制約を全て満たしている場合のみ計算
            # 分散乗車制約
            if len(set_distribute) == 0:
                # 分散乗車制約なし
                list_H.append(calc_energy(state))
                
            else:
                # 分散乗車制約あり                
                for car in range(num_car):
                    # 乗車メンバー(+1が立っている)のインデックスを抽出
                    target = (np.array(state[car::num_car]) + 1) / 2
                    riding = target * (arr_all_member + 1)
                    member = set(riding[riding!=0] - 1)

                    if len(member & set_distribute) == 0:
                        # 分散乗車制約違反
                        break

                else:
                    # 分散乗車対象メンバーが少なくとも一人ずつ各車に乗車している場合のみ計算
                    list_H.append(calc_energy(state))

        
print('経過時間: ' + str(time.time() - start))
print()

print('計算結果')
df_result = pd.DataFrame(np.array(list_H), columns=['スピン','値'])
print(df_result)
print()

print('経過時間: ' + str(time.time() - start))
print()

print('最小値')
val_min = df_result['値'].min()
print(val_min)
print()

print('経過時間: ' + str(time.time() - start))
print()

print('最小値を取る場合')
query = df_result['値'] == val_min
df_stable = df_result[query]
display(df_stable)
print()

elapsed_time = time.time() - start
print(' 総計算時間: ' + str(elapsed_time))

"""
後処理
計算結果を人間が読み取れる形式に逆変換
"""

func_idx2member = np.frompyfunc(lambda x: dict_idx2member[x], 1, 1)

list_col = ['状態No.', '車No.'] + [f'{i}人目' for i in range(np.max(arr_car_max))]
df_output = pd.DataFrame(columns=list_col)
for idx in range(len(df_stable)):
    state = df_stable.iloc[idx, :]['スピン']
    for car in range(num_car):
        target = (np.array(state[car::num_car]) + 1) / 2
        riding = target * (arr_all_member + 1)
        member = func_idx2member(riding[riding!=0] - 1)
        df_output = df_output.append(pd.Series([idx, car] + member.tolist(), index=list_col), ignore_index=True)
        

# 計算結果の出力
df_output.to_csv('output.csv', encoding='utf8')

