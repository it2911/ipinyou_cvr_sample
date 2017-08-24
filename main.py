import ctr_nn.utils as utils
import numpy as np
from ctr_nn.models import LR, FM, PNN1, PNN2, FNN, CCPM
from sklearn.metrics import roc_auc_score

# ↓↓↓↓↓↓↓　グローバル変数設定　↓↓↓↓↓↓↓↓
train_file = '../data/train.yx.txt'
test_file = '../data/test.yx.txt'
# fm_model_file = '../data/fm.model.txt'

# DIMの設定
input_dim = utils.INPUT_DIM

# データを読み込む
train_data = utils.read_data(train_file)
# データをシャッフルする
train_data = utils.shuffle(train_data)
# 検証用データを読み込む
test_data = utils.read_data(test_file)

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')

# データセットのデータ件数
train_size = train_data[0].shape[0]
# 検証用データセットのデータ件数
test_size = test_data[0].shape[0]
# 特徴のデータ件数
num_feas = len(utils.FIELD_SIZES)

# 最小ロール回数
min_round = 1
# 最大ロール回数
num_round = 1000
# 50回ロールして、loss値が変わらない場合、プログラムが中止
early_stop_round = 50
batch_size = 1024

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS
# ↑↑↑↑↑↑↑　グローバル変数設定　↑↑↑↑↑↑↑↑

# Train
def train(model):
    history_score = []
    for i in range(num_round):
        # Tensorflow で loss value を計算する結果に対して、チューニングする
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = model.run(model.y_prob, utils.slice(train_data)[0])
        test_preds = model.run(model.y_prob, utils.slice(test_data)[0])
        # AUCで予測結果を評価する
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)

        if i > min_round and i > early_stop_round:
            # np.argmax 最大値となる要素のインデックス
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[-1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break


algo = 'pnn2'

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0
    }

    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }

    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'random_seed': 0
    }

    model = FNN(**fnn_params)
elif algo == 'ccpm':
    ccpm_params = {
        'layer_sizes': [field_sizes, 10, 5, 3],
        'layer_acts': ['tanh', 'tanh', 'none'],
        'drop_out': [0, 0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': 0
    }

    model = CCPM(**ccpm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN1(**pnn1_params)
elif algo == 'pnn2':
    pnn2_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN2(**pnn2_params)

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)

train(model)

# X_i, y_i = utils.slice(train_data, 0, 100)
# fetches = [model.tmp1, model.tmp2]
# tmp1, tmp2 = model.run(fetches, X_i, y_i)
# print tmp1.shape
# print tmp2.shape
