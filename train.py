import numpy as np
import tensorflow as tf
from models import MsGAT
from utils import process
from tool.smalibat import docompile
from tool.process_permission_2_list import process_permission
from tool.generate_final_Matrix import get_Matrix
from tool.process_permission_2_Matrix import permission_Matrix
import os
import scipy.io as scio

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.flags.DEFINE_string(
    "Benign_apk_dir", "original_Benign_apk", "The path to hold the Benign apks",
)
tf.flags.DEFINE_string(
    "Mal_apk_dir", "original_Mal_apk", "The path to hold the malwares",
)
tf.flags.DEFINE_string(
    "Benign_decompile_dir", "decompile_Benign_result", "The path to hold the decompile apks",
)
tf.flags.DEFINE_string(
    "Mal_decompile_dir", "decompile_Mal_result", "The path to hold the decompile apks",
)
tf.flags.DEFINE_string(
    "per_name", "data/per_name", "The path to hold the permission name",
)
tf.flags.DEFINE_string(
    "Benign_per_matrix", "data/per_matrix.mat"
)
tf.flags.DEFINE_string(
    "Mal_per_matrix", "data/mal_matrix.mat"
)
tf.flags.DEFINE_string(
    "Per_matrix", "data/per_matrix.mat"
)
FLAGS = tf.flags.FLAGS

#################################data process#######################################

docompile(FLAGS.Benign_apk_dir, FLAGS.Benign_decompile_dir)
docompile(FLAGS.Mal_apk_dir, FLAGS.Mal_decompile_dir)
print("Finish decompilation!")
process_permission(FLAGS.Benign_apk_dir, FLAGS.Mal_apk_dir, FLAGS.per_name)
permission_Matrix(FLAGS.per_name,FLAGS.Benign_apk_dir,FLAGS.Benign_per_matrix,True)
permission_Matrix(FLAGS.per_name,FLAGS.Mal_apk_dir,FLAGS.Mal_per_matrix,False)
get_Matrix(FLAGS.Benign_per_matrix,FLAGS.Mal_per_matrix,FLAGS.Per_matrix)

checkpt_file = 'matrix/'
label_file = 'labels/'

#################################model#######################################

batch_size = 1
nb_epochs = 20
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [16, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    with open(checkpt_file + 'output.txt', 'a+') as fi:
        fi.write("We have GPU"+ '\n')
else:
    print("Please install GPU version of TF")
    with open(checkpt_file + 'output.txt', 'a+') as fi:
        fi.write("Please install GPU version of TF"+ '\n')


print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

l = len(FLAGS.Per_matrix)
adj_list= scio.loadmat(FLAGS.Per_matrix)['permission']
features = np.eye(l)
fea_list = [features]

print('features.shape:', features.shape)
print('adj_list.shape:', adj_list.shape)
print('fea_list.shape:', np.array(fea_list).shape)

lastlabel = np.load(label_file+'one_hot_labels.npy')
train_data = np.load(label_file+'train_idx.npy')
print(train_data)

test_data = np.load(label_file+'test_idx.npy')
print("Test data",test_data)

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print('alllabel_shape:',lastlabel.shape)
print('train_data_shape:',train_data.shape)
print('test_data_shape:',test_data.shape)
print('train_size:',train_size)
print('test_size:',test_size)

train_idx = np.zeros([1,train_size],dtype=int)
test_idx = np.zeros([1,test_size],dtype=int)
train_idx[0]=train_data
test_idx[0]=test_data

train_mask = sample_mask(train_idx, lastlabel.shape[0])
test_mask = sample_mask(test_idx, lastlabel.shape[0])

y_train = np.zeros(lastlabel.shape)
y_test = np.zeros(lastlabel.shape)

y_train[train_mask, :] = lastlabel[train_mask, :]
y_test[test_mask, :] = lastlabel[test_mask, :]
print('y_train:{}, y_test:{}, train_mask:{}, test_mask:{}'.format(y_train.shape,
                                                                        y_test.shape,
                                                                        train_mask.shape,
                                                                        test_mask.shape))

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

print('fea_list:{},adj_list:{},y_train:{}, y_test:{}, train_mask:{}, test_mask:{}'.format(np.array(fea_list).shape,
                                                                                                                 np.array(adj_list).shape,
                                                                                                                 y_train.shape,
                                                                                                                 y_test.shape,
                                                                                                                 train_mask.shape,
                                                                                                                 test_mask.shape))

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]
print('biases_list:',np.array(biases_list).shape)

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        print('ftr_in_list:',np.array(ftr_in_list).shape)
        # bias_in_list [1,N,N]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        print('bias_in_list:',np.array(bias_in_list).shape)
        # lbl_in [1，N，c]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        # msk_in [1,N]
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val, r2= model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        max_train_acc = -1
        max_test_acc = -1

        train_loss_avg = 0
        train_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            print('Training: loss = %.8f, acc = %.8f ' %(train_loss_avg / tr_step, train_acc_avg / tr_step))

            train_loss_avg = 0
            train_acc_avg = 0

            with open(checkpt_file+'output.txt','a+') as fi:
                fi.write(str(train_acc_avg / tr_step)+'\n')

            if (train_acc_avg / tr_step) > max_train_acc:
                max_train_acc = train_acc_avg / tr_step
                with open(checkpt_file+'output.txt','a+') as fi:
                    fi.write('best Acc：'+str(max_train_acc)+'\n')

            ts_size = fea_list[0].shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
                fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                       msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],

                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                      feed_dict=fd)
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print(epoch)
            print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)
            if(epoch>5):
                if (ts_acc / ts_step) > max_test_acc:
                    max_test_acc = ts_acc / ts_step
                    with open(checkpt_file+'output.txt','a+',) as fi:
                        fi.write('Best Test Acc：'+str(max_test_acc)+'\n')
                    saver.save(sess, checkpt_file, global_step=epoch)
                    feas = np.array(jhy_final_embedding)
                    np.save('emb/feats.npy', feas)
        sess.close()
