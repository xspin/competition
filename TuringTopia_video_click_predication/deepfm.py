import gc
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.callbacks as kc
import tensorflow.keras.backend as kb
import tensorflow.keras.regularizers as kr
import tensorflow.keras.optimizers as ko
from tensorflow.keras.utils import plot_model
import utils
import logging
import time
import glob
from sklearn import metrics
tf.get_logger().setLevel('INFO')
tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S')
class Config:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 5
        self.ckpt_dir = 'ckpt'
        self.embedding_method = 'random'
        self.is_debug = True
        self.data_dir = 'dataset'
        self.load_weights = False
        self.deep_layers = [32, 32]

def Attention(input_dim):
    def func(inputs):
        # a = kl.Permute([2,1])(inputs)
        probs = kl.Dense(input_dim, activation='softmax')(inputs)
        # probs = kl.Permute([2,1])(a)
        outputs = kl.Multiply()([inputs, probs])
        return outputs, probs
    return func

def MLP(shapes, drop_rate, activation='relu', name='MLP'):
    def func(x):
        if len(shapes)==0: return x
        for i,n in enumerate(shapes):
            x = kl.Dense(n, activation=activation, name='{}-{}'.format(name, i))(x)
            if drop_rate>1e-4:
                x = kl.Dropout(rate=drop_rate)(x)
        return x
    return func

def Classifier(shapes, drop_rate=0, activation='relu' , name='Classifier'):
    def func(x):
        x = MLP(shapes[:-1], drop_rate, activation=activation, name='%s_MLP'%name)(x)
        act = 'sigmoid' if shapes[-1]==1 else 'softmax'
        x = kl.Dense(shapes[-1], activation=act, name='{}_output'.format(name))(x)
        return x
    return func 

def Discriminator(**args):
    if 'name' not in args:
        args['name'] = 'Discriminator'
    return Classifier(**args)

class Model:
    def __init__(self, feat_dim, input_dim, deep_layers = [32, 32]):
        print(f'Building deepfm model with deep layers: {deep_layers} ...')
        self.ph_x_neg_i = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_x_neg_i')
        self.ph_x_neg_v = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_x_neg_v')
        self.ph_x_pos_i = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_x_pos_i')
        self.ph_x_pos_v = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_x_pos_v')
        self.ph_x_test_i = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_test_i')
        self.ph_x_test_v = tf.placeholder(tf.float32, shape=[None, input_dim], name='PH_test_v')
        # self.ph_y_dis = tf.placeholder(tf.int32, shape=[None, 1], name='PH_y_dis')
        self.ph_y = tf.placeholder(tf.int32, shape=[None, 1], name='PH_y')
        self.ph_size = tf.placeholder(tf.int32, name='PH_size')
        self.drop_rate = tf.placeholder(tf.float32, name='PH_drop_rate')
        self.xi = tf.concat([self.ph_x_neg_i, self.ph_x_pos_i, self.ph_x_test_i], axis=0, name='Concat_i')
        self.xv = tf.concat([self.ph_x_neg_v, self.ph_x_pos_v, self.ph_x_test_v], axis=0, name='Concat_v')

        embedding_size = int(np.log2(feat_dim)) + 1
        initializer1 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2020)
        initializer2 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2021)
        Embedding =  kl.Embedding(feat_dim, embedding_size, 
                        embeddings_initializer=initializer1
                     )
        Embedding_bias =  kl.Embedding(feat_dim, 1, 
                        embeddings_initializer=initializer2
                     )

        input_idx = kl.Input(tensor=self.xi, name='Input_idx')
        input_val = kl.Input(tensor=self.xv, name='Input_val')
        embeddings = Embedding(input_idx)
        # feat_value2 = tf.reshape(feat_val, shape=[-1, field_size, 1])
        feat_val = kl.Lambda(lambda x:tf.expand_dims(x, axis=-1))(input_val)
        embeddings = kl.Multiply()([embeddings, feat_val])
        # embeddings = kl.Lambda(lambda x:tf.multiply(*x))([embeddings, feat_val])

        # ---------- first order term ----------
        y_first_order = Embedding_bias(input_idx)
        y_first_order = kl.Multiply()([y_first_order, feat_val])
        y_first_order = kl.Lambda(lambda x:tf.reduce_sum(x, 2))(y_first_order)  # None * F
        y_first_order = kl.Dropout(rate=self.drop_rate)(y_first_order)

        # ---------- second order term ---------------
        # sum_square part
        summed_features_emb = kl.Lambda(lambda x:tf.reduce_sum(x, 1))(embeddings)  # None * K
        summed_features_emb_square = kl.Lambda(tf.square)(summed_features_emb)  # None * K

        # square_sum part
        squared_features_emb = kl.Lambda(tf.square)(embeddings)
        squared_sum_features_emb = kl.Lambda(lambda x:tf.reduce_sum(x, 1))(squared_features_emb)  # None * K

        # second order
        y_second_order = kl.Lambda(lambda x:0.5*tf.subtract(*x))([summed_features_emb_square, squared_sum_features_emb])  # None * K
        y_second_order = kl.Dropout(rate=self.drop_rate)(y_second_order)  # None * K

        # ---------- Deep component ----------
        # y_deep = tf.reshape(embeddings, shape=[-1, input_dim * embedding_size]) # None * (F*K)
        y_deep = kl.Flatten()(embeddings)
        y_deep = kl.Dropout(rate=self.drop_rate)(y_deep)
        for units in deep_layers:
            y_deep = kl.Dense(units)(y_deep)
            y_deep = kl.BatchNormalization()(y_deep)
            y_deep = kl.Activation(activation='relu')(y_deep)
            y_deep = kl.Dropout(rate=self.drop_rate)(y_deep)

        # ---------- DeepFM ----------
        # concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
        concat_input = kl.Concatenate(axis=1)([y_first_order, y_second_order, y_deep])
        concat_input = kl.Dropout(rate=self.drop_rate)(concat_input)

        y_pred_dis = Discriminator(shapes=[8,1], drop_rate=0, activation='relu', name='Disc')(concat_input)

        concat_input = kl.Lambda(lambda x: x[:self.ph_size])(concat_input)
        # self.y_pred = kl.Dense(1, activation='sigmoid', name='Output')(concat_input)
        self.y_pred = Classifier(shapes=[8,1], activation='relu', name='Output')(concat_input)
        self.loss = tf.losses.log_loss(labels=self.ph_y, predictions=self.y_pred)

        y_dis = tf.concat([tf.zeros([self.ph_size,1], dtype='int32'), tf.ones([self.ph_size,1], dtype='int32')], axis=0)
        self.loss_dis = tf.losses.log_loss(labels=y_dis, predictions=y_pred_dis)
        # Optimization of FE and Classifier
        alpha = 0.4
        self.loss_total = alpha*self.loss - (1-alpha)*self.loss_dis

        self.model = km.Model([input_val, input_idx], [self.y_pred, y_pred_dis])

        opt = tf.train.AdamOptimizer(name='Adam')
        self.train_class()
        self.op_optimize = opt.minimize(self.loss_total, var_list=self.model.trainable_weights)

        # Optimization of Discriminator
        self.train_disc()
        self.op_optimize_dis = opt.minimize(self.loss_dis, var_list=self.model.trainable_weights)
    
        self.acc = self.binary_acc(self.ph_y, self.y_pred)
        self.acc_dis = self.binary_acc(y_dis, y_pred_dis)

    def train_class(self):
        for layer in self.model.layers:
            if 'Disc' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True

    def train_disc(self):
        for layer in self.model.layers:
            if 'Disc' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

    def binary_acc(self, y_true, y_pred):
        y_pred_cmp = tf.math.greater(y_pred, tf.constant(0.5))
        y_pred_lb = tf.cast(y_pred_cmp, tf.int32)
        acc_cnt = tf.equal(y_pred_lb, y_true)
        acc = tf.reduce_mean(tf.cast(acc_cnt, tf.float32))
        return acc

    def best_f1_thr(self, y_true, y_score):
        t0 = 0.05
        v = 0.002
        best_t = t0
        best_f1 = 0
        cnt_no_inc = 0
        for step in range(201):
            curr_t = t0 + step * v
            y = [1 if x >= curr_t else 0 for x in y_score]
            curr_f1 = metrics.f1_score(y_true, y_score)
            if curr_f1 > best_f1:
                best_t = curr_t
                best_f1 = curr_f1
                cnt_no_inc = 0
            else:
                cnt_no_inc += 1
                if cnt_no_inc > 15:
                    break
        return best_f1, best_t

    def predict(self, sess, xdata, batch_size=64):
        y_score = []
        for x_batch in utils.gen_batch2(xdata, batch_size, drop_last=False, one_epoch=True, shuffle=False):
            n = len(x_batch[0])
            zeros = np.zeros((n,)+xdata[0].shape[1:], dtype='float32')
            feed_dict = {self.ph_x_neg_i: x_batch[0][:n//2], 
                        self.ph_x_pos_i: x_batch[0][n//2:],
                        self.ph_x_neg_v: x_batch[1][:n//2:],
                        self.ph_x_pos_v: x_batch[1][n//2:],
                        self.ph_x_test_i: zeros,
                        self.ph_x_test_v: zeros,
                        self.ph_size: n,
                        self.drop_rate: 0
                        }
            pred = sess.run(self.y_pred, feed_dict)
            y_score.extend(pred)
        y_score = np.array(y_score)
        return y_score

    def evaluate(self, sess, xdata, y_true, batch_size=64, threshold=0.5):
        print('  Evaluating: pos {}, neg {}'.format(y_true.sum(), len(y_true)-y_true.sum()))
        y_score = self.predict(sess, xdata, batch_size)
        fpr, tpr, thr = metrics.roc_curve(y_true, y_score)
        # print('thr', thr)
        auc = metrics.auc(fpr, tpr)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score>threshold] = 1
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        print('    acc: {:.4f}, auc: {:.4f}, f1: {:.4f}'.format(acc, auc, f1))
        t = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
        print('    TP {} FN {} FP {} TN {}'.format(t[1,1], t[1,0], t[0,1], t[0,0]), end='')
        if (t[1,1]+t[0,1])>0:
            p = t[1,1]/(t[1,1]+t[0,1])
        else:
            p = 0
        r = t[1,1]/(t[1,1]+t[1,0])
        print('    P={:.4f}, R={:.4f}, F1={:.4f}'.format(p, r, 2*p*r/(p+r)))
        # best_f1, best_thr = self.best_f1_thr(y_true, y_score)
        # print('    Best_f1={:.4f}, Best_thr={:.4f}'.format(best_f1, best_thr))
        return acc, f1, auc
    def run_batch(self, sess, x_neg_batch, x_pos_batch, x_test_batch, y, config):
        feed_dict = {self.ph_x_neg_i: x_neg_batch[0], 
            self.ph_x_neg_v: x_neg_batch[1], 
            self.ph_x_pos_i: x_pos_batch[0],
            self.ph_x_pos_v: x_pos_batch[1],
            self.ph_x_test_i: x_test_batch[0],
            self.ph_x_test_v: x_test_batch[1],
            self.ph_size: config.batch_size,
            self.ph_y: y, 
            self.drop_rate: 0.5}
        self.train_disc()
        sess.run([self.op_optimize_dis], feed_dict=feed_dict)
        self.train_class()
        _, acc, loss, acc_dis, loss_dis = sess.run([self.op_optimize, self.acc, self.loss, self.acc_dis, self.loss_dis], feed_dict=feed_dict)
        return acc, loss, acc_dis, loss_dis

    def run(self, sess, x_neg, x_pos, x_val, y_val, x_test, config, epoch_offset=0):
        print(f'Training with batch size {config.batch_size}, epochs {config.epochs}')
        timer_epoch = utils.Clock(config.epochs)
        exp_acc = 0
        exp_loss = 0
        alpha = 0.2
        p_neg = len(x_neg[0])/(len(x_neg[0])+len(x_pos[0]))
        print('p_neg:', p_neg)
        p_inc = np.power(p_neg/0.5, 1.0/5)
        tmp_p_neg = 0.5
        pre_acc, pre_f1 = 0, 0
        cnt_converge = 0
        for epoch in range(config.epochs):
            print(f'\nEpoch {epoch+1}/{config.epochs}  p_neg {round(tmp_p_neg, 3)} [{time.ctime()}]')
            tmp_num_neg = int(config.batch_size * tmp_p_neg)
            # tmp_p_neg = min(tmp_p_neg*p_inc, p_neg)
            y = np.zeros([config.batch_size,1], dtype='int32')
            y[tmp_num_neg:,0] = 1
            num_batchs = x_neg[0].shape[0]//config.batch_size
            gen_x_neg = utils.gen_batch2(x_neg, tmp_num_neg)
            gen_x_pos = utils.gen_batch2(x_pos, config.batch_size-tmp_num_neg)
            gen_x_test = utils.gen_batch2(x_test, config.batch_size)
            timer_batch = utils.Clock(num_batchs)
            avg_acc, avg_loss = 0, 0
            for batch in range(num_batchs):
                x_neg_batch, x_pos_batch = next(gen_x_neg), next(gen_x_pos)
                x_test_batch = next(gen_x_test)
                acc, loss, acc_dis, loss_dis = self.run_batch(sess, x_neg_batch, x_pos_batch, x_test_batch, y, config)
                exp_acc = (1-alpha)*exp_acc + alpha*acc
                exp_loss = (1-alpha)*exp_loss + alpha*loss
                avg_acc = (avg_acc*batch + acc)/(batch+1)
                avg_loss = (avg_loss*batch + loss)/(batch+1)
                toc = timer_batch.toc(batch)
                if (batch+1)%(num_batchs//10) == 0:
                    # acc, loss = sess.run([self.acc, self.loss], feed_dict=feed_dict)    
                    stat = [avg_acc, avg_loss, exp_acc, exp_loss]
                    print('  Batch {}/{}: acc {:.4f} | loss {:.4f} | d_acc {:.4f} | d_loss {:.4f} | Elapsed {} | ETA {}'.format(batch+1, num_batchs, acc, loss, acc_dis, loss_dis, *toc))
                    print('        avg_acc {:.4f} | avg_loss {:.4f} | exp_acc {:.4f} | exp_loss {:.4f}'.format(*stat))
            scores = self.evaluate(sess, x_val, y_val, config.batch_size)
            toc = timer_epoch.toc(epoch)
            ckpt_path = '{:04d}_acc{:.4f}_f{:.4f}_auc{:.4f}.h5'.format(epoch_offset+epoch, *scores)
            ckpt_path = os.path.join(config.ckpt_dir, ckpt_path)
            print('  Saving model to {}'.format(ckpt_path))
            # self.model.save_weights(os.path.join(config.ckpt_dir, ckpt_path))
            self.model.save_weights(ckpt_path)
            print('Elapsed {} | ETA {}'.format(*toc))
            if scores[0]==pre_acc and scores[1]==pre_f1:
                cnt_converge += 1
            else:
                pre_acc, pre_f1 = scores[:2]
                cnt_converge = 0
            if cnt_converge >= 2:
                print('Converged and exist.')
                break 

        print('\nEvaluating on train dataset ...')
        x_train_i = np.concatenate([x_neg[0], x_pos[0]], axis=0)
        x_train_v = np.concatenate([x_neg[1], x_pos[1]], axis=0)
        x_train = (x_train_i, x_train_v)
        y_train = np.zeros([len(x_train_i), 1], dtype='int32')
        y_train[len(x_neg[0]):,0] = 1
        self.evaluate(sess, x_train, y_train, config.batch_size)
        print('done.')

def get_data(config):
    df, sparse_cols, num_train = utils.get_dataset_df(config.data_dir, debug=config.is_debug)
    print('cols:', ', '.join(df.columns))
    print('Sparse_cols:', sparse_cols)
    ignore_cols = ['target']
    dense_cols = [col for col in df.columns if col not in sparse_cols+ignore_cols]
    sparse_cols.sort()
    dense_cols.sort()
    # print('Get embeddings ....')
    # # embedding = {col:utils.binary_embedding(df[col].max()+1) for col in sparse_cols}
    # embedding = utils.load_embedding(df, sparse_cols, config.data_dir, method=config.embedding_method)
    # print('Processing embeddings ....')
    # x = utils.process_embedding(df, dense_cols, embedding)

    feat_dict, feat_dim = utils.gen_feat_dict(df, sparse_cols, ignore_cols)
    Xi, Xv = utils.parse_dataset(df, feat_dict, sparse_cols, ignore_cols)

    idx0 = df[df['target']==0].index.values
    idx1 = df[df['target']==1].index.values
    del df
    gc.collect()

    x_neg_train_i = Xi[idx0,:]
    x_pos_train_i = Xi[idx1,:]
    x_neg_train_v = Xv[idx0,:]
    x_pos_train_v = Xv[idx1,:]
    x_test_i = Xi[num_train:]
    x_test_v = Xv[num_train:]
    x_neg_train = (x_neg_train_i, x_neg_train_v)
    x_pos_train = (x_pos_train_i, x_pos_train_v)
    x_test = (x_test_i, x_test_v)
    return x_neg_train, x_pos_train, x_test, feat_dim

def get_ckpt_path(ckpt_dir, epoch=-1):
    ckpt_paths = glob.glob(os.path.join(ckpt_dir, '*.h5'))
    if ckpt_paths:
        ckpt_path = sorted(ckpt_paths)[epoch]
        t = os.path.basename(ckpt_path)
        epoch = int(t[:t.find('_')])
    else:
        ckpt_path = None 
        epoch = 0
    return ckpt_path, epoch

config = Config()
config.data_dir = 'dataset'
config.batch_size = 128
config.epochs = 10
config.embedding_method = 'random'
config.ckpt_dir = 'ckpt'
config.is_debug = True
config.load_weights = True
config.deep_layers = [16, 16]

if sys.platform == 'linux':
    config.is_debug = False
    config.ckpt_dir = '/root/ckpt'

if config.is_debug:
    config.epochs = 3

def main():

    x_neg_train, x_pos_train, x_test, feat_dim = get_data(config)
    print('Splitting ...')
    from sklearn.model_selection import train_test_split
    x_neg_i, x_neg_i_val, x_neg_v, x_neg_v_val = train_test_split(*x_neg_train, test_size=0.1, random_state=2020)
    x_pos_i, x_pos_i_val, x_pos_v, x_pos_v_val = train_test_split(*x_pos_train, test_size=0.1, random_state=2020)
    x_pos = (x_pos_i, x_pos_v)
    x_neg = (x_neg_i, x_neg_v)
    del x_neg_train, x_pos_train
    gc.collect()

    x_val_i = np.concatenate([x_neg_i_val, x_pos_i_val], axis=0)
    x_val_v = np.concatenate([x_neg_v_val, x_pos_v_val], axis=0)
    x_val = (x_val_i, x_val_v)
    y_val = np.zeros((len(x_val_i),1), dtype='int32')
    y_val[len(x_neg_i_val):,0] = 1
    print(f'  x_neg: {x_neg_i.shape}, x_neg_val: {x_neg_i_val.shape}')
    print(f'  x_pos: {x_pos_i.shape}, x_pos_val: {x_pos_i_val.shape}')

    x_test_i = np.concatenate([x_test[0], x_val_i], axis=0)
    x_test_v = np.concatenate([x_test[1], x_val_v], axis=0)
    x_test = (x_test_i, x_test_v)

    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    model = Model(feat_dim, x_neg[0].shape[1], deep_layers=config.deep_layers)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    # tfconfig.device_count = {"CPU": 4, 'GPU': 1},
    with tf.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer()) 
        if config.load_weights:
            ckpt_path, epoch_offset = get_ckpt_path(config.ckpt_dir)
            if ckpt_path:
                print('Loading weights from {} with epoch_offset {} ...'.format(ckpt_path, epoch_offset))
                model.model.load_weights(ckpt_path)
        else:
            epoch_offset = 0
        model.run(sess, x_neg, x_pos, x_val, y_val, x_test, config, epoch_offset)

if __name__ == '__main__':
    main()
    # m = Model(2048, 100)
    # m.model.summary()
    # m.train_disc()
    # for l in m.model.layers:
    #     print(l.__dict__, l.name, l.trainable)

    

