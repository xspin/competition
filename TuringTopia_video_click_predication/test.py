from deepfm import *
import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('INFO')
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    submit_dir = '.'
    if sys.platform == 'linux':
        submit_dir = '/root'

    x_neg_train, x_pos_train, x_test, feat_dim = get_data(config)
    print('Splitting ...')
    from sklearn.model_selection import train_test_split
    x_neg_i, x_neg_i_val, x_neg_v, x_neg_v_val = train_test_split(*x_neg_train, test_size=0.1, random_state=2020)
    x_pos_i, x_pos_i_val, x_pos_v, x_pos_v_val = train_test_split(*x_pos_train, test_size=0.1, random_state=2020)
    x_pos = (x_pos_i, x_pos_v)
    x_neg = (x_neg_i, x_neg_v)
    input_dim = x_neg[0].shape[1]
    del x_neg_train, x_pos_train, x_neg, x_pos
    gc.collect()

    x_val_i = np.concatenate([x_neg_i_val, x_pos_i_val], axis=0)
    x_val_v = np.concatenate([x_neg_v_val, x_pos_v_val], axis=0)
    x_val = (x_val_i, x_val_v)
    y_val = np.zeros((len(x_val_i),1), dtype='int32')
    y_val[len(x_neg_i_val):,0] = 1

    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    model = Model(feat_dim, input_dim, deep_layers=config.deep_layers)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    # tfconfig.device_count = {"CPU": 4, 'GPU': 1},
    with tf.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer()) 
        if config.load_weights:
            ckpt_path, epoch_offset = get_ckpt_path(config.ckpt_dir, epoch=-1)
            if ckpt_path:
                print('Loading weights from {} with epoch_offset {} ...'.format(ckpt_path, epoch_offset))
                model.model.load_weights(ckpt_path)
        else:
            epoch_offset = 0
        # model.run(sess, x_neg, x_pos, x_val, y_val, config, epoch_offset)
        best_thr, best_f1 = 0.5, 0
        if 1:
            for thr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
                print('\nEvaluate on validation with threshold %.2f'%thr)
                acc, f1, auc = model.evaluate(sess, x_val, y_val, config.batch_size, threshold=thr)
                if f1>best_f1:
                    best_f1 = f1
                    best_thr = thr

        print('\nPredicting on test ...')
        pred = model.predict(sess, x_test, config.batch_size)
        y_pred = np.array([1 if p>best_thr else 0 for p in pred], dtype='int32')
        print('pos:', y_pred.mean())
        submit_path = os.path.join(submit_dir, 'submission_{}_{}_{}.csv'.format(best_thr, best_f1, y_pred.mean()))
        print('Saving to %s ...'%submit_path)
        submit = pd.read_csv('dataset/sample.csv')
        submit['target'] = y_pred
        submit.to_csv(submit_path, index=False)

if __name__ == '__main__':
    main()
  