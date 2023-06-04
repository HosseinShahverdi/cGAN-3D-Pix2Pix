import os
import glob
import pre_util as pu
import tensorflow as tf
from solver import Solver
from pathlib import Path
from build_data import data_writer


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have muliple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'pix2pix_db', 'dataset name, default: pix2pix_db')

tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer('iters', 20000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 1000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 50, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_string('load_model', '20221203-201138', 'folder of saved model that you wish to continue training '
                                           '(e.g. 20181203-1647), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        # create nii to predicted nii
        idx = 0
        name_patient_list = []
        z_size = []
        sum = 0
        file_name  = glob.glob(os.path.abspath("pix2pix_db/nifti_sample/*gz"))
        for f in file_name:
            f_split = f.split("\\")
            name_patient = f_split[-1]
            name_patient_list.append(name_patient)
            z = pu.nii_to_sample(f, 'ct', idx)
            z_size.append(z)
            sum += z
            idx += 1 
        data_writer(os.path.abspath("dataset/ready_oneSample"),"test")
        solver.test(sum)
        pu.creat_nii(name_patient_list,z_size)
        pu.add_header(name_patient_list)

if __name__ == '__main__':
    tf.app.run()
