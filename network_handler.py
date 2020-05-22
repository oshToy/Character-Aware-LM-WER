import logging
import os
from train import main as train, define_flags
from evaluate import main as test
import datetime
import json
import tensorflow as tf
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        if self.epoch in [1, 5, 10, 15, 20]:
            output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
            model.save(output_path)
            print('model save')
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

CONFIG_FILE = 'config.json'
flags = tf.flags

def main():
    config_dict = import_config_settings()
    logs_folder = config_dict['logs_folder']
    data_sets_folder = config_dict['data_sets_folder']
    trained_models_folder = config_dict['trained_models_folder']

    for model in config_dict['models']:
        logger = logger_for_print(folder=logs_folder)
        fasttext_model_path = model['fasttext_model_path']

        #copy_data_files(data_folder=data_sets_folder + model['data_set'])
        data_set = model['data_set']
        del_all_flags(tf.flags.FLAGS)

        flags.DEFINE_string('data_dir', data_sets_folder + '/' + data_set,
                            'data directory. Should contain train.txt/valid.txt/test.txt with input data')



        flags.DEFINE_string('fasttext_model_path', fasttext_model_path, 'fasttext trained model path')
        flags.DEFINE_string('embedding', model['embedding'], 'embedding method')

        define_flags()
        if model['training']:
            trained_model_folder = trained_models_folder + '/' + data_set + '_' + str(
                datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
            flags.DEFINE_string('train_dir', trained_model_folder,
                                'training directory (models and summaries are saved there periodically)')
            train(logger)
        if model['testing']:
            trained_model_folder = model['checkpoint_file_for_test']
            flags.DEFINE_string('train_dir', os.path.dirname(os.path.abspath(trained_model_folder)),
                                'training directory (models and summaries are saved there periodically)')
            checkpoint_file = checkpoint_file_from_number(model, trained_model_folder)
            logger("test on model file : " + str(checkpoint_file))
            if not checkpoint_file:
                break
            checkpoint_file = checkpoint_file.replace(".index", "")
            tf.flags.DEFINE_string('load_model_for_test',  checkpoint_file,
                                   '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
            test(logger)


def checkpoint_file_from_number(model, trained_model_folder):
    if model["checkpoint_file_for_test"] is not None:
        return model["checkpoint_file_for_test"]

    files_list = os.listdir(trained_model_folder)
    if not model['checkpoint_number_for_test_or_null_for_last_checkpoint'] and model['training']:
        print("testing last checkpoint from training folder : " + str(trained_model_folder))
        index_numbers_list = list()
        for file in files_list:
            if ".index" in file:
                index_numbers_list.append(int(file.split('_')[0].split('h')[1]))
        last_checkpoint_number = max(index_numbers_list)
        for file in files_list:
            if ".index" in file and str(last_checkpoint_number) in  file.split('_')[0]:
                return trained_model_folder + file

    elif model['checkpoint_number_for_test_or_null_for_last_checkpoint']:
        for file in files_list:
            if ".index" in file and str(model['checkpoint_number_for_test_or_null_for_last_checkpoint']) in file.split('_')[0]:
                return trained_model_folder + file
        print("checkpoint_number_for_test: " + str(model['checkpoint_number_for_test_or_null_for_last_checkpoint']) + "Not Found")
        return None
    else:
        print("checkpoint_number_for_test is missing")

        return None


def import_config_settings():
    with open(CONFIG_FILE) as json_file:
        return json.load(json_file)


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def logger_for_print(folder='', file_name='logger'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    date = str(datetime.datetime.now()).replace(':', '_').replace('.', '_')
    logger.addHandler(logging.FileHandler(folder + '\\' + file_name + date+'.log', 'a'))
    return logger.info


def copy_data_files(data_folder):
    root_folder = os.getcwd()
    os.system('robocopy ' + data_folder + ' ' + root_folder + '/data')
    return


if __name__ == "__main__":
    main()