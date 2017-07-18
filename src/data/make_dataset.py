# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from glob import glob
from xml.etree import ElementTree
import numpy as np
from dotenv import find_dotenv, load_dotenv
import cPickle as pickle

def get_text(data_file, text_files):
    # Check if the data_file correspond to at least one text file (data is not complete)
    for i in xrange(len(text_files)): 
        text_file_name = text_files[i].split("/")[-1]
        if data_file.split("-")[1] in text_file_name.split(".")[0].split("-"):
            f = open(text_files[i], "r")
            lines = []
            start = 0
            # Parse the line of the text corresponding to the CSR version
            for l in f:
                l = l.rstrip()
                lines.append(l)
            f.close()
            for j in xrange(len(lines)):
                l = lines[j]
                if l.find("CSR:") >= 0:
                    start = j+2
                    break

            idx = int(data_file.split("-")[2].split(".")[0])
            return lines[start:][idx-1]
    return None

def get_stroke_position(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()

    data = None
    for stroke in root.getiterator("Stroke"):
        part = []
        for point in list(stroke):
            part.append([float(point.get("x")), float(point.get("y")), (0)])
        part[-1][2] = 1
        if data is not None:
            data = np.r_[data, part]
        else:
            data = np.asarray(part)

    offs = data[1:, 0:2] - data[0:-1, 0:2]
    offs = np.c_[offs, data[1:, 2:]]
    return offs


def get_writer_id(writer_info, name):
    line = list(writer_info['writerID'].ix[writer_info['id'] == name])
    return line[0] if len(line) > 0 else None


def read_data(data_path, text_path, writer_info):
    logger = logging.getLogger(__name__)
    data = []
    text = []
    writer_id = []
    file_name = []
    persons = glob(data_path + '/*')

    # Create the dataset for each person
    for person in persons:
        person_dir = person.split("/")[-1]
        person_path = data_path + "/" + person_dir
        logger.info(person_path)

        trials = glob(person_path + "/*")
        for trial in trials:
            trial_dir = trial.split("/")[-1]
            trial_path = person_path + "/" + trial_dir
            text_files = glob(text_path + "/" + person_dir + "/" + trial_dir + "/*")
            data_files = glob(trial_path + "/*")
            for data_file in data_files:
                data_file_name = data_file.split("/")[-1]
                # Get the text, position, file name and writer id for this trial
                text_file = get_text(data_file_name, text_files)
                text.append(text_file)
                offset = get_stroke_position(data_file)
                data.append(offset)
                file_name.append([data_file])
                wids = get_writer_id(writer_info, data_file.split("-")[0] + "-" + data_file.split("-")[1])
                if wids is None:
                    wids = get_writer_id(writer_info, trial)
                writer_id.append([wids if wids is not None else -1])

    return data, text, file_name, writer_id

def get_normalized_data(train_data, valid_data):
    logger = logging.getLogger(__name__)
    tmp = None
    for offsets in [train_data, valid_data]:
        for data in offsets:
            if tmp is not None:
                tmp = np.r_[tmp, np.copy(data[:, 0:2])]
            else:
                tmp = np.copy(data[:, 0:2])

    mean_x = np.mean(tmp[:, 0])
    mean_y = np.mean(tmp[:, 1])
    std_x = np.std(tmp[:, 0])
    std_y = np.std(tmp[:, 1])
    logger.info("before normalize: (mean_x, mean_y, std_x, std_y) = (%.2f, %.2f, %.2f, %.2f)" %(mean_x, mean_y, std_x, std_y))

    norm_train = []
    for arr in train_data:
        arr1 = np.copy(arr)
        arr1[:,0] -= mean_x
        arr1[:,1] -= mean_y
        arr1[:,0] /= std_x
        arr1[:,1] /= std_y
        norm_train.append(arr1)

    norm_valid = []
    for arr in valid_data:
        arr1 = np.copy(arr)
        arr1[:,0] -= mean_x
        arr1[:,1] -= mean_y
        arr1[:,0] /= std_x
        arr1[:,1] /= std_y
        norm_valid.append(arr1)

    return norm_train, norm_valid, (mean_x, mean_y, std_x, std_y)

def get_data_padded(data, callback):
    dest = []
    for arr in data:
        print(arr)
        l_pad = callback(arr)
        pads = np.zeros((l_pad, 3))
        pads[:, 2] = 2.0
        arr1 = np.r_[np.copy(arr), pads]
        print(arr1)
        exit()
        dest.append(arr1)

    return dest

def get_char_to_index(text):
    vocab = {}
    cha_index = []
    for l in text:
        tmp = []
        for s in l:
            dataset = np.ndarray(len(s[0]), dtype=np.int32)
            for i, cha in enumerate(s[0]):
                if cha not in vocab:
                    vocab[cha] = len(vocab)
                dataset[i] = vocab[cha]
            tmp.extend(dataset)
        cha_index.append(np.asarray(tmp))
    return cha_index, vocab

def save_data(path, data):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/raw/OnlineHandWriting', help='Directory containing data.')
@click.option('--out_dir', type=click.Path(exists=False), default='data/processed/OnlineHandWriting', help='Output directory of the processed data')
@click.option('--segment_length', type=click.INT, default=200, help='Minimum padding length.')
def main(data_dir, out_dir, segment_length):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Read the CSV map data
    writer_info = pd.read_csv(data_dir + "/writer_info.csv")

    # Create the training set and the valid data
    train_data, train_text, train_file_name, train_writer_id = read_data(data_dir + "/train/lineStrokes", data_dir + "/train/ascii", writer_info)
    valid_data, valid_text, valid_file_name, valid_writer_id = read_data(data_dir + "/valid/lineStrokes", data_dir + "/valid/ascii", writer_info)

    # Normalize the data
    norm_train, norm_valid, stats = get_normalized_data(train_data, valid_data)

    # Pad the data to the minimal length
    norm_train_padded = get_data_padded(norm_train, lambda arr: segment_length - arr.shape[0]%segment_length if arr.shape[0] < 1000 else 1400 - arr.shape[0])
    norm_valid_padded = get_data_padded(norm_valid, lambda arr: 2000 - arr.shape[0])

    # Convert characters to index
    line_text_data = train_text + valid_text
    cha_index, vocab = get_char_to_index(line_text_data)
    train_characters = cha_index[0:len(train_text)]
    valid_characters = cha_index[len(train_text):]

    # Save the results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    logger.info("Saving the training data")

    if not os.path.exists(out_dir + "/train"):
        os.makedirs(out_dir + "/train")

    save_data("data/processed/OnlineHandWriting/train/train_data", norm_train_padded)
    save_data("data/processed/OnlineHandWriting/train/train_characters", train_characters)
    save_data("data/processed/OnlineHandWriting/train/train_text", train_text)
    save_data("data/processed/OnlineHandWriting/train/train_writer_id", train_writer_id)

    logger.info("Saving the validation data")

    if not os.path.exists(out_dir + "/valid"):
        os.makedirs(out_dir + "/valid")

    save_data("data/processed/OnlineHandWriting/valid/valid_data", norm_valid_padded)
    save_data("data/processed/OnlineHandWriting/valid/valid_characters", valid_characters)
    save_data("data/processed/OnlineHandWriting/valid/valid_text", valid_text)
    save_data("data/processed/OnlineHandWriting/valid/valid_writer_id", valid_writer_id)

    logger.info("Saving the general data")
    save_data("data/processed/OnlineHandWriting/vocabulary", vocab)
    save_data("data/processed/OnlineHandWriting/statistics", stats)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
