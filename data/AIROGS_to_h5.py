import h5py
from PIL import Image
import numpy as np
import os.path as osp
import glob
import pandas as pd
import random
import argparse

def write_range_to_hdf5(images, counter_from, counter_to, img_data, id, ids):
    """ writes range of 5 images to hdf5 file
                    Parameters:
                    ---------------
                    images        list of arrays (images)
                    counter_from  write from
                    counter_to    write to
                    img_data hdf5 dataset
                    ids array ids
                    id hdf5 dataset
    """
    # add images
    img_arr = np.asarray(images)
    img_data[counter_from:counter_to] = img_arr

    # add ids
    dt = h5py.special_dtype(vlen=str)
    id[counter_from:counter_to] = np.asarray(ids, dtype=dt)


def add_images(input_file, image_paths, img_data, counter_from, id):
    """ preprocesses images and adds them to .
            Parameters:
            ---------------
            image_paths   list of paths to all imamges
            orig          whether it's an original image and the mask must be used (boolean)
            device_type   Bosch, Forus or Remidio
            img_data      hdf5 dataset
            id            hdf5 dataset
    """
    max_write_buffer = 4
    write_buffer = 0
    images = []
    ids = []

    # go through all images, then preprocess them and write them to hdf5 files in batches
    for i in range(len(image_paths)):
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
            write_range_to_hdf5(images, counter_from, counter_to, img_data, id, ids)
            # delete cash/lists
            images.clear()
            ids.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4


        # get id of image
        img_id = osp.splitext(osp.basename(image_paths[i]))[0]
        ids.append(img_id)

        #path_to_image = osp.join(input_file, 'preprocessed_img', img_id + "." + file["file_extension_data_out"])
        #if not osp.exists(path_to_image):
        # convert image PIL to numpy array
        image_pillow = Image.open(image_paths[i])
        image = np.asarray(image_pillow)
        image_pillow.close()

        # load preprocessed image
        preprocessed_image_pillow = Image.open(osp.join(input_file, 'preprocessed_img',
            img_id + ".png"))
        preprocessed_image = np.asarray(preprocessed_image_pillow, dtype="uint8")
        preprocessed_image_pillow.close()
        images.append(preprocessed_image)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
        write_range_to_hdf5(images, counter_from, counter_to, img_data, id, ids)


def create_hdf5_dataset(input_file, hdf5_file_path):
    """ cerates dataset to add to hdf5 file.
                Parameters:
                ---------------
                input_file   root directory
                Returns:
                ----------
                images      filled images array to add ot hdf5 file
    """
    print("Converting images")
    # paths to the images
    # add all image addresses
    img_addr = glob.glob(input_file + "/*.jpg")

    hdf5_file = h5py.File(hdf5_file_path, "w")

    # ---- train val split ----
    # compute indices for train, val, test (70/20/10)
    ind_shuffled = list(range(len(img_addr)))
    random.shuffle(ind_shuffled)
    train, val, test = np.split(np.asarray(ind_shuffled), [int(len(ind_shuffled) * 0.7), int(len(ind_shuffled) * 0.9)])
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()

    # number of images in each set
    number_train_images = len(train)
    number_val_images = len(val)
    number_test_images = len(test)

    img_data_train = hdf5_file.create_dataset("train/images",shape=(number_train_images, 320, 320, 3), dtype="uint8")  # resize size = 320
    img_data_val = hdf5_file.create_dataset("val/images", shape=(number_val_images, 320, 320, 3), dtype="uint8")  # resize size = 320
    img_data_test = hdf5_file.create_dataset("test/images", shape=(number_test_images, 320, 320, 3), dtype="uint8")  # resize size = 320


    dt = h5py.special_dtype(vlen=str)
    id_train = hdf5_file.create_dataset("train/id", shape=(number_train_images,), dtype=dt)
    id_val = hdf5_file.create_dataset("val/id", shape=(number_val_images,), dtype=dt)
    id_test = hdf5_file.create_dataset("test/id", shape=(number_test_images,), dtype=dt)

    # ---- train ----
    add_images(input_file, [img_addr[i] for i in train], img_data_train, counter_from=0, id=id_train)

    # ---- val ----
    add_images(input_file, [img_addr[i] for i in val], img_data_val, counter_from=0, id=id_val)

     # ---- test ----
    add_images(input_file, [img_addr[i] for i in test], img_data_test, counter_from=0, id=id_test)

    # ---- diagnoses ----
    print("Converting diagnoses")
    diagnoses_train = hdf5_file.create_dataset("train/diagnosis", shape=(number_train_images,), dtype="i")
    diagnoses_val = hdf5_file.create_dataset("val/diagnosis", shape=(number_val_images,), dtype="i")
    diagnoses_test = hdf5_file.create_dataset("test/diagnosis", shape=(number_test_images,), dtype="i")

    
    labels = create_labels_dataset(input_file)

    # train
    expert_labels = [labels[i] for i in train]
    diagnoses_train[:] = np.asarray(expert_labels, dtype='i')
    expert_labels.clear()
    # val
    expert_labels = [labels[i] for i in val]
    diagnoses_val[:] = np.asarray(expert_labels, dtype='i')
    expert_labels.clear()
    # test
    expert_labels = [labels[i] for i in test]
    diagnoses_test[:] = np.asarray(expert_labels, dtype='i')
    expert_labels.clear()

    hdf5_file.close()


def create_labels_dataset(input_file):
    """ main function ot convert the directory of images to a hdf5 file.
                Parameters:
                ---------------
                input_file   file where dataset is
    """
    # paths to the labels
    path_label = osp.join(input_file + "/train_labels.csv")

    # read entire csv file
    csv_labels = pd.read_csv(path_label)

    labels = []

    csv_labels['class'].apply(lambda x: labels.append(0) if x == 'NRG' else labels.append(1))
    return labels


def convert_to_hdf5(input_file, output_file):
    """ main function ot convert the directory of images to a hdf5 file.
                Parameters:
                ---------------
                input_file   root directory
    """
    create_hdf5_dataset(input_file, hdf5_file_path=output_file)

          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converting to H5")
    parser.add_argument(
        "--input_file",
        dest="input_file",
        action="store",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_file",
        dest="output_file",
        action="store",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    convert_to_hdf5(input_file=args.input_file,
                    output_file=args.output_file)