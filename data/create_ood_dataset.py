# divide test set into five roughly equal parts
# create h5 file, copy h5/test contents
# shuffle indices
# apply levels of ood
# write images, id, diagnosis to file

import h5py
from PIL import Image
from torchvision.transforms import v2
import numpy as np
import random
import cv2

def blur(image, level):
    sigma = [5,11,15,19, 31]
    return cv2.GaussianBlur(image,(sigma[level],sigma[level]),0)

def speckle(image, level):
    var = [0.05, 0.1, 0.15, 0.2, 1.2]
    image = image / 255.0
    noise = np.random.normal(loc=0.0, scale=var[level], size=image.shape).astype(np.float32)
    noisy_image = np.clip(image + image * noise, 0, 1)
    return (noisy_image * 255).astype(np.uint8)

def write_range_to_hdf5(images, counter_from, counter_to, img_data, levels, levels_data, ids_data, ids, diagnoses_data, diagnoses):
    """ writes range of 5 images to hdf5 file
                    Parameters:
                    ---------------
                    images        list of arrays (images)
                    counter_from  write from
                    counter_to    write to
                    img_data hdf5 dataset
                    levels array levels
                    levels_data hdf5 dataset
    """
    # add images
    img_arr = np.asarray(images)
    img_data[counter_from:counter_to] = img_arr
    levels = np.asarray(levels)
    levels_data[counter_from:counter_to] = levels
    ids = np.asarray(ids)
    ids_data[counter_from:counter_to] = ids
    diagnoses = np.asarray(diagnoses)
    diagnoses_data[counter_from:counter_to] = diagnoses

def add_images(airogs_file, indices, img_data, counter_from, level, levels_data, ids_data, diagnoses_data):
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
    levels = []
    ids = []
    diagnoses = []


    # go through all images, then preprocess them and write them to hdf5 files in batches
    for i in indices:
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            write_range_to_hdf5(images, counter_from, counter_to, img_data, levels, levels_data, ids_data, ids, diagnoses_data, diagnoses)
            # delete cash/lists
            images.clear()
            levels.clear()
            ids.clear()
            diagnoses.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4

        ids.append(airogs_file["test/id"][i])
        diagnoses.append(airogs_file["test/diagnosis"][i])
        # load image from orginal dataset h5
        image = airogs_file["test/images"][i]

        if level == 1:
            #image = Image.fromarray((image * 255).astype(np.uint8))
            #transformer = v2.ElasticTransform(alpha=100, sigma=2)
            #image = transformer(image)
            image = blur(image, level-1)
            #image = np.asarray(image)
        elif level == 2:
            #image = Image.fromarray((image * 255).astype(np.uint8))
            #transformer = v2.ElasticTransform(alpha=200, sigma=2)
            #image = transformer(image)
            image = blur(image, level-1)
            #image = np.asarray(image)
        elif level == 3:
            #image = Image.fromarray((image * 255).astype(np.uint8))
            #transformer = v2.ElasticTransform(alpha=300, sigma=2)
            #image = transformer(image)
            image = blur(image, level-1)
            #image = np.asarray(image)
        elif level == 4:
            #image = Image.fromarray((image * 255).astype(np.uint8))
            #transformer = v2.ElasticTransform(alpha=500, sigma=2)
            #image = transformer(image)
            image = blur(image, level-1)
            #image = np.asarray(image)
        elif level == 5:
            #image = Image.fromarray((image * 255).astype(np.uint8))
            #transformer = v2.ElasticTransform(alpha=500, sigma=2)
            #image = transformer(image)
            image = blur(image, level-1)
            #image = np.asarray(image)

        images.append(image)
        levels.append(level)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        write_range_to_hdf5(images, counter_from, counter_to, img_data, levels, levels_data, ids_data, ids, diagnoses_data, diagnoses)

# ---------- copy test data --------------
# new h5 file for ood dataset
ood_hdf5_file = h5py.File("/project/home/annawundram/datasets/AIROGS/ood_sameimages_blur.h5", "w")

# open AIROGS h5 file
airogs_hdf5_file = h5py.File("/project/home/annawundram/datasets/AIROGS/AIROGS.h5", "r")

num_images = airogs_hdf5_file["test/images"].shape[0]

ind_shuffled = list(range(num_images))
random.shuffle(ind_shuffled)
ind_shuffled = np.asarray(ind_shuffled)
one, two = np.array_split(ind_shuffled, 2)

size = len(one) + 5 * len(two)


images = ood_hdf5_file.create_dataset("images",shape=(size, 320, 320, 3), dtype="uint8")
dt = h5py.special_dtype(vlen=str)
ids = ood_hdf5_file.create_dataset("id", shape=(size,), dtype=dt)
diagnoses = ood_hdf5_file.create_dataset("diagnosis", shape=(size,), dtype="i")
levels = ood_hdf5_file.create_dataset("level", shape=(size,), dtype="i")

# ---------- go through all test images, augment some and save them in new h5 file --------------


# one: original images
add_images(airogs_hdf5_file, one, images, 0, 0, levels, ids, diagnoses)

# two: level 1 images
add_images(airogs_hdf5_file, two, images, len(one), 1, levels, ids, diagnoses)

# three: level 2 images
add_images(airogs_hdf5_file, two, images, len(one) + len(two), 2, levels, ids, diagnoses)

# four: level 3 images
add_images(airogs_hdf5_file, two, images, len(one) + len(two) + len(two), 3, levels, ids, diagnoses)

# five: level 4 images
add_images(airogs_hdf5_file, two, images, len(one) + len(two) + len(two) + len(two), 4, levels, ids, diagnoses)

# six: level 5 images
add_images(airogs_hdf5_file, two, images, len(one) + len(two) + len(two) + len(two) + len(two), 5, levels, ids, diagnoses)

airogs_hdf5_file.close()
ood_hdf5_file.close()