import numpy as np
import os
from scipy import ndimage, misc, special
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def generate_filenames_list(subdirectory = 'data/train/', subfolders = True) :
    """Returns a list of filenames in a given directory.  If subfolders is
    set to True, then fn will also iterate through all subfolders."""
    if subfolders :
        for i, species_ID in enumerate(os.listdir(subdirectory)[1:]) :
            fish_file_names = []
            fish_file_names = [subdirectory+species_ID+'/'+x for x in os.listdir(subdirectory+'/'+species_ID) ]
            fish_count = len(fish_file_names)

            try :
                master_file_names = master_file_names + fish_file_names
            except :
                master_file_names = fish_file_names
    else :
        master_file_names = [subdirectory+x for x in os.listdir(subdirectory)]
    return master_file_names




def show_panel(image) :
    """Shows an RGB montage of an image in array form."""
    plt.figure(figsize=(16,8))
    plt.subplot(1,4,1)
    plt.imshow(image[:,:,0], cmap = 'Reds')
    plt.subplot(1,4,2)
    plt.imshow(image[:,:,1], cmap = 'Greens')
    plt.subplot(1,4,3)
    plt.imshow(image[:,:,2], cmap = 'Blues')
    plt.subplot(1,4,4)
    plt.imshow(image)
    plt.show()

def boxit(coarse_dims = [64, 112, 3], fov_dim = 72) :
    # get list of filenames to be sampled
    with open('label_dictionary.pickle', 'rb') as handle :
        label_dictionary = pickle.load(handle)

    f_list = []
    for key in label_dictionary.keys() :
        if label_dictionary.get(key).get('label') != 'NoF' :
            if label_dictionary.get(key).get('scale') is None :
                f_list.append(key)


    while True :

        f = f_list.pop(np.random.randint(0, len(f_list)))
        print("="*50)
        print("{} Keys remaining ".format(len(f_list)))
        print("-"*50)
        print(f)
        print("Prediction exists? {}".format(label_dictionary.get(f).get('FiNoF') is not None))
        print("-"*50)

        img = misc.imread(f, mode = 'RGB')
        shape = img.shape
        if label_dictionary.get(f).get('FiNoF') is not None :
            print('FiNoF Probability : {}'.format(label_dictionary.get(f).get('FiNoF')))
            print("Scale : {}".format(label_dictionary.get(f).get('box_preds')[0]))
            print("YX Coords : {}".format(label_dictionary.get(f).get('box_preds')[1:]))
            scale = label_dictionary.get(f).get('box_preds')[0]
            coords = label_dictionary.get(f).get('box_preds')[1:]
            plt.imshow(retrieve_fovea(f, top_left_coords = coords, scale = scale, fov_dim = fov_dim))
            plt.show()
            use_box = input("Use fovea specifics from predictions? (y/n)   ")
            if use_box == 'y' :
                label_dictionary[f]['coord'] = coords
                label_dictionary[f]['scale'] = np.array([scale])
            else :
                plt.imshow(img)
                plt.show()
                shape = img.shape

                imgC = misc.imresize(img, size = coarse_dims, mode = 'RGB')

                plt.imshow(imgC)
                plt.grid(b=True, which = 'both', linestyle = '--', color = 'red')
                plt.show()

                top_left_y = input("y coordinate of top left border   ")
                top_left_x = input("x coordinate of the top left border   ")
                bottom_right_y = input("y coordinate of bottom right border   ")
                bottom_right_x = input("x coordinate of bottom right border   ")

                tl = np.array([int(top_left_y), int(top_left_x)]) / np.array(coarse_dims[0:2])
                br = np.array([int(bottom_right_y), int(bottom_right_x)]) / np.array(coarse_dims[0:2])

                TL = np.round(tl * shape[0:2]).astype(int)
                BR = np.round(br * shape[0:2]).astype(int)

                dims = BR - TL
                dim = np.max(dims)
                ext = (dim - np.min(dims)) // 2

                if dims[0] > dims[1] :
                    TL[1] = TL[1] - ext
                else :
                    TL[0] = TL[0] - ext

                fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]

                plt.imshow(fov)
                plt.show()

                adjust = input("Adjustments needed? (y/n)    ")

                if adjust == 'n' :
                    proceed = True
                else :
                    proceed = False
                while proceed == False :
                    ad_horizontal = int(input("adjust left or right? (neg is left)   "  ))
                    ad_vertical = int(input("adjust up or down? (neg is up)     "))
                    ad_zoom = float(input("percent zoom (1.0 = same size, >1 -> zoom out)?    "))

                    TL[1] = TL[1] + ad_horizontal
                    TL[0] = TL[0] + ad_vertical

                    dim = np.round(dim * ad_zoom).astype(int)

                    fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]
                    plt.imshow(fov)
                    plt.show()

                    adjust = input("Adjustments needed? (y/n)    ")

                    if adjust == 'n' :
                        proceed = True
                    else :
                        proceed = False

                scale = fov_dim / dim
                # convert pixel offsets of the top left coordinate to the proportion of the y and x dimensions of the original high-resolution image
                TL = TL / shape[0:2]

                label_dictionary[f]['coord'] = TL
                label_dictionary[f]['scale'] = np.array([scale])

        else :
            plt.imshow(img)
            plt.show()
            shape = img.shape

            imgC = misc.imresize(img, size = coarse_dims, mode = 'RGB')

            plt.imshow(imgC)
            plt.grid(b=True, which = 'both', linestyle = '--', color = 'red')
            plt.show()

            top_left_y = input("y coordinate of top left border   ")
            top_left_x = input("x coordinate of the top left border   ")
            bottom_right_y = input("y coordinate of bottom right border   ")
            bottom_right_x = input("x coordinate of bottom right border   ")

            tl = np.array([int(top_left_y), int(top_left_x)]) / np.array(coarse_dims[0:2])
            br = np.array([int(bottom_right_y), int(bottom_right_x)]) / np.array(coarse_dims[0:2])

            TL = np.round(tl * shape[0:2]).astype(int)
            BR = np.round(br * shape[0:2]).astype(int)

            dims = BR - TL
            dim = np.max(dims)
            ext = (dim - np.min(dims)) // 2

            if dims[0] > dims[1] :
                TL[1] = TL[1] - ext
            else :
                TL[0] = TL[0] - ext

            fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]

            plt.imshow(fov)
            plt.show()

            adjust = input("Adjustments needed? (y/n)    ")

            if adjust == 'n' :
                proceed = True
            else :
                proceed = False
            while proceed == False :
                ad_horizontal = int(input("adjust left or right? (neg is left)   "  ))
                ad_vertical = int(input("adjust up or down? (neg is up)     "))
                ad_zoom = float(input("percent zoom (1.0 = same size, >1 -> zoom out)?    "))

                TL[1] = TL[1] + ad_horizontal
                TL[0] = TL[0] + ad_vertical

                dim = np.round(dim * ad_zoom).astype(int)

                fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]
                plt.imshow(fov)
                plt.show()

                adjust = input("Adjustments needed? (y/n)    ")

                if adjust == 'n' :
                    proceed = True
                else :
                    proceed = False

            scale = fov_dim / dim
            # convert pixel offsets of the top left coordinate to the proportion of the y and x dimensions of the original high-resolution image
            TL = TL / shape[0:2]

            label_dictionary[f]['coord'] = TL
            label_dictionary[f]['scale'] = np.array([scale])


        print(label_dictionary[f])

        plt.imshow(retrieve_fovea(f, TL, scale, fov_dim = 72))
        plt.show()

        commit = input("Save dictionary? (y/n)  ")
        if commit == 'y' :
            with open('label_dictionary.pickle', 'wb') as fld :
                pickle.dump(label_dictionary, fld)


def retrieve_fovea(f, top_left_coords, scale, fov_dim = 72) :

    img = misc.imread(f, mode = 'RGB')
    img_shape = img.shape

    offsets = np.round((img_shape[0:2] * top_left_coords).astype(int))
    fov = img[offsets[0]:, offsets[1]:, :]
    sc_fov = misc.imresize(fov, size = scale, mode = 'RGB')
    new_fov = sc_fov[:fov_dim, :fov_dim, :]
    """
    sc_img = misc.imresize(img, size = scale, mode = 'RGB')
    sc_shape = sc_img.shape
    offsets = np.round( top_left_coords * sc_shape[0:2]   ).astype(int)
    overshoot_value = (sc_shape[0:2] - (offsets+fov_dim) )
    overshoot_bool = (offsets+fov_dim > sc_shape[0:2]).astype(int)
    offsets += (overshoot_value*overshoot_bool)
    new_img = sc_img[offsets[0]:offsets[0]+fov_dim,
                     offsets[1]:offsets[1]+fov_dim,
                     :]
    """
    return new_fov


"""Functions for preparing bateches into the FishNoF model"""

def bundle(f_list, label_dictionary, coarse_dims = [64,112,3]) :
    """
    Generates an array of coarse images with corresponding FishNoF lables from
    an input list of filenames.
    """

    for i, f in enumerate(f_list) :
        img = misc.imresize(misc.imread(f, mode = 'RGB'), size = coarse_dims, mode = 'RGB')
        is_fish = label_dictionary.get(f).get('is_fish')

        if i == 0 :
            fish_vector = np.expand_dims(is_fish,0)
            coarse_arr = np.expand_dims(img,0)
        else :
            fish_vector = np.concatenate([fish_vector, np.expand_dims(is_fish,0)])
            coarse_arr = np.concatenate([coarse_arr, np.expand_dims(img,0)], 0)

    return coarse_arr, fish_vector


def process_fovea(fovea, pixel_norm = 'standard', mutation = False) :
    """
    Fn preprocesses a single fovea array.
    If mutation == True, modifications to input images will be made, each with 0.5
    probability:
        * smallest dimension resized to standard height and width supplied in size param
        * each channel centered to mean near zero.  Deviation is not normalized.
        * if mutate == True :
            * random flip left right
            * random flip up down
            * random rotation 90 degrees
            * TODO : random colour adjustment
    """
    if mutation :
        if np.random.randint(0,2,1) == 1 :
            fovea = np.fliplr(fovea)
        if np.random.randint(0,2,1) == 1 :
            fovea = np.flipud(fovea)
        if fovea.shape[0] == fovea.shape[1] :
            if np.random.randint(0,2,1) == 1 :
                fovea = np.rot90(fovea)

    #pixel normalization
    if pixel_norm == 'standard' :
        fovea = fovea.astype(np.float32)
        fovea = (fovea / 255.0) - 0.5
    elif pixel_norm == 'float' :
        fovea = fovea.astype(np.float32)
        fovea = (fovea / 255.0)
        fovea = np.clip(fovea, a_min = 0.0, a_max = 1.0)
    elif pixel_norm == 'centre' :
        red = 96.48265253757386
        green = 107.20367931267522
        blue = 99.97448662926035
        fovea = fovea.astype(np.float32)
        fovea[:, :, 0] = fovea[:, :, 0] - red
        fovea[:, :, 1] = fovea[:, :, 1] - green
        fovea[:, :, 2] = fovea[:, :, 2] - blue
    else :
        pass
    return fovea


def bundle_mt(f_list, label_dictionary, coarse_dims = [64,112,3], fov_dim = 72) :

    for i, f in enumerate(f_list) :
        # To speed retrieval I have saved the coarse images for every high resolution image in
        # an alternative directory that starts with 'data/coarse_train/...' as opposed to
        # 'data/train/...'
        coarse_key = 'data/coarse_'+f[5:]
        img = misc.imread(coarse_key, mode = 'RGB')
        is_fish = label_dictionary.get(f).get('is_fish')
        scale = label_dictionary.get(f).get('scale')
        coords = label_dictionary.get(f).get('coord')


        if (scale is None) or (coords is None) :
            scale = np.array([-1])
            coords = np.array([-1,-1])
            weights = np.array([0])
        else :
            weights = np.array([1])

        if i == 0 :
            fish_vector = np.expand_dims(is_fish,0)
            coarse_arr = np.expand_dims(img,0)
            scale_vector = np.expand_dims(scale,0)
            coords_arr = np.expand_dims(coords, 0)
            weights_vector = np.expand_dims(weights, 0)

        else :
            fish_vector = np.concatenate([fish_vector, np.expand_dims(is_fish,0)])
            coarse_arr = np.concatenate([coarse_arr, np.expand_dims(img,0)], 0)
            scale_vector = np.concatenate([scale_vector, np.expand_dims(scale, 0)], 0)
            coords_arr = np.concatenate([coords_arr, np.expand_dims(coords, 0)], 0)
            weights_vector = np.concatenate([weights_vector, np.expand_dims(weights, 0)], 0)

    return coarse_arr, fish_vector.astype(np.int32), np.concatenate([scale_vector, coords_arr], 1), weights_vector


def prepare_FishyFish_batch(f_list, embedding_df, annotated_fovea_directory, predicted_fovea_directory, annotated_boxes, box_preds,
                            label_df, FiNoF_prob_series, class_weight_dictionary, fov_weight_predicted = 0.2, fov_crop = 64) :
    """
    Function retrieves arrays for training or prediction of FishyFish model.
    """


    batch_embedding = embedding_df.loc[f_list, :]
    batch_FiNoF = FiNoF_prob_series.loc[f_list]
    batch_labs = label_df.loc[f_list]

    for key in f_list :
        new_key = key[11:]
        if key in list(annotated_boxes.index) :
            fov = misc.imread(annotated_fovea_directory+new_key, mode = 'RGB')
            fov_weight = np.array([1])

        else :
            fov = misc.imread(predicted_fovea_directory+new_key, mode = 'RGB')
            fov_weight = np.array([fov_weight_predicted])


        rand_y = np.random.randint(0,8)
        rand_x = np.random.randint(0,8)
        fov = fov[rand_y:rand_y+fov_crop, rand_x:rand_x+fov_crop, :]
        fov = process_fovea(fov, pixel_norm = 'centre', mutation = False)

        if fov.shape[0] != fov_crop or fov.shape[1] != fov_crop :
            fov = misc.imresize(fov, size = [fov_crop, fov_crop, 3])

        try :
            fov_stack = np.concatenate([fov_stack, np.expand_dims(fov, 0)], 0)
            fov_weight_stack = np.concatenate([fov_weight_stack, fov_weight], 0)
        except :
            fov_stack = np.expand_dims(fov, 0)
            fov_weight_stack = fov_weight


    class_labels = np.argmax(np.array(batch_labs), 1)
    batch_label_weights = np.ones([len(class_labels)])
    for ix, label in enumerate(class_labels) :
        batch_label_weights[ix] = class_weight_dictionary.get(label)

    return batch_embedding, batch_FiNoF, batch_labs, batch_label_weights, fov_stack, fov_weight_stack
