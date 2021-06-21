## main.py -- sample code to test attack procedure
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
## Edited: 06.2021 - Tuan Anh, Le - <tuananhle@dai-labor.de>

import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
import logging

from keras.layers import Lambda

from src.setup_mnist import MNIST, MNISTModel
from src.utils import *
from src.aen_CEM import AEADEN

log = setup_logger(__name__)


def setup_logging(log_level: str):
    try:
        log_level = getattr(logging, log_level)
        log.setLevel(log_level)
    except:
        log.setLevel(logging.INFO)
    log.debug(f"Log level set to {log_level}.")


def main(args):
    setup_logging(args["log_level"])

    with tf.compat.v1.Session() as sess:
        random.seed(121)
        np.random.seed(1211)

        image_id = args["img_id"]
        arg_max_iter = args["maxiter"]
        arg_b = args["binary_steps"]
        arg_init_const = args["init_const"]
        arg_mode = args["mode"]
        arg_kappa = args["kappa"]
        arg_beta = args["beta"]
        arg_gamma = args["gamma"]

        # init AE, MNIST data, NN
        AE_model = load_AE("mnist_AE_1")
        data, model = MNIST(), MNISTModel("models/mnist", sess, False)

        # get original image prediction
        orig_prob, orig_class, orig_prob_str = model_prediction(
            model, np.expand_dims(data.test_data[image_id], axis=0)
        )
        target_label = orig_class
        log.info("Image:{}, infer label:{}".format(image_id, target_label))
        # get one hot vector
        orig_img, target = generate_data(data, image_id, target_label)

        attack = AEADEN(
            sess,
            model,
            mode=arg_mode,
            AE=AE_model,
            batch_size=1,
            kappa=arg_kappa,
            init_learning_rate=1e-2,
            binary_search_steps=arg_b,
            max_iterations=arg_max_iter,
            initial_const=arg_init_const,
            beta=arg_beta,
            gamma=arg_gamma,
        )

        adv_img = attack.attack(orig_img, target)

        adv_prob, adv_class, adv_prob_str = model_prediction(model, adv_img)
        delta_prob, delta_class, delta_prob_str = model_prediction(
            model, orig_img - adv_img
        )

        INFO = "id:{}, kappa:{}, Orig class:{}, Adv class:{}, Delta class: {}\n, Orig prob:{}\n, Adv prob:{}\n, Delta prob:{}\n".format(
            image_id,
            arg_kappa,
            orig_class,
            adv_class,
            delta_class,
            orig_prob_str,
            adv_prob_str,
            delta_prob_str,
        )
        log.info(INFO)

        suffix = "id{}_kappa{}_Orig{}_Adv{}_Delta{}".format(
            image_id, arg_kappa, orig_class, adv_class, delta_class
        )
        arg_save_dir = "{}_ID{}_Gamma_{}".format(arg_mode, image_id, arg_gamma)
        os.system("mkdir -p results/{}".format(arg_save_dir))

        # save original image
        save_img(
            orig_img, "results/{}/Orig_original{}.png".format(arg_save_dir, orig_class)
        )
        # save adv img
        save_img(adv_img, "results/{}/Adv_{}.png".format(arg_save_dir, suffix))
        # save PP / PN
        save_img(
            np.absolute(orig_img - adv_img) - 0.5,
            "results/{}/Delta_{}.png".format(arg_save_dir, suffix),
        )

        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_level", type=str, default="INFO")
    parser.add_argument("-i", "--img_id", type=int)
    parser.add_argument("-m", "--maxiter", type=int, default=1000)
    parser.add_argument("-b", "--binary_steps", type=int, default=9)
    parser.add_argument("-c", "--init_const", type=float, default=10.0)
    parser.add_argument("--mode", choices=["PN", "PP"], default="PN")
    parser.add_argument("--kappa", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--gamma", type=float, default=0)

    args = vars(parser.parse_args())
    main(args)
