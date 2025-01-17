## aen_attack.py -- attack a network optimizing elastic-net distance with an en decision rule
##                  when autoencoder loss is applied
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

import sys
import numpy as np
from typing import Type

import tensorflow.compat.v1 as tf
import keras

from .utils import setup_logger

tf.disable_v2_behavior()
log = setup_logger(__name__)


class AEADEN:
    def __init__(
        self,
        sess: Type[tf.compat.v1.Session],
        model,
        mode: str,
        AE: Type[tf.keras.Sequential],
        batch_size: int,
        kappa: float,
        init_learning_rate: float,
        binary_search_steps: int,
        max_iterations: int,
        initial_const: float,
        beta: float,
        gamma: float,
    ):
        image_size, num_channels, nun_classes = (
            model.image_size,
            model.num_channels,
            model.num_labels,
        )
        shape = (batch_size, image_size, image_size, num_channels)

        self.sess = sess
        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.AE = AE
        self.mode = mode
        self.beta = beta
        self.gamma = gamma

        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img_s = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.target_lab = tf.Variable(
            np.zeros((batch_size, nun_classes)), dtype=tf.float32
        )
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)

        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img_s = tf.placeholder(tf.float32, shape)
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size, nun_classes))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        self.zt = tf.divide(self.global_step, self.global_step + tf.cast(3, tf.float32))

        cond1 = tf.cast(
            tf.greater(tf.subtract(self.adv_img_s, self.orig_img), self.beta),
            tf.float32,
        )
        cond2 = tf.cast(
            tf.less_equal(
                tf.abs(tf.subtract(self.adv_img_s, self.orig_img)), self.beta
            ),
            tf.float32,
        )
        cond3 = tf.cast(
            tf.less(tf.subtract(self.adv_img_s, self.orig_img), tf.negative(self.beta)),
            tf.float32,
        )
        upper = tf.minimum(
            tf.subtract(self.adv_img_s, self.beta), tf.cast(0.5, tf.float32)
        )
        lower = tf.maximum(tf.add(self.adv_img_s, self.beta), tf.cast(-0.5, tf.float32))
        self.assign_adv_img = (
            tf.multiply(cond1, upper)
            + tf.multiply(cond2, self.orig_img)
            + tf.multiply(cond3, lower)
        )

        cond4 = tf.cast(
            tf.greater(tf.subtract(self.assign_adv_img, self.orig_img), 0), tf.float32
        )
        cond5 = tf.cast(
            tf.less_equal(tf.subtract(self.assign_adv_img, self.orig_img), 0),
            tf.float32,
        )
        if self.mode == "PP":
            self.assign_adv_img = tf.multiply(cond5, self.assign_adv_img) + tf.multiply(
                cond4, self.orig_img
            )
        elif self.mode == "PN":
            self.assign_adv_img = tf.multiply(cond4, self.assign_adv_img) + tf.multiply(
                cond5, self.orig_img
            )

        self.assign_adv_img_s = self.assign_adv_img + tf.multiply(
            self.zt, self.assign_adv_img - self.adv_img
        )
        cond6 = tf.cast(
            tf.greater(tf.subtract(self.assign_adv_img_s, self.orig_img), 0), tf.float32
        )
        cond7 = tf.cast(
            tf.less_equal(tf.subtract(self.assign_adv_img_s, self.orig_img), 0),
            tf.float32,
        )
        if self.mode == "PP":
            self.assign_adv_img_s = tf.multiply(
                cond7, self.assign_adv_img_s
            ) + tf.multiply(cond6, self.orig_img)
        elif self.mode == "PN":
            self.assign_adv_img_s = tf.multiply(
                cond6, self.assign_adv_img_s
            ) + tf.multiply(cond7, self.orig_img)

        self.adv_updater = tf.assign(self.adv_img, self.assign_adv_img)
        self.adv_updater_s = tf.assign(self.adv_img_s, self.assign_adv_img_s)

        """--------------------------------"""
        # prediction BEFORE-SOFTMAX of the model
        self.delta_img = self.orig_img - self.adv_img
        self.delta_img_s = self.orig_img - self.adv_img_s
        if self.mode == "PP":
            self.ImgToEnforceLabel_Score = model.predict(self.delta_img)
            self.ImgToEnforceLabel_Score_s = model.predict(self.delta_img_s)
        elif self.mode == "PN":
            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
            self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)

        # distance to the input data
        self.L2_dist = tf.reduce_sum(tf.square(self.delta_img), [1, 2, 3])
        self.L2_dist_s = tf.reduce_sum(tf.square(self.delta_img_s), [1, 2, 3])
        self.L1_dist = tf.reduce_sum(tf.abs(self.delta_img), [1, 2, 3])
        self.L1_dist_s = tf.reduce_sum(tf.abs(self.delta_img_s), [1, 2, 3])
        self.EN_dist = self.L2_dist + tf.multiply(self.L1_dist, self.beta)
        self.EN_dist_s = self.L2_dist_s + tf.multiply(self.L1_dist_s, self.beta)

        # compute the probability of the label class versus the maximum other
        self.target_lab_score = tf.reduce_sum(
            (self.target_lab) * self.ImgToEnforceLabel_Score, 1
        )
        target_lab_score_s = tf.reduce_sum(
            (self.target_lab) * self.ImgToEnforceLabel_Score_s, 1
        )
        self.max_nontarget_lab_score = tf.reduce_max(
            (1 - self.target_lab) * self.ImgToEnforceLabel_Score
            - (self.target_lab * 10000),
            1,
        )
        max_nontarget_lab_score_s = tf.reduce_max(
            (1 - self.target_lab) * self.ImgToEnforceLabel_Score_s
            - (self.target_lab * 10000),
            1,
        )
        if self.mode == "PP":
            Loss_Attack = tf.maximum(
                0.0, self.max_nontarget_lab_score - self.target_lab_score + self.kappa
            )
            Loss_Attack_s = tf.maximum(
                0.0, max_nontarget_lab_score_s - target_lab_score_s + self.kappa
            )
        elif self.mode == "PN":
            Loss_Attack = tf.maximum(
                0.0, -self.max_nontarget_lab_score + self.target_lab_score + self.kappa
            )
            Loss_Attack_s = tf.maximum(
                0.0, -max_nontarget_lab_score_s + target_lab_score_s + self.kappa
            )
        # sum up the losses
        self.Loss_L1Dist = tf.reduce_sum(self.L1_dist)
        self.Loss_L1Dist_s = tf.reduce_sum(self.L1_dist_s)
        self.Loss_L2Dist = tf.reduce_sum(self.L2_dist)
        self.Loss_L2Dist_s = tf.reduce_sum(self.L2_dist_s)
        self.Loss_Attack = tf.reduce_sum(self.const * Loss_Attack)
        self.Loss_Attack_s = tf.reduce_sum(self.const * Loss_Attack_s)
        if self.mode == "PP":
            self.Loss_AE_Dist = self.gamma * tf.square(
                tf.norm(self.AE(self.delta_img) - self.delta_img)
            )
            self.Loss_AE_Dist_s = self.gamma * tf.square(
                tf.norm(self.AE(self.delta_img) - self.delta_img_s)
            )
        elif self.mode == "PN":
            self.Loss_AE_Dist = self.gamma * tf.square(
                tf.norm(self.AE(self.adv_img) - self.adv_img)
            )
            self.Loss_AE_Dist_s = self.gamma * tf.square(
                tf.norm(self.AE(self.adv_img_s) - self.adv_img_s)
            )

        self.Loss_ToOptimize = (
            self.Loss_Attack_s + self.Loss_L2Dist_s + self.Loss_AE_Dist_s
        )
        self.Loss_Overall = (
            self.Loss_Attack
            + self.Loss_L2Dist
            + self.Loss_AE_Dist
            + tf.multiply(self.beta, self.Loss_L1Dist)
        )

        self.learning_rate = tf.train.polynomial_decay(
            self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5
        )
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        self.train = optimizer.minimize(
            self.Loss_ToOptimize,
            var_list=[self.adv_img_s],
            global_step=self.global_step,
        )
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.orig_img.assign(self.assign_orig_img))
        self.setup.append(self.target_lab.assign(self.assign_target_lab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.adv_img.assign(self.assign_adv_img))
        self.setup.append(self.adv_img_s.assign(self.assign_adv_img_s))

        self.init = tf.variables_initializer(
            var_list=[self.global_step] + [self.adv_img_s] + [self.adv_img] + new_vars
        )

    def attack(self, imgs, labs):
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                # x[y] -= self.kappa if self.PP else -self.kappa
                if self.mode == "PP":
                    x[y] -= self.kappa
                elif self.mode == "PN":
                    x[y] += self.kappa
                x = np.argmax(x)
            if self.mode == "PP":
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.init_const
        Const_UB = np.ones(batch_size) * 1e10
        # the best l2, score, and image attack
        overall_best_dist = [1e10] * batch_size
        overall_best_attack = [np.zeros(imgs[0].shape)] * batch_size

        for binary_search_steps_idx in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            img_batch = imgs[:batch_size]
            label_batch = labs[:batch_size]

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup,
                {
                    self.assign_orig_img: img_batch,
                    self.assign_target_lab: label_batch,
                    self.assign_const: CONST,
                    self.assign_adv_img: img_batch,
                    self.assign_adv_img_s: img_batch,
                },
            )

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                self.sess.run([self.train])
                self.sess.run([self.adv_updater, self.adv_updater_s])

                Loss_Overall, Loss_EN, OutputScore, adv_img = self.sess.run(
                    [
                        self.Loss_Overall,
                        self.EN_dist,
                        self.ImgToEnforceLabel_Score,
                        self.adv_img,
                    ]
                )
                Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist = self.sess.run(
                    [
                        self.Loss_Attack,
                        self.Loss_L2Dist,
                        self.Loss_L1Dist,
                        self.Loss_AE_Dist,
                    ]
                )
                target_lab_score, max_nontarget_lab_score_s = self.sess.run(
                    [self.target_lab_score, self.max_nontarget_lab_score]
                )

                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    log.info("iter:{} const:{}".format(iteration, CONST))
                    log.info(
                        "Loss_Overall:{:.4f}, Loss_Attack:{:.4f}".format(
                            Loss_Overall, Loss_Attack
                        )
                    )
                    log.debug(
                        "Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}".format(
                            Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist
                        )
                    )
                    log.debug(
                        "target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f} \n".format(
                            target_lab_score[0], max_nontarget_lab_score_s[0]
                        )
                    )
                    sys.stdout.flush()

                for batch_idx, (the_dist, the_score, the_adv_img) in enumerate(
                    zip(Loss_EN, OutputScore, adv_img)
                ):
                    if the_dist < current_step_best_dist[batch_idx] and compare(
                        the_score, np.argmax(label_batch[batch_idx])
                    ):
                        current_step_best_dist[batch_idx] = the_dist
                        current_step_best_score[batch_idx] = np.argmax(the_score)
                    if the_dist < overall_best_dist[batch_idx] and compare(
                        the_score, np.argmax(label_batch[batch_idx])
                    ):
                        overall_best_dist[batch_idx] = the_dist
                        overall_best_attack[batch_idx] = the_adv_img

            # adjust the constant as needed
            for batch_idx in range(batch_size):
                if (
                    compare(
                        current_step_best_score[batch_idx],
                        np.argmax(label_batch[batch_idx]),
                    )
                    and current_step_best_score[batch_idx] != -1
                ):
                    # success, divide const by two
                    Const_UB[batch_idx] = min(Const_UB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    Const_LB[batch_idx] = max(Const_LB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2
                    else:
                        CONST[batch_idx] *= 10

        # return the best solution found
        overall_best_attack = overall_best_attack[0]
        return overall_best_attack.reshape((1,) + overall_best_attack.shape)
