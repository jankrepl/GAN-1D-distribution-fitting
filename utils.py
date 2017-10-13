import time

import matplotlib
import numpy as np
import scipy.stats
import tensorflow as tf

matplotlib.use('TkAgg')  #
import matplotlib.pyplot as plt

if False:
    scipy.stats


# FUNCTIONS
def create_summary_folder_name():
    """ Creates a name for a folder in which we store summaries and figures

    :return: folder name string
    :rtype: str
    """
    dir_str = '' # ADD PATH
    time_stamp_str = time.strftime("%a, %d %b %Y %H:%M:%S/", time.gmtime())
    param_str = ''
    return dir_str + time_stamp_str + param_str


def fc_layer(input_layer, nodes_input, nodes_output, name_scope, final_layer=False):
    """ Generic layer creator so that we can create flexible architectures

    :param input_layer: input tensor
    :type input_layer: tensor
    :param nodes_input: number of nodes in the input tensor
    :type nodes_input: int
    :param nodes_output: number of nodes in the output tensor
    :type nodes_output: int
    :param name_scope: string that is added to a variables name
    :type name_scope: str
    :param final_layer: is final layer (no activation applied)
    :type final_layer: bool

    :return: output tensor
    :rtype: tensor
    """
    W = tf.get_variable(name=name_scope + 'W', shape=[nodes_input, nodes_output],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name=name_scope + 'b', shape=[nodes_output], initializer=tf.constant_initializer(0))

    if final_layer:
        return tf.matmul(input_layer, W) + b  # no activation
    else:
        return tf.nn.relu(tf.matmul(input_layer, W) + b)  # relu activation
        # return tf.sigmoid(tf.matmul(input_layer, W) + b)  # sigmoid activation


def pdf_define(param_td):
    """ Generates a pdf as a lambda function

    :param param_td: parameter dictionary that contains the family and the parameters of the true distribution
    :type param_td: dict
    :return: pdf as a lambda function
    :rtype: lambda function
    """
    parametric_family = param_td['parametric_family']
    true_parameters = param_td['true_parameters']

    eval_str = 'scipy.stats.' + str(parametric_family) + \
               '.pdf' + '(x, ' + str(true_parameters)[1:-1] + ')'

    eval_str = eval_str.replace("'", "").replace(":", "=")

    return lambda x: eval(eval_str)  # define pdf_true


def sample_from_distribution(n, param_dict, seed=None):
    """ Generates n samples from a distribution specified in param_dict

    :param n: number of samples
    :type n: int
    :param param_dict: parameter dictionary that contains the family and the parameters of the true distribution
    :type param_dict: dict
    :param seed: seed, if None -> random
    :type seed: int
    :return: vector of samples
    :rtype: ndarray
    """

    parametric_family = param_dict['parametric_family']
    true_parameters = param_dict['true_parameters']

    eval_str = 'scipy.stats.' + str(parametric_family) + \
               '.rvs(' + str(true_parameters)[1:-1] + ',size=n, random_state=seed)'

    eval_str = eval_str.replace("'", "").replace(":", "=")
    output_vec = eval(eval_str)
    return np.reshape(output_vec, (len(output_vec), 1))


# CLASSES
class Computational_graph:
    def __init__(self, arch_D, arch_G, OS_label_smoothing,
                 optimizer_parameter_D, optimizer_parameter_G):

        # Save parameters as attributes
        self.arch_D = arch_D
        self.arch_G = arch_G
        self.OS_label_smoothing = OS_label_smoothing
        self.optimizer_parameter_D = optimizer_parameter_D
        self.optimizer_parameter_G = optimizer_parameter_G

        # Placeholders
        self.z_placeholder = tf.placeholder(tf.float32, [None, 1], name='z_placeholder')
        self.x_placeholder = tf.placeholder(tf.float32, [None, 1], name='x_placeholder')

        # Define forward pass
        self.Gz = self.__generator(self.z_placeholder)
        self.Dx = self.__discriminator(self.x_placeholder)
        self.Dg = self.__discriminator(self.Gz, reuse_variables=True)

        # Losses
        if OS_label_smoothing[0]:
            # prevents the discriminator from being overly confident
            d_loss_real_labels = OS_label_smoothing[1] * tf.ones_like(self.Dx)
        else:
            d_loss_real_labels = tf.ones_like(self.Dx)

        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx, labels=d_loss_real_labels) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, labels=tf.zeros_like(self.Dg)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, labels=tf.ones_like(self.Dg)))

        # Extract D and G variables
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'D_' in var.name]
        g_vars = [var for var in tvars if 'G_' in var.name]

        # Optimizers
        self.d_opt = tf.train.AdamOptimizer(self.optimizer_parameter_D).minimize(self.d_loss,
                                                                                 var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.optimizer_parameter_G).minimize(self.g_loss, var_list=g_vars)

        # Summaries
        tf.summary.scalar('Generator_loss', self.g_loss)
        tf.summary.scalar('Discriminator_loss', self.d_loss)

        self.merged_summary = tf.summary.merge_all()

        # Initialize
        self.session = tf.InteractiveSession()
        self.folder_name = create_summary_folder_name()
        self.writer = tf.summary.FileWriter(self.folder_name, self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def __discriminator(self, inp, reuse_variables=None):
        """ Defines the forward pass of a discriminator. Note that it needs to be called twice
        because once the inputs are real observations (x) and other time it is the fake obs ( G(z) )

        :param inp: placeholder
        :type inp: placeholder
        :param reuse_variables: if true, variables in the NN stay the same but different placeholder
        :type reuse_variables: bool

        """
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
            nodes_input = 1
            for i in range(len(self.arch_D)):
                nodes_output = self.arch_D[i]
                inp = fc_layer(inp, nodes_input, nodes_output, 'D_' + str(i + 1) + '_')
                nodes_input = self.arch_D[i]

            return fc_layer(inp, self.arch_D[-1], 2,
                            'D_end_',
                            final_layer=True)

    def __generator(self, inp):
        """ Defines the forward passs of the generator.

        :param inp: placeholder
        :type inp: placeholder
        """
        nodes_input = 1
        for i in range(len(self.arch_G)):
            nodes_output = self.arch_G[i]
            inp = fc_layer(inp, nodes_input, nodes_output, 'G_' + str(i + 1) + '_')
            nodes_input = self.arch_G[i]

        return fc_layer(inp, self.arch_G[-1], 1,
                        'G_end_',
                        final_layer=True)


class PreTrainer:
    def __init__(self, computational_graph, iter, batch_size, summary_frequency, param_td, param_nd):
        for i in range(iter):
            z_batch = sample_from_distribution(batch_size, param_nd)
            x_batch = sample_from_distribution(batch_size, param_td)

            feed_dict = {computational_graph.x_placeholder: x_batch, computational_graph.z_placeholder: z_batch}
            op_list = [computational_graph.merged_summary, computational_graph.d_opt]  # only discriminator is learning

            summary, _ = computational_graph.session.run(op_list, feed_dict=feed_dict)

            if i % summary_frequency == 0:
                computational_graph.writer.add_summary(summary, -(iter - i))  # numbered with negative numbers
                print('PreTraining:' + str(i))
        print('Pretraining done ! ! !')


class Trainer:
    def __init__(self, computational_graph, iter, batch_size, steps_D, steps_G, summary_frequency,
                 param_td, param_nd, visualizer_frequency, visualizer_samples, sort_samples):
        for i in range(iter):
            if sort_samples:
                # alligning small values of z with small values of x -> forces the G to be increasing/decreasing

                z_batch_g = np.sort(sample_from_distribution(batch_size, param_nd))

                z_batch_d = np.sort(sample_from_distribution(batch_size, param_nd))

                x_batch = np.sort(sample_from_distribution(batch_size, param_td))

            else:
                z_batch_d = sample_from_distribution(batch_size, param_nd)
                z_batch_g = sample_from_distribution(batch_size, param_nd)
                x_batch = sample_from_distribution(batch_size, param_td)

            feed_dict_d = {computational_graph.x_placeholder: x_batch, computational_graph.z_placeholder: z_batch_d}
            feed_dict_g = {computational_graph.x_placeholder: x_batch, computational_graph.z_placeholder: z_batch_g}

            op_list_d = [computational_graph.d_opt]
            op_list_g = [computational_graph.g_opt]

            # Train discriminator
            for j in range(steps_D):
                computational_graph.session.run(op_list_d, feed_dict=feed_dict_d)

            # Train generator
            for j in range(steps_G):
                computational_graph.session.run(op_list_g, feed_dict=feed_dict_g)

            if i % summary_frequency == 0:
                summary = computational_graph.session.run(computational_graph.merged_summary, feed_dict=feed_dict_g)
                computational_graph.writer.add_summary(summary, i)
                print('Training: ' + str(i))

            if i % visualizer_frequency == 0:
                z_noise = sample_from_distribution(visualizer_samples, param_nd)
                z_grid = np.reshape(np.linspace(0, 1, 100), (100, 1))  # for Z uniform (0,1)
                x_grid = np.reshape(np.linspace(-4, 4, 100), (100, 1)) # for X normal(0,1)

                feed_dict_noise = {computational_graph.z_placeholder: z_noise}
                feed_dict_grid = {computational_graph.z_placeholder: z_grid}
                feed_dict_D = {computational_graph.x_placeholder: x_grid}

                # G
                g = computational_graph.session.run(computational_graph.Gz, feed_dict=feed_dict_grid)
                fig = plt.figure()
                plt.plot(z_grid, g)
                plt.title(str(i))
                fig.savefig(computational_graph.folder_name + str(i) + '_g.png')
                fig.clear()

                # Decision boundary
                db = computational_graph.session.run(tf.nn.sigmoid(computational_graph.Dx), feed_dict=feed_dict_D)
                fig = plt.figure()
                plt.plot(x_grid, db[:, 0])
                plt.title(str(i))
                fig.savefig(computational_graph.folder_name + str(i) + '_db.png')
                fig.clear()

                # Histogram
                pdf_fake = computational_graph.session.run(computational_graph.Gz, feed_dict=feed_dict_noise)
                fig = plt.figure()
                plt.hist(pdf_fake, int(15))
                plt.title(str(i))
                fig.savefig(computational_graph.folder_name + str(i) + '_pdf.png')
                fig.clear()

        print('Training done ! ! !')
