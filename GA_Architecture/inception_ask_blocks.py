import numpy as np
import random
import copy

import torch
import torch.nn as nn
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from torchsummary import summary

from build_blocks import *


class ArchitectureEncoding():
    def __init__(self, in_channels,
                 max_num_blocks=3,
                 max_num_subblocks=5,
                 inception_output_channels=128,
                 subblock_types=['conv', 'maxpool', 'avgpool', 'identity'],
                 conv_kernel_opts=[1, 3, 5, 7],
                 out_channel_opts=[16, 32, 64, 128, 256]):
        self.in_channels = in_channels
        self.max_num_blocks = max_num_blocks
        self.max_num_subblocks = max_num_subblocks
        self.inception_output_channels = inception_output_channels
        self.subblock_types = subblock_types
        self.conv_kernel_opts = conv_kernel_opts
        self.out_channel_opts = out_channel_opts

    def architecture_encoding_generator(self):
        """
        Generate random architecture encoders - consisting of inception-like blocks.
        Each inception block can consist of several blocks - convolutional, avg/max pooling with kernel (1,1).
        The inception blocks also contain 1x1 convolutional layer to ensure same number of filters are outputed
        after each block for easier manipulation later.
        Size of input channels has to be predifined. Maximal number of inception blocks and number
        of the sub-blocks inside them can be defined. Size of convolutional kernel as well as number
        of its channels are chosen from a pre-set list, that can also be modified.
        """
        # number of input channels entering the first block
        in_channels = self.in_channels
        # initialize empty dict to store the encoding architecture of inception blocks as dict
        # with key names as blocks and values as lists of subblocks
        encoding = dict()
        num_output_channels = []
        for i in range(self.max_num_blocks):
            key = 'block_' + str(i)
            encoding[key] = dict()

        num_blocks = random.randint(1, self.max_num_blocks)  # get number of inception blocks
        for block in range(num_blocks):
            # dict key name as dynamic variable
            block_name = 'block_' + str(block)
            # variable to count the final number of channels after concating all subblocks
            output_channels = 0
            # get number of subblocks within each block
            num_subblocks = random.randint(1, self.max_num_subblocks)
            subblock_dict = dict()
            for subblock in range(num_subblocks):
                # randomly choose block type, kernel size and size of output channels
                subblock_type = random.choice(self.subblock_types)
                kernel = random.choice(self.conv_kernel_opts)
                out_channels = random.choice(self.out_channel_opts)

                ### let identity have same number of outputs as inputs
                if subblock_type == 'identity':
                    out_channels = in_channels
                # print(f'subblock {subblock_type}\t input {in_channels}\t output {out_channels}')
                ###

                # count all the output channels dimension so that it can be new input size for next block
                output_channels = output_channels + out_channels
                # create a dict with **kwargs to input into the subblocks
                subblock_params = dict(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(kernel, kernel))
                subblock_key = subblock_type + '_' + str(subblock)
                subblock_dict[subblock_key] = subblock_params
                # save it into a temporary inception block list
            # appear the list of subblocks (the complete inception block) into the whole architecture encoding
            encoding[block_name] = subblock_dict
            # update the number of input channels for the following block
            num_output_channels.append(output_channels)
            in_channels = self.inception_output_channels
        return encoding, num_output_channels, self.inception_output_channels

    def architecture_crossover(self, architecture_1, architecture_2):
        """
        Creates a new architecture from two architectures that are input to this function.
        The crossover is based on random generator of sub-blocks from the concated available options at each level (block 1, block 2, ..., block N).
        Specify maximal number of subblocks to limit the number of subblocks in each inception block.
        """
        encoding = dict()  # empty dict to store the final crossovered architecture
        num_output_channels = list()  # list with number of output channels for each block
        # TBD - the dictionaries should both have same max dimension based on previous functions, if not, take care of it... - which they do now, but might be potential bug later?
        arch_blocks = architecture_1.keys()  # get key names from the blocks (architecture 1 and 2 should have them same, based on the design)
        # loop through all the inception blocks of both architectures
        for block in arch_blocks:
            block_tmp = dict()  # empty dict to store the types of block of each inception block
            output_channels = 0  # variable to count the final number of channels after concating all subblocks
            keys_ls = list()  # stores key names from both architectures
            values_ls = list()  # stores values from both architectures

            # extract all the keys and parameter values into coherent lists
            keys_ls.append(list(architecture_1[block].keys()))
            keys_ls.append(list(architecture_2[block].keys()))
            keys_ls = sum(keys_ls, [])  # merge the keys into one list
            keys_ls = [key.split('_')[0] for key in keys_ls]  # get rid of the numbering

            values_ls.append(list(architecture_1[block].values()))
            values_ls.append(list(architecture_2[block].values()))
            values_ls = sum(values_ls, [])  # merge the values into one list

            # find the lenght of the merged list
            len_samples = len(keys_ls)
            sample_idxs = np.linspace(0, len_samples - 1, len_samples, dtype='u8')  # imitate indexes
            #  if larger than max_num_subblocks then limit the k variable to this number - needed for randomly choosing subbblock samples for new architecture
            if len_samples < self.max_num_subblocks:
                k = len_samples
            else:
                k = self.max_num_subblocks
            # randomly choose samples/subblock types (as indexes) from the merged list of the subblocks
            new_block_idxs = random.sample(list(sample_idxs), k=k)  # k value limits the number of newly chosen blocks

            # iterate through the newly chosen indexes and store in temporary dictionary that is later saved into the final encoding
            for i, idx in enumerate(new_block_idxs):
                key_name = keys_ls[idx] + '_' + str(i)
                block_tmp[key_name] = values_ls[idx]
                output_channels = output_channels + values_ls[idx]['out_channels']  # sum dimension of all subblocks
            if output_channels > 0:
                num_output_channels.append(
                    output_channels)  # append the sum of the output channels within one inception block

            # save the inception block to the final encoding list
            encoding[block] = block_tmp
        return encoding, num_output_channels, self.inception_output_channels

    def architecture_mutation(self, architecture, inc_channels, mutation_rate=0.05):
        """
        Mutate the architecture - either put whole subblock within the inception block as identity
        (does nothing), only passes the signal further or edits number of output filters.
        """

        architecture_copy = copy.deepcopy(architecture)
        encoding = dict()  # empty dict to store the final crossovered architecture
        num_output_channels = list()  # list with number of output channels for each block
        arch_blocks = architecture_copy.keys()  # the dictionaries should both have same max dimension based on previous functions, if not, take care of it...
        in_channels = self.in_channels
        # in_channels = inc_channels

        # loop through all the inception blocks of both architectures
        for block in arch_blocks:
            # check if block empty or not - if yes continue with mutation process
            if architecture_copy.get(block) != {}:
                subblock_dict = dict()
                output_channels = 0
                pop_subblock = None

                # MUTATION OPTION #1
                # delete random subblock within the inception block if
                # the subblock has more than one layer
                if random.random() < mutation_rate:
                    block_keys = list(architecture_copy[block].keys())
                    if len(block_keys) > 1:
                        pop_subblock = random.choice(block_keys)
                        del architecture_copy[block][pop_subblock]

                # MUTATION OPTION #2
                # changes output channels and number of kernels in conv layer
                # not applied if the layer is 'identity'
                for i, subblock in enumerate(architecture_copy[block]):
                    if random.random() < mutation_rate and subblock.split('_')[0] != 'identity':
                        # correct the numeration of the subblocks - in case some were deleted in previous step
                        key_name = subblock.split('_')[0]
                        key_name = key_name + '_' + str(i)
                        in_channels = architecture_copy[block][subblock]['in_channels']  # copy number of input channels
                        kernel = random.choice(self.conv_kernel_opts)  # change kernel size
                        out_channels = random.choice(self.out_channel_opts)  # change output size
                        # update the parameters with new values
                        subblock_params = dict(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=(kernel, kernel))
                    else:
                        # correct the numeration of the subblocks - in case some were deleted in previous step
                        key_name = subblock.split('_')[0]
                        key_name = key_name + '_' + str(i)
                        subblock_params = architecture_copy[block][subblock]
                    # append subblock into dictionary with all the sublocks of one block
                    subblock_dict[key_name] = subblock_params
                    # update the number of output channels
                    output_channels = output_channels + subblock_params[
                        'out_channels']  # sum dimension of all subblocks
                num_output_channels.append(
                    output_channels)  # append the sum of the output channels within one inception block
                encoding[block] = subblock_dict  # save the inception block to the final encoding list
                # inc_channels = in_channels
                in_channels = self.inception_output_channels
            # if empty, output empty block as well
            else:
                encoding[block] = {}
        return encoding, num_output_channels, self.inception_output_channels


class Architecture_Encoder(nn.Module):
    """
    Function that takes the generated architecture with inception blocks and "builds" it as nn.Module.
    (If any new block are added to the option, check that you also edited them into a for loop bellow..!)
    """

    def __init__(self, in_channels, architecture, output_channels, inception_output_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        self.output_channels = output_channels
        self.inception_output_channels = inception_output_channels
        self.n_classes = n_classes
        # create empty module list to store the inception blocks into
        self.architecture_encoder = nn.ModuleList([])
        # get architecture keys (block_0, ..., block_n)
        self.arch_blocks = architecture.keys()


        # iterate through the blocks
        for block, channels in zip(self.arch_blocks, self.output_channels):
            # initiate empty list to store the initialize subblocks modules into
            inception_block = nn.ModuleList()
            # get subblock types and their input parameters
            arch_keys = self.architecture[block].keys()
            arch_values = self.architecture[block].values()
            # see what type each subblock is and call the class function with the nn module
            # save each to the inception block
            for arch_k, arch_v in zip(arch_keys, arch_values):
                key = arch_k.split('_')[0]
                if key == 'conv':
                    inception_block.append(conv_nxn_subblock(**arch_v))
                elif key == 'maxpool':
                    inception_block.append(maxpool_subblock(**arch_v))
                elif key == 'avgpool':
                    inception_block.append(avgpool_subblock(**arch_v))
                else:
                    inception_block.append(identity_subblock(**arch_v))
            # initialize and save the whole inception block into the finale architecture module list
            self.architecture_encoder.append(
                InceptionBlock(channels, inception_block, output_channels=self.inception_output_channels))
            # architecture head
            self.first_layer = conv_nxn_subblock(1, self.in_channels)
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(self.inception_output_channels, self.n_classes)

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.architecture_encoder:
            x = block(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
