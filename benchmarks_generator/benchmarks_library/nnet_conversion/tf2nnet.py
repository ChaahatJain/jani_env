#
# This file is part of the JANI benchmarks repository for PlaJA.
# Copyright (c) (2019 - 2024) Marcel Vinzent, Lukas Wilde.
# See README.md in the top-level directory for licensing information.
#

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


class OptionParser:
    def __init__(self):
        self.option_parser = argparse.ArgumentParser(description="Tensorflow SavedModel conversion to NNet format")

    def add_options(self):
        self.option_parser.add_argument("--input", default=None, help="The directory of the saved model.")
        # self.option_parser.add_argument("--task", default=None, help="The task of the NN to be converted, either `gridworld` or `turtlebot`")
        self.option_parser.add_argument("--out", default=None, help="The output NNet file.")
        self.option_parser.add_argument("--scale-inputs", nargs='+', default=None, type=float, help="For each input a scaling factor on the state variable inputs.")
        self.option_parser.add_argument("--min-inputs", nargs='+', default=None, type=float, help="Lower bound on each input.")
        self.option_parser.add_argument("--max-inputs", nargs='+', default=None, type=float, help="Upper bound on each input.")

    def arg_parse(self):
        self.add_options()
        return self.option_parser.parse_args()


class ModelConverter:
    def __init__(self, args):
        # Args.
        self.input_dir = args.input
        # self.task = args.task
        self.output_file = args.out
        self.scale_inputs = args.scale_inputs
        self.min_inputs = args.min_inputs
        self.max_inputs = args.max_inputs
        # Load model.
        # model = keras.models.load_model(self.input_dir)
        print("Loading model")
        self.model = tf.saved_model.load(self.input_dir)

    def convert(self):
        self.header = '//Neural Network File Format by Kyle Julian, Stanford 2016\n'
        self.header += f'//Converted model: {self.input_dir}'
        self.deprecated_flag = 0

        # Model.variables contains bias vectors, so divide by 2.
        self.num_layers = len(self.model.variables) // 2
        self.num_inputs = self.model.variables[0].shape[0]
        # Second to last for last weight matrix, last entry is last bias vector.
        self.num_outputs = self.model.variables[-2].shape[1]
        # Skip bias vector sizes
        self.layer_sizes = [self.num_inputs] + [layer.shape[1] for layer in self.model.variables[0::2]]
        self.max_layer_size = max(self.layer_sizes)

        assert self.min_inputs is None or len(self.min_inputs) == self.num_inputs
        assert self.max_inputs is None or len(self.max_inputs) == self.num_inputs
        self.min_input_values = [-1000] * self.num_inputs if self.min_inputs is None else self.min_inputs
        self.max_input_values = [1000] * self.num_inputs if self.max_inputs is None else self.max_inputs
        self.mean_values = [0] * self.num_inputs + [0]
        self.range_values = [1] * self.num_inputs + [1]

        self.weights = dict()
        for i, (weight, bias) in enumerate(zip(self.model.variables[0::2], self.model.variables[1::2])):
            weight = weight.numpy()
            bias = bias.numpy()
            if i == 0 and self.scale_inputs is not None:  # self.task is not None:
                assert len(self.scale_inputs) == self.num_inputs
                weight_shape = weight.shape
                # print(f"Shape of the first layer weights:\n{weight_shape}")
                # if self.task == "gridworld":
                rescaling_weights = np.array(self.scale_inputs)  # np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
                rescaling_matrix = np.tile(rescaling_weights, (weight_shape[1], 1)).T
                # print(f"Rescaling matrix applied:\n{rescaling_matrix}")
                weight = np.multiply(weight, rescaling_matrix)
            self.weights[i] = (weight.T, np.atleast_2d(bias).T)

    def write_file(self):
        def join_int_list(values: list[int]) -> str:
            return ','.join(map(lambda x: str(x), values)) + ',\n'

        s = ""
        s += f"{self.header}\n"
        s += f"{self.num_layers},{self.num_inputs},{self.num_outputs},{self.max_layer_size},\n"
        s += join_int_list(self.layer_sizes)
        s += f"{self.deprecated_flag},\n"
        s += join_int_list(self.min_input_values)
        s += join_int_list(self.max_input_values)
        s += join_int_list(self.mean_values)
        s += join_int_list(self.range_values)

        np.set_printoptions(suppress=True, precision=8, floatmode='maxprec_equal', linewidth=1000)
        for (weight, bias) in self.weights.values():
            s += np.array2string(weight, separator=',').replace('[', '').replace(']', '').replace(' ', '')
            s += ',\n'
            s += np.array2string(bias, separator=',').replace('[', '').replace(']', '').replace(' ', '')
            s += ',\n'

        output_path = Path(self.output_file).parent.absolute()
        output_path.mkdir(exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write(s)


if __name__ == "__main__":
    args = OptionParser().arg_parse()
    model_converter = ModelConverter(args)
    model_converter.convert()
    model_converter.write_file()

