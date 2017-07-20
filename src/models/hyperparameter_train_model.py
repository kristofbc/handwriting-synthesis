import os
import click
import time
import logging

from train_model import main as network_main

from trainer.curriculum import Curriculum
from trainer.visualizer import Visualizer
from trainer.visualizer import VisualizerDefinition
from trainer.trainer import Trainer
from trainer.target import Target
from trainer.program import Program


@click.command()
@click.option('--gpu', type=click.INT, default=-1, help='GPU ID (negative value is CPU).')
@click.option('--epochs', type=click.INT, default=50, help='Number of epochs to compare')
@click.option('--repeat_program', type=click.INT, default=5, help='Mitigate the stochasticity by repeating each program n times.')
def main(gpu, epochs, repeat_program):
    """ Hyperparameters to test """
    base_parameters = {
        "data_dir": "data/processed/OnlineHandWriting",
        "output_dir": "models",
        "peephole": 0,
        "resume": "",
        "gpu": gpu,
        "epochs": epochs,
        "grad_clip": 0,
        "save_interval": 1,
        "adaptive_noise": 1,
        "update_weight": 1,
        "use_weight_noise": 1,
        "random_seed": None
    }

    """ Test 20 combinations """
    # Truncated_data_samples should start at 500 but truncated_data_samples%truncated_back_prop_len = 0
    parameters = [
        [
            {"batch_size": 64, "truncated_back_prop_len": 50, "truncated_data_samples": 500, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ],
        [
            {"batch_size": 128, "truncated_back_prop_len": 50, "truncated_data_samples": 500, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ],
        [
            {"batch_size": 64, "truncated_back_prop_len": 75, "truncated_data_samples": 525, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ],
        [
            {"batch_size": 128, "truncated_back_prop_len": 75, "truncated_data_samples": 525, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ],
        [
            {"batch_size": 64, "truncated_back_prop_len": 100, "truncated_data_samples": 500, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ],
        [
            {"batch_size": 128, "truncated_back_prop_len": 100, "truncated_data_samples": 500, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
            {"train_loss_mean": ["<=", -800], "valid_loss_mean": ["<=", -1000]}
        ]
    ]
    hyperparameters = []
    # To mitigate stochasticity run each Program n times
    for i in xrange(len(parameters)):
        for j in xrange(repeat_program):
            hyperparameters.append(parameters[i])

    #for i in xrange(20):
    #    # Alternate parameters
    #    bprop_len = (50+25*(i-(i+1)%2))
    #    combination = [
    #        {"batch_size": (50+14*(i-i%2)), "truncated_back_prop_len": prop_len, "truncated_data_samples": 500, "rnn_layers_number": 3, "rnn_cells_number": 400, "win_unit_number": 10, "mix_com_number": 20},
    #        {"train_loss_mean": [">=", 0], "valid_loss_mean": [">=", 0]}
    #    ]

    #    hyperparameters.append(combination)

    """ Training Curriculum for the hyperparameters """
    curriculum = Curriculum()
    for i in xrange(len(hyperparameters)):
        activities = dict(hyperparameters[i][0].items() + base_parameters.items())
        targets_dict = hyperparameters[i][1].items()
        objectives = []
        for j in xrange(len(targets_dict)):
            objectives.append(Target(targets_dict[j][0], targets_dict[j][1][0], targets_dict[j][1][1]))
        curriculum.add(Program(activities, objectives))

    """ Visualize the effect on the loss and mean squared error """
    visualizer = Visualizer([
        VisualizerDefinition("batch_size", "train_loss_mean", "Compare loss for hyperparameter: batch_size. Iteration #{}".format(base_parameters["epochs"])),
        VisualizerDefinition("truncated_back_prop_len", "train_loss_mean", "Compare loss for hyperparameter: truncated_back_prop_len. Iteration #{}".format(base_parameters["epochs"]))
    ])

    # Trainer used for the training
    trainer = Trainer(curriculum, visualizer)

    """ Training callback """
    def trainer_callback(act):
        loss_network_train, loss_network_valid = network_main(
            act["data_dir"], act["output_dir"], act["batch_size"], act["peephole"], act["epochs"], act["grad_clip"], act["resume"], act["gpu"],
            act["adaptive_noise"], act["update_weight"], act["use_weight_noise"], act["save_interval"], act["truncated_back_prop_len"], 
            act["truncated_data_samples"], act["rnn_layers_number"], act["rnn_cells_number"], act["win_unit_number"], act["mix_com_number"], act["random_seed"]
        )

        """ Generate a prediction with the trained model and compare the mean squared error """
        

        return {"train_loss_mean": loss_network_train[act["epochs"]-1, 0], "valid_loss_mean": loss_network_valid[act["epochs"]-1, 0]}

    """ Begin training """
    model_suffix_dir = "{0}-{1}-{2}".format(time.strftime("%Y%m%d-%H%M%S"), 'with_peephole' if base_parameters["peephole"] == 1 else 'no_peephole', "hyperparameter-trainer")
    output_path = base_parameters["output_dir"] + "/" + model_suffix_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    trainer.train(output_path, trainer_callback)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
