# Code to generate hyper parameters required to Pytorch Ligitning Trainer and rest of experiment code

def get_hparams(
        image_size,
        max_epochs,
        batch_size,
        device,
        gpu,
        generator_dropout_p,
        discriminator_dropout_p,
        generator_lr,
        discriminator_lr,
        weight_decay,
        lr_scheduler_T_0,
        lr_scheduler_T_mult,
        lbda,
        input_channel,
        output_channel


):

    """
    outputs a dictionary data structure of hyper parameters

    :param image_size: image size which was resized
    :param max_epochs: epochs
    :param batch_size: batch size of the experiment dataset
    :param device: cuda or cpu
    :param gpu: pytorch ligthning trainer gpu option
    :param generator_dropout_p: Generator Architecture dropout values
    :param discriminator_dropout_p: Discriminator Architecture dropout values
    :param generator_lr: learning rate of the generator
    :param discriminator_lr: learning rate of the discriminator
    :param weight_decay: weight decay for optimizers for L2 regularization
    :param lr_scheduler_T_0: Learning rate scheduler - Number of iterations for the first restart.
    :param lr_scheduleer_T_mult: Learning rate scheduler - A factor increases T_{i}T after a restart. Default: 1.
    :param lbda: weight in generator loss
    :param input_channel: 3 if RGB or 1 if Gray
    :param output_channel: 3 if RGB or 1 if Gray
    :return: dict of hyperparameters
    """
    hparams = {
        "image_size": image_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "gpu": gpu,
        "generator_dropout_p": generator_dropout_p,
        "discriminator_dropout_p": discriminator_dropout_p,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "weight_decay": weight_decay,
        "lr_scheduler_T_0": lr_scheduler_T_0,
        "lr_scheduler_T_mult": lr_scheduler_T_mult,
        "device": device,
        "lambda": lbda,
        "input_channel":input_channel ,
        "output_channel": output_channel
    }
    return hparams