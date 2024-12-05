# imports
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
import os
from torchvision.datasets import CIFAR10, CIFAR100, Food101
from torch.utils.data import DataLoader
import torchvision.models as models
from ray import tune, train
from ray.tune import CLIReporter
from filelock import FileLock
import utils
from models import resnet
from torchvision import transforms



# runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-type', type=str, required=True, help='Type of experiment: {tuning, training}')
parser.add_argument('--description', default='', type=str, help='short description of the experiment')
parser.add_argument('--num-epochs', type=int, required=True, help='Number of epochs for training')
parser.add_argument('--num-trials', type=int, required=True, help='Number of trials')
parser.add_argument('--model', type=str, help='name of the model to be used for the experiment')
parser.add_argument('--dataset', type=str, help='name of the dataset to be used for the experiment')
parser.add_argument('--seed-list', type=int, nargs='+', default=42, help='list of seeds for trials')
parser.add_argument('--base-hp-experiment-path', type=str, required=False, help='Path to the results_dir of the base HP experiment, to use its best trial weights and hyperparameters')
parser.add_argument('--weight-initialization', type=str, required=False, help='Weight initialization policy: {random, hp-init, hp-final, hp-epoch, imagenet}')
parser.add_argument('--hp-epoch', type=int, required=False, default=-1, help='Load weights from the specified epoch of the winning HP trial')
parser.add_argument('--batch-size', type=int, required=False, default=128, help='Batch size for training')
args = parser.parse_args()


# set up the experiment configuration based on the runtime arguments
experiment_type = args.experiment_type
num_epochs = args.num_epochs
num_trials = args.num_trials
seed_list = args.seed_list
batch_size = args.batch_size


# configuration related to training experiments

if experiment_type == 'training':
    base_results_path = args.base_hp_experiment_path
    weight_initialization = args.weight_initialization
    weight_epoch = -1
    if weight_initialization == 'hp-epoch':
        weight_epoch = args.hp_epoch

# reproducibility functions
# note that complete reproducibility is not guaranteed
# more information can be found at https://pytorch.org/docs/stable/notes/randomness.html

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# TODO double check determinism
# For gpu deterministic behavior. Can't be pickled so have to set in main()
def set_determinism(is_deterministic=False):
    if is_deterministic == True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def tune_hyperparameters(config):



    trial_id = train.get_context().get_trial_id()
    trial_number = int(trial_id.split("_")[-1])
    seed = seed_list[trial_number]
    config["seed"] = seed
    set_seeds(seed)

    save_dir = os.getcwd() + '/hpweights'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    experiment_name=f"{args.description}-trial_{trial_id}"

    # UNCOMMENT TO USE WANDB
    # # Initialize wandb
    # wandb.init(project="tuning-weights", entity="YOUR_TEAM", name=experiment_name)
    # wandb.config.update(config)

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    num_classes = 0
    if args.dataset.lower() in ['cifar10', 'cifar100', 'tiny-imagenet', 'food101']:

        with FileLock(os.path.expanduser("~/.data.lock")):
            if args.dataset.lower() == 'cifar10':
                transform_train, _ = utils.get_cifar10_transforms()
                train_dataset = CIFAR10(root="~/datasets", train=True,  transform=transform_train, download=True)
                num_classes = 10
            elif args.dataset.lower() == 'cifar100':
                transform_train, _ = utils.get_cifar100_transforms()
                train_dataset = CIFAR100(root="~/datasets", train=True, transform=transform_train, download=True)
                num_classes = 100

            elif args.dataset.lower() == 'tiny-imagenet':
                pass
            elif args.dataset.lower() == 'food101':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                train_dataset = Food101(root="~/datasets", split="train", transform=transform_train, download=True)
                num_classes = 101
            
        ### ADD OTHER DATASETS HERE

    else:
        raise RuntimeError(
            "Specified dataset is not supported or cannot be found. Make sure you pass a correct --dataset runtime argument. \n The experiment will be terminated."
        )
    
    # random split to train and val
    # generator for reproducible random_split()
    generator1 = torch.Generator().manual_seed(seed)

    # tiny-imagenet has its own dataloaders
    if args.dataset.lower() == 'tiny-imagenet':
        tiny_dataloaders = utils.get_tiny_imagenet_dataloaders(batch_size=batch_size)
        train_loader = tiny_dataloaders["train"]
        val_loader = tiny_dataloaders["val"]
        num_classes = 200
    else:
        # e.g. for cifar-10 and cifar-100, we need manual splits
        validation_ratio = 0.2
        num_train = len(train_dataset)
        num_val = int(validation_ratio * num_train)
        num_train = num_train - num_val

        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val], generator=generator1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False)



    # Initialize model with random weights (no pre-training)
    if args.model.lower() == 'resnet18':
        model = resnet.ResNet18(num_classes=num_classes)
        if args.dataset.lower() == 'tiny-imagenet':
            model = models.resnet18(pretrained=False)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 200)
    elif args.model.lower() == 'resnet152':
        model = resnet.ResNet152(num_classes=num_classes)
    elif args.model.lower() == 'inception_v3':
        model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)

    ### ADD OTHER MODELS HERE

    else:
        raise RuntimeError(
            "Specified model is not supported or cannot be found. Make sure you pass a correct --model runtime argument. \n The experiment will be terminated."
        )
    
    model = model.to(device)

    # save the initial weights of the model
    torch.save(model.state_dict(), save_dir+f"/trial_{trial_id}_initial_weights.pth")
   
    criterion = torch.nn.CrossEntropyLoss()

    if args.model.lower() == 'inception_v3':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"], 
                                    momentum=config["momentum"], 
                                    weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])
    
    elif args.model.lower() == 'resnet18' or args.model.lower() == 'resnet152':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"], 
                                    momentum=0.9,
                                    weight_decay=config["decay_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
 

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss= 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        train_accuracy = 100. * correct / total
        train_loss /= len(train_loader.dataset)



        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print('Validation Loss: %.3f | Acc: %.3f%% (%d/%d)' % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
        val_accuracy = 100. * correct / total
        val_loss /= len(val_loader.dataset)

        # UNCOMMENT TO USE WANDB    
        # # Log metrics to wandb
        # wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy, "learning_rate": optimizer.param_groups[0]['lr']})
        # Update learning rate scheduler
        scheduler.step()

        # Report metrics to Ray
        # tune.report(loss=val_loss, accuracy=val_accuracy)
        train.report({'loss': val_loss, 
                      'accuracy': val_accuracy})
        # Save weights of the epochs
        torch.save(model.state_dict(), save_dir+f"/trial_{trial_id}_epoch_{epoch}_weights.pth")
    
    # duplicate saving of last epoch's weights but it does not matter for now, backward compatibility
    torch.save(model.state_dict(), save_dir+f"/trial_{trial_id}_final_weights.pth")
    
    # UNCOMMENT TO USE WANDB
    # # Close wandb
    # wandb.finish()

def train_model(config):



    trial_id = train.get_context().get_trial_id()
    set_seeds(config["seed"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    experiment_name=f"{args.description}-{weight_initialization}-{weight_epoch}-trial_{trial_id}"

    # UNCOMMENT TO USE WANDB
    # # add weight initialization to config
    # # Initialize wandb
    # wandb.init(project="tuning-weights", entity="dcatkh", name=experiment_name)
    # wandb.config.update(config)
    # wandb.config.update(args)
    
    # First intialize the dataset and then set the num_classes to be passed to model init
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100' or args.dataset.lower() == 'tiny-imagenet' or args.dataset.lower() == 'food101':

        with FileLock(os.path.expanduser("~/.data.lock")):
            num_classes = 0
            if args.dataset.lower() == 'cifar10':
                
                transform_train, transform_test = utils.get_cifar10_transforms()
                train_dataset = CIFAR10(root="~/datasets", train=True, transform=transform_train, download=True)
                test_dataset = CIFAR10(root="~/datasets", train=False, transform=transform_test)
                num_classes = 10
                # TODO check cifar100 transforms
            elif args.dataset.lower() == 'cifar100':
                # use the same transformation for CIFAR100 as CIFAR10
                transform_train, transform_test = utils.get_cifar10_transforms()
                train_dataset = CIFAR100(root="~/datasets", train=True, transform=transform_train, download=True)
                test_dataset = CIFAR100(root="~/datasets", train=False, transform=transform_test)
                num_classes = 100
            
            elif args.dataset.lower() == 'tiny-imagenet':
                pass
            elif args.dataset.lower() == 'food101':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                transform_test = transforms.Compose([
                transforms.Resize((299, 299)),  # inception_v3 specific   
                transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                train_dataset = Food101(root="~/datasets", split="train", transform=transform_train, download=True)
                test_dataset = Food101(root="~/datasets", split="test", transform=transform_test, download=True)
                num_classes = 101

            ### ADD OTHER DATASETS HERE

    else:
        raise RuntimeError(
            "Specified dataset is not supported or cannot be found. Make sure you pass a correct --dataset runtime argument. \n The experiment will be terminated."
        )
    
        # tiny-imagenet has its own dataloaders
    if args.dataset.lower() == 'tiny-imagenet':
        tiny_dataloaders = utils.get_tiny_imagenet_dataloaders(batch_size=batch_size)
        train_loader = tiny_dataloaders["train"]
        test_loader = tiny_dataloaders["test"]
        num_classes = 200
    else:

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    # find the model weights file

    path_to_weights = best_dir+"/hpweights/"
    weights_prefix =  path_to_weights + f"trial_{best_trial.trial_id}_"
    # random, hp-init, hp-final, hp-epoch
    if weight_initialization == "hp-init":
        weights_path = weights_prefix + "initial_weights.pth"
    elif weight_initialization == "hp-final":
        weights_path = weights_prefix + "final_weights.pth"
    elif weight_initialization == "hp-epoch":
        weights_path = weights_prefix + f"epoch_{weight_epoch}_weights.pth"
        

    # Initialize model with random or pretrained weights
    if args.model.lower() == 'resnet18':
        if weight_initialization == "imagenet":
            weights='IMAGENET1K_V1'
        # weight initialization
        model = resnet.ResNet18(num_classes=num_classes)

        if args.dataset.lower() == 'tiny-imagenet':
            model = models.resnet18(pretrained=False)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 200)
        # model = resnet18(weights=weights)
        # model.fc = nn.Linear(512, num_classes)
        model.to(device)

    elif args.model.lower() == 'resnet152':
        if weight_initialization == "imagenet":
            weights='IMAGENET1K_V1'
        # weight initialization
        model = resnet.ResNet152(num_classes=num_classes)
        # model.fc = nn.Linear(2048, num_classes)
        model.to(device)

    elif args.model.lower() == 'inception_v3':
        model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)
        model.to(device)


    ### ADD OTHER MODELS HERE

    else:
        raise RuntimeError(
            "Specified model is not supported or cannot be found. Make sure you pass a correct --model runtime argument. \n The experiment will be terminated."
        )

    # if another weight initialization strategy is required:
    #TODO rewrite with enum
    if weight_initialization in ["hp-init", "hp-final", "hp-epoch"]:
        model.load_state_dict(torch.load(weights_path))
        print(f"***** LOADED WEIGHTS FROM {weights_path}")

    # Initialize criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if args.model.lower() == 'inception_v3':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"], 
                                    momentum=config["momentum"], 
                                    weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])
    
    elif args.model.lower() == 'resnet18' or args.model.lower() == 'resnet152':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"], 
                                    momentum=0.9,
                                    weight_decay=config["decay_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        train_accuracy = 100. * correct / total
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / total
        
        # UNCOMMENT TO USE WANDB
        # # Log metrics to wandb
        # wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_accuracy, "learning_rate": optimizer.param_groups[0]['lr']})
        # # always call wandb.log() once! send all metrics in a single call, else wandb's global step will be messed up
        # Report metrics to Ray
        # tune.report(loss=test_loss, accuracy=test_accuracy)
        train.report({'loss': test_loss, 
                      'accuracy': test_accuracy})
        
        # Update learning rate scheduler
        # scheduler.step()
        scheduler.step()

    # UNCOMMENT TO USE WANDB
    # # Close wandb
    # wandb.finish()


# main block


if __name__ == '__main__':

    # set determinicism
    set_determinism(True)


    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    if experiment_type == 'tuning':
        config_inception_v3 = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.uniform(0.8, 0.99),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "T_max": tune.choice([50, 100, 200]),
            # "batch_size": tune.choice([32, 64, 128])
        }

        config_other = {
            "lr": tune.sample_from(lambda _: np.random.choice([0.01, 0.03, 0.05, 0.1, 0.2, 0.3])),
            "decay_rate": tune.sample_from(lambda _: np.random.choice([0.0003, 0.001, 0.003])),
            # "batch_size": tune.sample_from(lambda _: np.random.choice([128, 256]))
        }
 
        if args.model.lower() == 'inception_v3':
            config = config_inception_v3
        else:
            config = config_other

        result = tune.run(
            tune_hyperparameters,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=config,
            num_samples=num_trials,
            progress_reporter=reporter
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    elif experiment_type == 'training':
        # Load the Ray Tune results for the specific trial

        trial_analysis = tune.ExperimentAnalysis(base_results_path)
        
        best_trial = trial_analysis.get_best_trial(metric="accuracy", mode="max")
        best_dir = best_trial.logdir

        # Extract the hyperparameters from the trial's configuration
        hyperparameters = best_trial.config

        # Run trials in parallel on multiple GPUs
        trial_resources = {
            "cpu": 2,  # Number of CPU cores per trial
            "gpu": 1  # Number of GPUs per trial
        }

        # this parameter space can be used to construct a list of trials

        if args.model.lower() == 'inception_v3':
            param_space = {
                # "params":{
                    'lr': hyperparameters['lr'],
                    'momentum': hyperparameters['momentum'],
                    'weight_decay': hyperparameters['weight_decay'],
                    'T_max': hyperparameters['T_max'],
                    'seed': tune.grid_search([
                        i for i in seed_list
                    ])
                # }
            }
        elif args.model.lower() == 'resnet18' or args.model.lower() == 'resnet152':
            param_space = {
                # "params":{
                    'lr': hyperparameters['lr'],
                    # 'batch_size': hyperparameters['batch_size'],
                    'decay_rate': hyperparameters['decay_rate'], # only for resnet18 and resnet152

                    'seed': tune.grid_search([
                        i for i in seed_list
                    ])
                # }
            }

        trainable_with_gpu = tune.with_resources(train_model,
                                                {"cpu": 1, "gpu": 1},
                                                )

        tuner = tune.Tuner(trainable=trainable_with_gpu,
                        param_space=param_space,
        )
        results = tuner.fit()