import torch
import pytorch_lightning as pl
from torchvision import transforms
import config
import argparse

#------------------- Training imports ---------------------
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.ParetoEarlyStopping import ParetoEarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

# ------------------- Model imports ---------------------
from models.NN import NN
from models.RegularizedFunctor import RegularizedFunctor
from models.InvolutionFunctor import InvolutionFunctor
from models.ClassifierInvolutionFunctor import ClassifierInvolutionFunctor
#from models.ComplexRegularizedFunctor import ComplexRegularizedFunctor
from models.Classifiers import LinearClassifier
from models.IteratorFunctor import IteratorFunctor
from models.FixedDFunctor import FixedDFunctor
from models.MultiWFunctor import MultiWFunctor
from models.MultiWFunctorMNIST import MultiWFunctor as MultiWFunctorMNIST
from models.ClassifierFixedDFunctor import ClassifierFixedDFunctor
from models.ExponentRegularizedFunctor import ExponentRegularizedFunctor
#from models.ComplexExponentRegularizedFunctor import ComplexExponentRegularizedFunctor
from models.FixedMultiWFunctor import FixedMultiWFunctor
from models.Classifiers import EncoderClassifierFunctor

# ------------------- Dataset imports ---------------------
from datasets.TMNIST_dataset import PairedTMNISTDataModule
from datasets.MNIST_dataset import PairedMNISTDataModule
from datasets.TMNIST_multiW_dataset import PairedMultiWTMNISTDataModule
from datasets.MNIST_covariates_dataset import PairedMNISTCovariatesDataModule
from datasets.MNIST_multiW_dataset import PairedMultiWMNISTDataModule
from datasets.ClassificationDataset import PairedClassificationDataModule
import medmnist
from medmnist import INFO


def get_dataset(args):
    if args.dataset == 'MNIST':
        dm = PairedMNISTDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                complex=complex,
                n_digits=args.n_digits,
                offset=args.offset
        )
    elif args.dataset == 'TMNIST':
        dm = PairedTMNISTDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                complex=complex,
                digits=args.digits,
                x2_transformation=args.x2_transformation,
                x2_angle=args.x2_angle
        )
    elif args.dataset == 'MultiWTMNIST':
        dm = PairedMultiWTMNISTDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                complex=complex,
        )
    elif args.dataset == 'MNIST_covariates':
        dm = PairedMNISTCovariatesDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                x2_angle=args.x2_angle
        )
    elif args.dataset == 'MultiWMNIST':
        dm = PairedMultiWMNISTDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                x2_angle=args.x2_angle
        )
    elif args.dataset == 'ClassificationDataset':
        dm = PairedClassificationDataModule(
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                transform=transforms.ToTensor(),
                val_split=config.VAL_SPLIT,
                x2_angle=args.x2_angle,
                dataset=args.data,
                data_flag=args.data_flag
        )
    return dm


def get_D_values(n_ones=0, n_neg_ones=0, mode='fixed', latent_dim=32):
    device = 'cuda' if config.ACCELERATOR == 'gpu' else 'cpu'
    if mode == 'random':
        D_values = torch.sign(torch.randn(latent_dim, device=device))

    if mode == 'fixed':
        assert n_ones + n_neg_ones == latent_dim
        D_values = torch.tensor([1.0 for _ in range(n_ones)] + [-1.0 for _ in range(n_neg_ones)], device=device)
    
    return D_values


def get_model_from_str(model_name, base_autoencoder=None, freeze_latents=True):
    if model_name == 'RegularizedFunctor':
        model_type = RegularizedFunctor
        model = RegularizedFunctor(lambda_t=lambda_t, 
                                   lambda_W=lambda_W, 
                                   latent_dim=args.latent_dim, 
                                   init_W=args.init_W, 
                                   W_exponent_algebra=args.W_exponent_algebra, 
                                   run_id=args.run_id,
                                   experiment_name=args.experiment_name)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={lambda_W}_init_W={args.init_W}_latent_dim={args.latent_dim}_reg={args.W_exponent_algebra}"
    
    elif model_name == 'InvolutionFunctor':
        model_type = InvolutionFunctor
        model = InvolutionFunctor(lambda_t=lambda_t, latent_dim=args.latent_dim)
        model_name = f"{model_name}_lambda_t={lambda_t}_latent_dim={args.latent_dim}"
    
    elif model_name == 'ComplexRegularizedFunctor':
        model_type = ComplexRegularizedFunctor
        model = ComplexRegularizedFunctor(lambda_t=lambda_t, lambda_W=lambda_W)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={lambda_W}"
    
    elif model_name == 'ClassifierInvolutionFunctor':
        model_type = ClassifierInvolutionFunctor
        model = ClassifierInvolutionFunctor(lambda_t=lambda_t, lambda_c=lambda_c)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_c={lambda_c}"
    
    elif model_name == 'IteratorFunctor':
        model_type = IteratorFunctor
        model = IteratorFunctor(lambda_t=lambda_t, alpha=alpha)
        model_name = f"{model_name}_lambda_t={lambda_t}_alpha={alpha}"
    
    elif model_name == 'FixedDFunctor':
        D_values = get_D_values(n_ones, n_neg_ones, mode, args.latent_dim)
        model_type = FixedDFunctor
        model = FixedDFunctor(lambda_t=lambda_t, D_values=D_values)
        model_name = f"{model_name}_lambda_t={lambda_t}_mode={mode}"
    
    elif model_name == 'ClassifierFixedDFunctor':
        D_values = get_D_values(n_ones, n_neg_ones, mode, args.latent_dim)
        model_type = ClassifierFixedDFunctor
        model = ClassifierFixedDFunctor(lambda_t=lambda_t, lambda_c=lambda_c, D_values=D_values)
        model_name = f"{model_name}_lambda_t={lambda_t}lambda_c={lambda_c}_mode={mode}"
    
    elif model_name == 'LinearClassifier':
        model_type = LinearClassifier
        model = LinearClassifier(autoencoder=base_autoencoder, freeze_latent=freeze_latents)
        model_name = 'LinearClassifier'
    
    elif model_name == 'MultiWFunctor':
        model_type = MultiWFunctor
        model = MultiWFunctor(lambda_t=lambda_t, lambda_W=args.lambda_W, lambda_comm=args.lambda_comm, latent_dim=args.latent_dim, run_id=args.run_id)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={args.lambda_W}_lambda_comm={args.lambda_comm}_latent_dim={args.latent_dim}"

    elif model_name == 'MultiWFunctorMNIST':
        model_type = MultiWFunctorMNIST
        model = MultiWFunctorMNIST(lambda_t=lambda_t, lambda_W=args.lambda_W, lambda_comm=args.lambda_comm, latent_dim=args.latent_dim, run_id=args.run_id)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={args.lambda_W}_lambda_comm={args.lambda_comm}_latent_dim={args.latent_dim}"

    elif model_name == 'FixedMultiWFunctor':
        model_type = FixedMultiWFunctor
        model = FixedMultiWFunctor(lambda_t=lambda_t, lambda_W=args.lambda_W, lambda_comm=args.lambda_comm, latent_dim=args.latent_dim, 
                                   change_of_coords=args.change_of_coords, run_id=args.run_id, )
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={args.lambda_W}_lambda_comm={args.lambda_comm}_latent_dim={args.latent_dim}_changecoords={args.change_of_coords}"
    
    elif model_name == 'ExponentRegularizedFunctor':
        model_type = ExponentRegularizedFunctor
        model = ExponentRegularizedFunctor(lambda_t=lambda_t, lambda_W=lambda_W, W_exponent_algebra=args.W_exponent_algebra, latent_dim=args.latent_dim, run_id=args.run_id)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={lambda_W}_latent_dim={args.latent_dim}_reg={args.W_exponent_algebra}"

    elif model_name == 'ComplexExponentRegularizedFunctor':
        model_type = ComplexExponentRegularizedFunctor
        model = ComplexExponentRegularizedFunctor(lambda_t=lambda_t, lambda_W=lambda_W, W_exponent_algebra=args.W_exponent_algebra, latent_dim=args.latent_dim, run_id=args.run_id)
        model_name = f"{model_name}_lambda_t={lambda_t}_lambda_W={lambda_W}_latent_dim={args.latent_dim}_reg={args.W_exponent_algebra}"

    elif model_name == 'EncoderClassifierFunctor':
        model_type = EncoderClassifierFunctor
        model = EncoderClassifierFunctor(latent_dim=args.latent_dim, x2_angle=args.x2_angle, lambda_W=args.lambda_W, lambda_t=args.lambda_t, run_id=args.run_id)
        model_name = f"{model_name}_latent_dim={args.latent_dim}_lambda_t={args.lambda_t}_lambda_W={args.lambda_W}"

    elif model_name == 'MedMNISTClassifierFunctor':
        info = INFO[args.data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
    
    return model_type, model, model_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dataset", default='MultiWMNIST', type=str, help="Dataset to use")
    parser.add_argument("--data", choices=['MNIST', 'CIFAR', 'MedMNIST'], default='MNIST', type=str, help="Dataset to use")
    parser.add_argument("--data_flag", default='pathmnist', type=str, help="Data flag for MedMNIST")
    parser.add_argument("--n_digits", default=10, type=int, help="Include MNIST digits from 0 to n_digits-1")
    parser.add_argument("--digits", default=None, type=list, help="What digits to include in MNIST. If None, include all")
    parser.add_argument("--offset", default=1, type=int, help="offset = |y2-y1|")
    parser.add_argument("--x2_transformation", default='rotation', type=str, help="Transformation to apply to x2")
    parser.add_argument("--x2_angle", default=None, type=float, help="Angle to rotate x2 by if x2_transformation is 'rotation'")

    # Autoencoder model parameters
    parser.add_argument("--model", default="MultiWFunctor", type=str, help="Which model to use")
    parser.add_argument("--name_prefix", default='', type=str, help="Suffix to add to the model name")
    parser.add_argument("--init_W", default='random', type=str, help="ONLY FOR REGULARIZED FUNCTOR. How to initialise W matrix")
    parser.add_argument("--lambda_t", default=0.5,  type=float, help="Weight of the transformation loss")
    parser.add_argument("--lambda_W", default=0.5,  type=float, help="Weight of the W algebra loss")
    parser.add_argument("--lambda_c", default=0.01,  type=float, help="Weight of the classification loss")
    parser.add_argument("--lambda_comm", default=0.05,  type=float, help="Weight of the commutativity loss in MultiWFunctor")
    parser.add_argument("--W_exponent_algebra", default=3,  type=int, help="Exponent of the W matrix for the algebra loss in RegularizedFunctor")
    parser.add_argument("--alpha", default=0.0001, type=float, help="Weighting of the iterator optimisation")
    parser.add_argument("--latent_dim", default=32, type=int, help="Latent dimension of the autoencoder")
    parser.add_argument("--change_of_coords", action='store_true', help="Whether to use change of coordinates in FixedMultiWFunctor")

    parser.add_argument("--experiment_name", default='rotations', type=str, help="Name of the experiment")
    parser.add_argument("--run_id", default=None, type=int, help="Run ID for the experiment")

    # Autoencoder training parameters
    parser.add_argument("--no_autoencoder_train", action='store_true', help="Do not train, only load model specified in args.model_path")
    parser.add_argument("--no_autoencoder_test", action='store_true', help="Do not test the autoencoder")
    parser.add_argument("--model_path", default='', type=str, help="Path to the model weights to load in ./checkpoint/ without .ckpt extension")
    parser.add_argument("--dev_run", action='store_true')
    parser.add_argument("--early_stopping", default='standard', type=str, help="Early stopping mechanism")

    # Classifier model parameters
    parser.add_argument("--classifier_name", default='LinearClassifier', type=str, help="Classifier to use if needed")

    # Classifier training parameters
    parser.add_argument("--classifier_on_latent", action='store_true', help="Whether to train a classifier on top of the latent space of the autoencoder")
    parser.add_argument("--freeze_latents", action='store_true', help="Whether to freeze the weights of the autoencoder when classifier_on_latent is true")
    parser.add_argument("--classifier_path", default='', type=str, help="Path to the classifier weights")
    parser.add_argument("--no_classifier_train", action='store_true', help="Do not train, only load model specified in args.classifier_path")
    parser.add_argument("--no_classifier_test", action='store_true', help="Do not test the classifier")

    # Fixed_D_parameters (For FixedDFunctor and ClassifierFixedDFunctor)
    parser.add_argument("--n_ones", default=0,  type=int, help="Num of 1s in fixed D if mode is 'fixed'")
    parser.add_argument("--n_neg_ones", default=0,  type=int, help="Num of -1s in fixed D if mode is 'fixed'")
    parser.add_argument("--mode", default='fixed',  type=str, help="How to fill the fixed D matrix. Choose between 'fixed' and 'random'")


    args = parser.parse_args()

    if args.no_autoencoder_train:
        best_autoencoder_model_path = f'checkpoints/{args.model_path}.ckpt'


    # General parameters
    lambda_t = args.lambda_t
    lambda_W = args.lambda_W
    lambda_c = args.lambda_c
    alpha = args.alpha
    model_name = args.model
    complex = True if 'Complex' in model_name else False 
    if complex:
        print("USING COMPLEX")

    # Parameters for fixed D
    n_ones = args.n_ones
    n_neg_ones = args.n_neg_ones
    mode = args.mode

    model_type, model, model_name = get_model_from_str(model_name)


    logger = TensorBoardLogger('tb_logs', name=args.name_prefix + model_name + f'_run_id={args.run_id}')
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )


    dm = get_dataset(args)
    
    if args.early_stopping == 'standard':
        early_stopping_callback = EarlyStopping(
            monitor='validation_loss',
            verbose=True,
            patience=10,
            min_delta=0.001,
        )
    
    elif args.early_stopping == 'pareto':
        early_stopping_callback = ParetoEarlyStopping(
            monitor=['validation_loss', 'validation_transformation_fidelity'],
            verbose=True,
            patience=10,
            min_delta=0.001,
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='checkpoints/',
        filename=f"best_{model_name}",
        save_top_k=1, 
        mode='min',
        save_weights_only=True
    )

    trainer = pl.Trainer(
        #profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=90,
        precision=config.PRECISION,
        callbacks=[checkpoint_callback, early_stopping_callback],
        fast_dev_run=args.dev_run
    )

    model.print_hyperparameters()

    if not args.no_autoencoder_train:
        trainer.fit(model, dm)
        best_autoencoder_model_path = checkpoint_callback.best_model_path   

    print(f"Loading best model from {best_autoencoder_model_path}")
    best_autoencoder_model = model_type.load_from_checkpoint(best_autoencoder_model_path)

    if not args.no_autoencoder_test:
        results = trainer.test(best_autoencoder_model, dm)
        save_path = f'{best_autoencoder_model.save_dir}/losses_{best_autoencoder_model.run_id}.pkl'
        import pickle
        with open(save_path, 'wb') as f:
            print("Saving results to ", save_path)
            pickle.dump(results, f)



    # Logic for classifiers on latent
    if args.classifier_on_latent:
        print(f"#"*200)
        print(f"Training the classifier {args.classifier_name} on the frozen latents")

        classifier_type, classifier, classifier_name = get_model_from_str(args.classifier_name, 
                                                                          base_autoencoder=best_autoencoder_model, 
                                                                          freeze_latents=args.freeze_latents)

        classifier_checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss',
            dirpath='checkpoints/',
            filename=f"best_{classifier_name}",
            save_top_k=1, 
            mode='min',
            save_weights_only=True
        )
        classifier_trainer = pl.Trainer(
            #profiler=profiler,
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=1,
            max_epochs=50,
            precision=config.PRECISION,
            callbacks=[classifier_checkpoint_callback, early_stopping_callback],
            fast_dev_run=args.dev_run
        )

        classifier_trainer.fit(classifier, dm)
        best_classifier_path = classifier_checkpoint_callback.best_model_path  # Contains weights of autoencoder too

        print(f"Loading best classifier model from {best_classifier_path} for testing")
        best_classifier = classifier_type.load_from_checkpoint(best_classifier_path, 
                                                        autoencoder=model,
                                                        freeze_latents=args.freeze_latents)

        classifier_trainer.test(best_classifier, dm)

