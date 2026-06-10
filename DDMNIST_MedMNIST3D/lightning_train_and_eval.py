import os
import pytorch_lightning as pl
import argparse
import time
from medmnist import INFO
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.MedMNIST2D_dataset import MedMNISTDataModule
from datasets.CnMedMNIST2D_dataset import CnMedMNISTDataModule
from datasets.Z2MedMNIST2D_dataset import Z2MedMNISTDataModule
from datasets.D4MedMNIST2D_dataset import D4MedMNISTDataModule
from datasets.D8MedMNIST2D_dataset import D8MedMNISTDataModule
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.S4MedMNIST3D_dataset import S4MedMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule
from datasets.D1MedMNIST3D_dataset import D1MedMNISTDataModule
from models.MedMNISTVanilla import MedMNISTModel
from models.FunctorModel import FunctorModel
from models.D4RegularFunctorModel import D4RegularFunctor
from models.D8RegularFunctorModel import D8RegularFunctor
from models.GxGRegularFunctorModel import GxGRegularFunctor
from models.MedMNISTGRegularFunctorModel import MedMNISTGxGRegularFunctor
import torch

class EpochTimerCallback(Callback):
    def __init__(self, fast_mode=False):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_times = []
        self.fast_mode = fast_mode
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Monitor GPU memory usage and print to console
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                print(f"Epoch {trainer.current_epoch}: {epoch_time:.2f}s, GPU Memory: {gpu_memory:.2f}GB")
            else:
                print(f"Epoch {trainer.current_epoch}: {epoch_time:.2f}s")
            
            # Clear GPU cache every 10 epochs to prevent memory fragmentation
            if trainer.current_epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"  -> Cleared GPU cache at epoch {trainer.current_epoch}")
            
            # Log to TensorBoard based on mode
            if not self.fast_mode and trainer.logger:
                # Log every epoch in normal mode
                trainer.logger.log_metrics({"epoch_time_seconds": epoch_time}, step=trainer.current_epoch)
            elif self.fast_mode and trainer.current_epoch % 10 == 0 and trainer.logger:
                # Log every 10 epochs in fast mode
                trainer.logger.log_metrics({"epoch_time_seconds": epoch_time}, step=trainer.current_epoch)

def get_dataset_from_args(args):
    if args.dataset == 'pairedcn':
        return CnMedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download, args.x2_angle, args.fixed_covariate)
    if args.dataset == 'vanilla':
        return MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    if args.dataset == 'z2':
        return Z2MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    if args.dataset == 'd4':
        return D4MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    if args.dataset == 'd8':
        return D8MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    if args.dataset == 'ddmnist_c4':
        dm = C4xC4DDMNISTDataModule(args.batch_size)
        dm.num_workers = args.num_workers
        dm.pin_memory = getattr(args, 'pin_memory', False)
        return dm
    if args.dataset == 'ddmnist_d4':
        dm = D4xD4DDMNISTDataModule(args.batch_size)
        dm.num_workers = args.num_workers
        dm.pin_memory = getattr(args, 'pin_memory', False)
        return dm
    if args.dataset == 'ddmnist_d1':
        dm = D1xD1DDMNISTDataModule(args.batch_size)
        dm.num_workers = args.num_workers
        dm.pin_memory = getattr(args, 'pin_memory', False)
        return dm
    if args.dataset == 'S4':
        return S4MedMNISTDataModule(args.data_flag, args.batch_size, resize=False, as_rgb=False, size=28, download=True)
    if args.dataset == 'D1':
        return D1MedMNISTDataModule(args.data_flag, args.batch_size, resize=False, as_rgb=False, size=28, download=True)
    raise NotImplementedError


def get_model_from_args(args, milestones=None, log_dir=None, n_channels=None, n_classes=None, task=None):
    if args.model == 'vanilla':
        return MedMNISTModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run,
                          milestones=milestones, output_root=log_dir)
    
    elif args.model == 'functor':
        return FunctorModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run, device=args.device, log_inputs=args.log_inputs,
                          milestones=milestones, output_root=log_dir, lambda_c=args.lambda_c, 
                          lambda_t=args.lambda_t, lambda_W=args.lambda_W, algebra_loss_criterion=args.algebra_loss_criterion,
                          W_init=args.W_init, fix_rep=args.fix_rep, W_block_size=args.block_size,
                          latent_transform_process=args.latent_transform_process, modularity_exponent=args.modularity_exponent,
                          lr=args.lr, gamma=args.gamma)
    elif args.model == 'D4regularfunctor':
        return D4RegularFunctor(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run, device=args.device,
                          milestones=milestones, output_root=log_dir, lambda_c=args.lambda_c,
                          lambda_t=args.lambda_t,
                          lr=args.lr, gamma=args.gamma)

    elif args.model == 'D8regularfunctor':
        return D8RegularFunctor(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run, device=args.device,
                          milestones=milestones, output_root=log_dir, lambda_c=args.lambda_c,
                          lambda_t=args.lambda_t,
                          lr=args.lr, gamma=args.gamma)
    elif args.model == 'GxGregularfunctor':
        return GxGRegularFunctor(lr=args.lr, gamma=args.gamma, milestones=milestones, output_root=log_dir, device=args.device, group=args.group,
                                   lambda_c=args.lambda_c, lambda_t=args.lambda_t, lambda_e=args.lambda_e, equivariant_layer_id=args.equivariant_layer_id, data_flag=args.data_flag,
                                   entanglement_type=args.entanglement_type)
    elif args.model == 'MedMNISTGxGregularfunctor':
        return MedMNISTGxGRegularFunctor(lr=args.lr, gamma=args.gamma, milestones=milestones, output_root=log_dir, device=args.device, group=args.group,
                                   lambda_c=args.lambda_c, lambda_t=args.lambda_t, equivariant_layer_id=args.equivariant_layer_id, data_flag=args.data_flag,)
    raise NotImplementedError


def get_args():
    parser = argparse.ArgumentParser(description='Train MedMNIST model with PyTorch Lightning')

    parser.add_argument('--dataset', type=str, default='z2')
    parser.add_argument('--data_flag', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='tb_logs', help='Where to save logs')
    parser.add_argument('--base_no_gshift', type=bool, default=True, help='Whether to use the gshift in the base model (on by default in the git branch ablate_gshit_baseline)')
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--not_rgb', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--x2_angle', type=float, default=90.0, help='Angle to rotate the second image in paired datasets')
    parser.add_argument('--fixed_covariate', type=int, default=None, help='Fixed covariate for paired datasets')

    parser.add_argument('--model_flag', type=str, default='resnet18')
    parser.add_argument('--model', type=str, default='functor')
    parser.add_argument('--lr', type=float, default=0.001) # Default DDMNIST: 5e-5
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--lambda_c', type=float, default=1)
    parser.add_argument('--lambda_t', type=float, default=0.5)
    parser.add_argument('--lambda_e', type=float, default=0.0, help='Weight for entanglement loss')
    parser.add_argument('--entanglement_type', type=str, default='bipartite', help='Entanglement type: bipartite/tripartite')
    parser.add_argument('--lambda_W', type=float, default=0.1)
    parser.add_argument('--latent_transform_process', type=str, default='from_generators')
    parser.add_argument('--W_init', type=str, default='orthogonal')
    parser.add_argument('--block_size', type=int, default=32, help="Size of W when using block diagonal initialisation")
    parser.add_argument('--fix_rep', action='store_true')
    parser.add_argument('--algebra_loss_criterion', type=str, default='mse')
    parser.add_argument('--log_inputs', action='store_true', help='Log inputs to TensorBoard')
    parser.add_argument('--equivariant_layer_id', type=int, nargs="+", default=[9], help='Layer ID of the equivariant layer in the model')
    parser.add_argument('--group', type=str, required=True, help='Group for the model. Can be C4xC4 or D4xD4')

    parser.add_argument('--num_epochs', type=int, default=150) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers (auto-detected if None)')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin_memory in DataLoaders for faster GPU transfer')
    
    parser.add_argument('--run', type=str, default='model1')
    parser.add_argument('--visible_gpus', type=str, default='0,1,2,3')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--fast', action='store_true', help='Disable expensive logging for speed')
    return parser.parse_args()


def run(args):
    """Run a single training + test cycle. Returns the list of test result dicts."""
    import copy
    args = copy.deepcopy(args)

    pl.seed_everything(args.seed, workers=True)

    # Optimize DataLoader settings for speed
    if args.num_workers is None:
        # Auto-detect optimal number of workers
        import os
        cpu_count = os.cpu_count()
        args.num_workers = min(cpu_count, 8) if args.fast else min(cpu_count // 2, 6)
    
    # Enable optimizations for faster data loading
    if args.fast:
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        args.pin_memory = True  # Force pin_memory in fast mode
        
    print(f"\n{'='*60}")
    print(f"Running with seed={args.seed}")
    print(f"{'='*60}")
    print(f"Using {args.num_workers} dataloader workers")
    print(f"Pin memory: {args.pin_memory}")

    if args.gpu_id != '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print("Using device: ", device)

    args.as_rgb = True if not args.not_rgb else False
    print("Using RGB: ", args.as_rgb)

    # WARNING: A BIT HACKY BUT GXGFUNCTOR DOESN'T NEED THEM
    if args.model == "GxGregularfunctor":
        n_channels = None
        n_classes = None
        task = None

    if args.data_flag is not None:
        info = INFO[args.data_flag]
        task = info['task']
        n_channels = 3 if args.as_rgb else info['n_channels']
        n_classes = len(info['label'])
    

    milestones = [int(0.5 * args.num_epochs), int(0.75 * args.num_epochs)]
    
    ###################################### logger and checkpoints #####################################
    model_name = f'{args.run}_{args.model_flag}_lambdaT_{args.lambda_t}{'_basenogshift' if args.base_no_gshift else ""}_lambdaE_{args.lambda_e}_Etype_{args.entanglement_type}_lambdaW_{args.lambda_W}_lr_{args.lr}_numepochs_{args.num_epochs}'
    logger = TensorBoardLogger(
        save_dir=args.output_root,
        name=f'{args.data_flag}/{args.dataset}/{model_name}'
    )
    log_dir = logger.log_dir

    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    ###################################### data_module #####################################
    data_module = get_dataset_from_args(args)
    
    # Apply DataLoader optimizations
    if hasattr(data_module, 'num_workers'):
        data_module.num_workers = args.num_workers
    if hasattr(data_module, 'pin_memory'):
        data_module.pin_memory = getattr(args, 'pin_memory', False)
    
    # Monkey patch DataLoader methods to use optimized settings
    if args.fast:
        original_train_dataloader = data_module.train_dataloader
        original_val_dataloader = data_module.val_dataloader
        original_test_dataloader = data_module.test_dataloader
        
        def optimized_train_dataloader():
            loader = original_train_dataloader()
            # Create new optimized DataLoader
            from torch.utils.data import DataLoader
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=getattr(args, 'pin_memory', False),
                persistent_workers=True,  # Keep workers alive between epochs
                prefetch_factor=2  # Prefetch batches for speed
            )
        
        def optimized_val_dataloader():
            loader = original_val_dataloader()
            from torch.utils.data import DataLoader
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=getattr(args, 'pin_memory', False),
                persistent_workers=True,
                prefetch_factor=2
            )
        
        def optimized_test_dataloader():
            loader = original_test_dataloader()
            from torch.utils.data import DataLoader
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=getattr(args, 'pin_memory', False),
                persistent_workers=True,
                prefetch_factor=2
            )
        
        data_module.train_dataloader = optimized_train_dataloader
        data_module.val_dataloader = optimized_val_dataloader
        data_module.test_dataloader = optimized_test_dataloader
    
    if args.dataset == 'pairedcn':
        assert 360 % args.x2_angle == 0, "x2_angle must be a divisor of 360"
        args.modularity_exponent = int(360 / int(args.x2_angle))
    elif args.dataset == 'z2':
        args.modularity_exponent = 2

    ###################################### model #####################################
    model = get_model_from_args(args, milestones=milestones, log_dir=log_dir, n_channels=n_channels, n_classes=n_classes, task=task).to(device)
    model.print_hyperparameters()

    ###################################### callbacks #####################################
    early_stop_callback = EarlyStopping(monitor='val_acc', mode='max', patience=args.patience, verbose=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', 
        mode='max', 
        save_top_k=1,
        dirpath=checkpoints_dir,  # Use the version-specific checkpoints directory
        filename='best_model',
        save_weights_only=True
    )

    # Timer callback to measure training time per epoch
    timer_callback = Timer(duration=None, interval="epoch")
    
    # Custom epoch timer callback for detailed per-epoch timing
    epoch_timer_callback = EpochTimerCallback(fast_mode=args.fast)

    
    ###################################### trainer #####################################
    #try:
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[int(args.gpu_id)],
        default_root_dir=log_dir,
        callbacks=[early_stop_callback, checkpoint_callback, timer_callback, epoch_timer_callback],
        logger=logger,
        log_every_n_steps=200 if args.fast else 31,
        enable_progress_bar=not args.fast,
        enable_model_summary=not args.fast
    )
    trainer.fit(model, data_module)

    ####################### test the best model #####################################
    best_model_path = checkpoint_callback.best_model_path
    print("Loading model from best checkpoint: ", best_model_path)
    model = type(model).load_from_checkpoint(best_model_path)
    if args.dataset == 'ddmnist_c4':
        data_module_augmented_test = C4xC4DDMNISTDataModule(256, augment_test=True)
        data_module_augmented_test.setup(stage='test')
        results = trainer.test(model, dataloaders=[data_module.test_dataloader(), data_module_augmented_test.test_dataloader()])
    elif args.dataset == 'ddmnist_d4':
        data_module_augmented_test = D4xD4DDMNISTDataModule(256, augment_test=True)
        data_module_augmented_test.setup(stage='test')
        results = trainer.test(model, dataloaders=[data_module.test_dataloader(), data_module_augmented_test.test_dataloader()])
    elif args.dataset == 'ddmnist_d1':
        data_module_augmented_test = D1xD1DDMNISTDataModule(256, augment_test=True)
        data_module_augmented_test.setup(stage='test')
        results = trainer.test(model, dataloaders=[data_module.test_dataloader(), data_module_augmented_test.test_dataloader()])
    else:
        results = trainer.test(model, data_module)

    return results


if __name__ == "__main__":
    import numpy as np

    args = get_args()
    seeds = args.seed  # list of seeds

    all_results = []
    for seed in seeds:
        args.seed = seed
        results = run(args)
        all_results.append(results)

        # Didn't work, so commenting it out:
        # # Reset state between runs so each seed behaves
        # # identically to a fresh process invocation
        # import gc
        # gc.collect()  # release Python refs to GPU tensors first
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.reset_peak_memory_stats()
        #     torch.backends.cudnn.benchmark = False

    # Aggregate and print mean/std across seeds
    if len(seeds) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS OVER {len(seeds)} SEEDS: {seeds}")
        print(f"{'='*60}")

        # results is a list (per seed) of lists (per dataloader) of dicts
        num_dataloaders = len(all_results[0])
        for dl_idx in range(num_dataloaders):
            if num_dataloaders > 1:
                print(f"\n--- Dataloader {dl_idx} ---")
            metrics = {}
            for seed_results in all_results:
                for key, val in seed_results[dl_idx].items():
                    metrics.setdefault(key, []).append(val)

            for key, vals in sorted(metrics.items()):
                vals = np.array(vals)
                print(f"  {key}: {vals.mean():.4f} +/- {vals.std():.4f}")
    else:
        print("\nSingle seed run, no aggregation needed.")