"""

"""
from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('agg')

import os
import torch
import argparse
import warnings
import pandas
from torch.utils.data import Dataset, DataLoader
from utils import *
from metrics import *
from dataloaders import *
from models import GridSearchTrainer
from sklearn.exceptions import UndefinedMetricWarning
from utils.viz import tsne_plot, analysis_plot
from transforms import *

try:
    # for python2
    import cPickle
except ImportError:
    # for python3
    import _pickle as cPickle

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# set reasonable pandas dataframe display defaults
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

torch.backends.cudnn.deterministic = True

def score(model, data_loader, classes, threshold=0.5, seed=1234, use_cuda=False, topSelection=None):
    """ Generate classfication report """
    np.random.seed(seed=int(seed))
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    y_proba, y_pred = model.predict(data_loader, threshold=threshold, binary=len(classes)==2, return_proba=True, topSelection=topSelection)
    print(y_proba)

    preds_table = "PID,Y_TRUE,Y_PROBA,Y_PRED\n"
    preds_table += "\n".join(["{},{},{},{}".format(data[0], data[1], y_proba[i], y_pred[i]) for i,data in enumerate(data_loader.dataset.get_labels())])

    try:
        y_true = np.hstack([y.numpy() for x,y in data_loader])
        results = classification_summary(y_true, y_pred, classes, y_proba)
        preds = {"y_true":y_true, "y_pred":y_pred, "y_proba": y_proba}
        return results, preds_table, preds
    except:
        preds = {"y_true":None, "y_pred":y_pred, "y_proba": y_proba}
        return None, preds_table, preds



def load_dataset(args):
    """
    Load UKBB or CIFAR10 datasets

    Image centering statistics

    /lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c/
        max:  192
        mean: 27.4613475359
        std:  15.8350095314

    /lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c_P/
        max:  4095
        mean: 2045.20689212
        std:  292.707986212

    /lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c_MAG/
        max:  336.0
        mean: 24.1274
        std:  14.8176


    :param args:
    :return:
    """
    if args.dataset == "UKBB":
        if args.cache_data:
            DataSet = UKBBCardiacMRICache
        elif args.meta_data:
            DataSet = UKBBCardiacMRIMeta
        else:
            DataSet = UKBBCardiacMRI
        classes = ("TAV", "BAV")

        # Get Preprocessing and Augmentation params
        preprocessing, augmentation, postprocessing = get_data_config(args)
        print_dict_pairs(preprocessing, title="Data Preprocessing Args")
        print_dict_pairs(augmentation, title="Data Augmentation Args")
        preprocessing["n_frames"] = args.n_frames

        # Preprocessing data should be computed on ALL datasets (train, val,
        #   and test). This includes:
        #       - Frame Selection
        #       - Rescale Intensity
        #       - Gamma Correction
        if (args.series == 3):
            preprocess_data = compose_preprocessing_multi(preprocessing)
        else:
            preprocess_data = compose_preprocessing(preprocessing)

        # HACK ignore augmentations (for now)
        #   data augmentations only to be used during training
        augment_train = None
        if (augmentation is not None):
            augment_train = compose_augmentation(augmentation, seed=args.data_seed)

        postprocess_data = None
        if (postprocessing is not None):
            if (args.series == 3):
                postprocess_data = compose_postprocessing_multi(postprocessing)
            else:
                postprocess_data = compose_postprocessing(postprocessing)

        train = DataSet("{}/{}".format(args.train, args.labelcsv), args.train,
                        series=args.series, N=args.n_frames,
                        image_type=args.image_type,
                        preprocess=preprocess_data,
                        augmentation=augment_train,
                        postprocess=postprocess_data,
                        rebalance=args.rebalance,
                        threshold=args.data_threshold,
                        seed=args.data_seed,
                        sample=args.sample,
                        sample_type=args.sample_type,
                        sample_split=args.sample_split,
                        n_samples=args.n_samples,
                        pos_samples=args.pos_samples,
                        neg_samples=args.neg_samples,
                        frame_label=args.use_frame_label,
                        rebalance_strategy=args.rebalance_strategy,
                        semi=args.semi, semi_dir=args.semi_dir, semi_csv=args.semi_csv)

        # randomly split dev into stratified dev/test sets
        if args.stratify_dev:
            df = stratified_sample_dataset("{}/labels.csv".format(args.dev), args.seed)
            dev = DataSet(df["dev"], args.dev,
                          series=args.series, N=args.n_frames,
                          image_type=args.image_type,
                          preprocess=preprocess_data,
                          postprocess=postprocess_data,
                          seed=args.data_seed)
            test = DataSet(df["test"], args.dev,
                           series=args.series, N = args.n_frames,
                           image_type=args.image_type,
                           preprocess=preprocess_data,
                           postprocess=postprocess_data,
                           seed=args.data_seed)

        # use manually defined dev/test sets
        else:
            dev = DataSet("{}/labels.csv".format(args.dev), args.dev,
                          series=args.series, N=args.n_frames,
                          image_type=args.image_type,
                          preprocess=preprocess_data,
                          postprocess=postprocess_data,
                          seed=args.data_seed)
            if args.test:
                test = DataSet("{}/labels.csv".format(args.test), args.test,
                               series=args.series, N=args.n_frames,
                               image_type=args.image_type,
                               preprocess=preprocess_data,
                               postprocess=postprocess_data,
                               seed=args.data_seed)
            else:
                test = None

        return train, dev, test, classes


    elif args.dataset == "CIFAR10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data_root = "data/CIFAR10/"
        if not os.path.exists(data_root):
            os.mkdir(data_root)
        num_samples = 500
        train = CIFAR10(data_root, split="train", num_samples=num_samples)
        dev   = CIFAR10(data_root, split="dev", num_samples=num_samples)
        test  = CIFAR10(data_root, split="test", num_samples=num_samples)

        return train, dev, test, classes

    else:
        logger.error("Dataset name not recognized")


def main(args):

    ts = int(time.time())

    # ------------------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------------------
    train, dev, test, classes = load_dataset(args)

    logger.info("[TRAIN] {}".format(len(train)))
    logger.info("[DEV]   {}".format(len(dev)))
    if args.test:
        logger.info("[TEST]  {}".format(len(test)))
    logger.info("Classes: {}".format(" ".join(classes)))

    # ------------------------------------------------------------------------------
    # Load Model and Hyperparameter Grid
    # ------------------------------------------------------------------------------
    #  - model_class:        target model class object
    #  - model_class_params: params required to initialize model
    #  - model_param_grid:   hyperparameter search space
    model_class, model_class_params, model_param_grid = get_model_config(args)
    model_class_params["seq_max_seq_len"] = args.n_frames
    model_class_params["pretrained"] = args.pretrained
    model_class_params["requires_grad"] = args.requires_grad

    # ------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------
    checkpoint_dir = "{}/{}_{}_{}".format(args.outdir, args.dataset, model_class.__name__, args.seed)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    trainer = GridSearchTrainer(model_class, model_class_params,
                                model_param_grid, args.n_model_search,
                                noise_aware=args.noise_aware,
                                use_cuda=args.use_cuda, seed=args.seed)

    num_frames = args.n_frames if model_class.__name__ == "VGG16Net" else None
    fit_time, model, _, tuned_threshold  = trainer.fit(train, dev, test, n_epochs=args.n_epochs, 
                                             update_freq=args.update_freq, checkpoint_freq=args.checkpoint_freq, 
                                             checkpoint_dir=checkpoint_dir, num_frames=num_frames, 
                                             tune_metric=args.tune_metric, metric=args.early_stopping_metric, 
                                             verbose=args.verbose)

    # ------------------------------------------------------------------------------
    # Score and Save Best Model
    # ------------------------------------------------------------------------------
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    results, preds_table, preds = score(model, test_loader, classes, threshold=tuned_threshold, 
                                        seed=args.seed, use_cuda=args.use_cuda,
                                        topSelection=args.top_selection)
    results.update({"time":fit_time})

    if args.outdir:
        trainer.save(model.state_dict(), checkpoint_dir, "best.{}".format(args.seed))
        cPickle.dump(results, open("{!s}/results_{!s}.pkl".format(checkpoint_dir, args.seed), "wb"))
        open("{}/predictions_{}.csv".format(checkpoint_dir, args.seed), "w").write(preds_table)

    # ------------------------------------------------------------------------------
    # Generate Plots and Reports
    # ------------------------------------------------------------------------------
    if args.outdir and args.report:
        tsne_plot(model, test_loader, "{}/{}.".format(checkpoint_dir, args.seed),
                  seed=args.seed, use_cuda=args.use_cuda, threshold=tuned_threshold, fmt="pdf",
                  topSelection=args.top_selection, save_coords=args.tsne_save_coords,
                  pred_only=args.tsne_pred_only, save_embeds=args.save_embeds,
                  classes=args.tsne_classes)
        plot_types=['plt_hist_plot', 'hist_plot', 'roc_curve', 'prc_curve']
        #analysis_plot(model=model, data_loader=test_loader, 
        #              outfpath="{}/{}.".format(checkpoint_dir, args.seed),
        #              types=plot_types, fmt="pdf",
        #              seed=args.seed, use_cuda=args.use_cuda)
        analysis_plot(y_true=preds["y_true"], y_proba=preds["y_proba"], 
                      outfpath="{}/{}.".format(checkpoint_dir, args.seed), 
                      types=plot_types, fmt="pdf")



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("-d", "--dataset", type=str, default="UKBB", help="dataset name")
    argparser.add_argument("-L", "--labelcsv", type=str, default="labels.csv", help="dataset labels csv filename")

    argparser.add_argument("--train", type=str, default=None, help="training set")
    argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
    argparser.add_argument("--test", type=str, default=None, help="test set")
    argparser.add_argument("--stratify_dev", action="store_true", help="split dev into stratified dev/test")

    argparser.add_argument("-c", "--config", type=str, default=None, help="load model config JSON")
    argparser.add_argument("-g", "--param_grid", type=str, default=None, help="load manual parameter grid from JSON")
    argparser.add_argument("-p", "--params", type=str, default=None, help="load `key=value,...` pairs from command line")
    argparser.add_argument("-o", "--outdir", type=str, default=None, help="save model to outdir")

    argparser.add_argument("-a", "--dconfig", type=str, default=None, help="load data config JSON")

    argparser.add_argument("-R", "--rebalance", action="store_true", help="rebalance training data")
    argparser.add_argument("--data_threshold", type=float, default=0.5, help="threshold cutoff to use when sampling patients")
    argparser.add_argument("--data_seed", type=int, default=4321, help="random sample seed")

    argparser.add_argument("--sample", action="store_true", help="sample training data")
    argparser.add_argument("--sample_type", type=int, default=0, choices=[0, 1, 2, 3],
        help="sample method to use [1: Random Sample, 1: Threshold Random Sample, 2: Top/Bottom Sample]")
    argparser.add_argument("--sample_split", type=float, default=0.5, help="ratio of 'positive' classes wanted")
    argparser.add_argument("--n_samples", type=int, default=100, help="number of patients to sample")
    argparser.add_argument("--pos_samples", type=int, default=0, help="number of positive patients to sample")
    argparser.add_argument("--neg_samples", type=int, default=0, help="number of negative patients to sample")
    argparser.add_argument("--rebalance_strategy", type=str, default="oversample", help="over/under sample")

    argparser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    argparser.add_argument("-N", "--n_model_search", type=int, default=1, help="number of models to search over")
    argparser.add_argument("-S", "--early_stopping_metric", type=str, default="roc_auc_score", help="the metric for checkpointing the model")
    argparser.add_argument("-T", "--tune_metric", type=str, default="roc_auc_score", help="the metric for "
                           "tuning the threshold. str-`roc_auc_score` for metric, float-`0.6` for fixed threshold")
    argparser.add_argument("-E", "--n_epochs", type=int, default=1, help="number of training epochs")
    argparser.add_argument("-M", "--n_procs", type=int, default=1, help="number processes (per model, CPU only)")
    argparser.add_argument("-W", "--n_workers", type=int, default=1, help="number of grid search workers")
    argparser.add_argument("-H", "--host_device", type=str, default="gpu", help="Host device (GPU|CPU)")
    argparser.add_argument("-U", "--update_freq", type=int, default=5, help="progress bar update frequency")
    argparser.add_argument("-C", "--checkpoint_freq", type=int, default=5, help="checkpoint frequency")
    argparser.add_argument("-I", "--image_type", type=str, default='grey', choices=['grey', 'rgb'], help="the image type, grey/rgb")
    argparser.add_argument("--use_cuda", action="store_true", help="whether to use GPU(CUDA)")
    argparser.add_argument("--cache_data", action="store_true", help="whether to cache data into memory")
    argparser.add_argument("--meta_data", action="store_true", help="whether to include meta data in model")
    argparser.add_argument("--semi", action="store_true", help="whether to use semi model")
    argparser.add_argument("--semi_dir", type=str, default='/lfs/1/heartmri/train32', help="path to train folder in semi model")
    argparser.add_argument("--semi_csv", type=str, default="labels.csv", help="semi dataset labels csv filename")
    argparser.add_argument("-F", "--n_frames", type=int, default=30, help="number of frames to select from a series")
    argparser.add_argument("--use_frame_label", action="store_true", help="whether to use frame level labels.")

    argparser.add_argument("--pretrained", action="store_true", help="whether to load pre_trained weights.")
    argparser.add_argument("--requires_grad", action="store_true", help="whether to fine tuning the pre_trained model.")
    argparser.add_argument("--noise_aware", action="store_true", help="whether to train on probability labels.")
    argparser.add_argument("--series", type=int, default=0, choices=[0, 1, 2, 3], help="which series to load for training")
    argparser.add_argument("--report", action="store_true", help="generate summary plots")
    argparser.add_argument("--seed", type=int, default=1234, help="random model seed")
    argparser.add_argument("--quiet", action="store_true", help="suppress logging")
    argparser.add_argument("--verbose", action="store_true", help="print debug information to log")
    argparser.add_argument("--top_selection", type=int, default=None, help="the number of positive cases to select from the test set")
    argparser.add_argument("--tsne_save_coords", action="store_true", help="whether to save coords of tsne.")
    argparser.add_argument("--tsne_pred_only", action="store_true", help="whether to plot preds only in tsne.")
    argparser.add_argument("--tsne_classes", default=None, type=int, action='append', help="the classes used to plot tsne plots. defaultto read from labels Y_TRUE.")
    argparser.add_argument("--save_embeds", action="store_true", help="whether to save the embedding of test set.")
    args = argparser.parse_args()

    if not args.quiet:
        logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)

    if not torch.cuda.is_available() and args.host_device.lower() == 'gpu':
        logger.error("Warning! CUDA not available, defaulting to CPU")
        args.host_device = "cpu"

    if torch.cuda.is_available():
        logger.info("CUDA PyTorch Backends")
        logger.info("torch.backends.cudnn.deterministic={}".format(torch.backends.cudnn.deterministic))

    # print summary of this run
    logger.info("python " + " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args")

    main(args)
