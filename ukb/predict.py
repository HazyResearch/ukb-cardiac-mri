"""

"""
from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('agg')

import glob
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

    model.eval()
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
    Load UKBB datasets

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


        postprocess_data = None
        if (postprocessing is not None):
            if (args.series == 3):
                postprocess_data = compose_postprocessing_multi(postprocessing)
            else:
                postprocess_data = compose_postprocessing(postprocessing)

        #test = DataSet("{}/{}".format(args.test, args.labelcsv), args.test,
        test = DataSet(args.labelcsv, args.test,
                       series=args.series, N=args.n_frames,
                       image_type=args.image_type,
                       preprocess=preprocess_data,
                       postprocess=postprocess_data,
                       seed=args.data_seed)

        return test, classes

    else:
        logger.error("Dataset name not recognized")


def get_best_model_weights_path(weights_path, model_name, dataset, seed):
    weights_path = "{}/{}_{}_{}".format(weights_path, dataset, model_name, str(seed))
    paths = glob.glob("{}/{}_BEST".format(weights_path, model_name))
    #paths = sorted(paths, key=lambda x: int(x.split('_')[-1]))
    print("{}/{}_BEST".format(weights_path, model_name))
    print("Weights loaded: {}".format(paths))
    return paths[0]


def main(args):

    ts = int(time.time())

    # ------------------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------------------
    test, classes = load_dataset(args)

    logger.info("[TEST]  {}".format(len(test)))
    logger.info("Classes: {}".format(" ".join(classes)))

    # ------------------------------------------------------------------------------
    # Load Model and Hyperparameter Grid
    # ------------------------------------------------------------------------------
    args.model_weights_path = get_best_model_weights_path(args.model_weights_path, args.model_name, args.dataset, args.seed)

    best_model = torch.load(args.model_weights_path)
    model = best_model['model']
    args.threshold = best_model['threshold']
    print("Threshold: {}".format(args.threshold))
    print("Best[DEV]score:")
    print("'fit'0.000sec")
    checkpoint_dir = "{}/{}_{}_{}".format(args.outdir, args.dataset, args.model_name, args.seed)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Score and Save Best Model
    # ------------------------------------------------------------------------------
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    results, preds_table, preds = score(model, test_loader, classes, threshold=args.threshold, 
                                        seed=args.seed, use_cuda=args.use_cuda,
                                        topSelection=args.top_selection)

    if args.outdir:
        if results is not None:
            cPickle.dump(results, open("{!s}/results_{!s}.pkl".format(checkpoint_dir, args.seed), "wb"))
        open("{}/predictions_{}.csv".format(checkpoint_dir, args.seed), "w").write(preds_table)


    # ------------------------------------------------------------------------------
    # Generate Plots and Reports
    # ------------------------------------------------------------------------------
    if args.outdir and args.report:
        tsne_plot(model, test_loader, "{}/{}.".format(checkpoint_dir, args.seed),
                  seed=args.seed, use_cuda=args.use_cuda, threshold=args.threshold, fmt="pdf",
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

    argparser.add_argument("--top_selection", type=int, default=None, help="the number of positive cases to select from the test set")
    argparser.add_argument("-d", "--dataset", type=str, default="UKBB", help="dataset name")
    argparser.add_argument("--threshold", type=float, default=0.5, help="threshold cutoff to use when evaluating test set")
    argparser.add_argument("--model_weights_path", type=str, help="the path to the saved model weights, e.g. `--model_weights_path test` when file is `test/UKBB_Dense4012FrameRNN_14/Dense4012FrameRNN_BEST`.")
    argparser.add_argument("--model_name", type=str, help="the name of the model")
    argparser.add_argument("-a", "--dconfig", type=str, default=None, help="load data config JSON")
    argparser.add_argument("-c", "--config", type=str, default=None, help="load model config JSON")
    argparser.add_argument("-L", "--labelcsv", type=str, default="labels.csv", help="dataset labels csv filename")

    argparser.add_argument("-o", "--outdir", type=str, default=None, help="save model to outdir")

    argparser.add_argument("--test", type=str, default=None, help="test set")
    argparser.add_argument("--data_seed", type=int, default=4321, help="random sample seed")

    argparser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    argparser.add_argument("-H", "--host_device", type=str, default="gpu", help="Host device (GPU|CPU)")
    argparser.add_argument("-I", "--image_type", type=str, default='grey', choices=['grey', 'rgb'], help="the image type, grey/rgb")
    argparser.add_argument("--use_cuda", action="store_true", help="whether to use GPU(CUDA)")
    argparser.add_argument("--cache_data", action="store_true", help="whether to cache data into memory")
    argparser.add_argument("--meta_data", action="store_true", help="whether to include meta data in model")
    argparser.add_argument("-F", "--n_frames", type=int, default=30, help="number of frames to select from a series")

    argparser.add_argument("--series", type=int, default=0, choices=[0, 1, 2, 3], help="which series to load for training")
    argparser.add_argument("--report", action="store_true", help="generate summary plots")
    argparser.add_argument("--seed", type=int, default=1234, help="random model seed")
    argparser.add_argument("--quiet", action="store_true", help="suppress logging")
    argparser.add_argument("--verbose", action="store_true", help="print debug information to log")
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
