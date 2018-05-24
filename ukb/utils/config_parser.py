import sys
import json
import models
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

def convert_param_string(s):
    """
    Convert string of hyperparamters into typed dictionary
    e.g., `lr=0.001,rebalance=False,attention=True`

    This is used to parse paramaters specificed on the command line

    :param s:
    :return:
    """
    config = dict([p.split("=") for p in s.split(",")])

    # force typecasting in this order
    types = [int, float]
    for param in config:
        v = config[param]
        for t in types:
            try:
                v = t(v)
            except:
                continue
            config[param] = v
            break
        if config[param] in ['true','True']:
            config[param] = True
        elif config[param] in ['false','False']:
            config[param] = False

    return config


def get_model_config(args, verbose=True):
    """
    Given command line arguments and (optional) JSON configuration file,
    setup model and paramater grid

    :param model_name:
    :param manual_config:
    :param verbose:
    :return:
    """
    # load config JSON
    if args.config:
        args.config = json.load(open(args.config,"rU"))
        model_class = getattr(models, args.config[u"model"])
        model_class_params = args.config[u'model_class_params']
        #model_hyperparams  = args.config[u'model_hyperparams']
        model_param_grid   = args.config[u'model_param_grid']
        logger.info("Loaded model config from JSON file...")

        # convert list params into tuples
        for param_name in model_param_grid:
            values = []
            for v in model_param_grid[param_name]:
                values.append(v if type(v) is not list else tuple(v))
            model_param_grid[param_name] = values

    # use model defaults
    elif args.model:
        model_class, model_class_params = {},{}
        model_param_grid = {}
        logger.info("Loaded model defaults...")

    else:
        logger.error("Please specify model config or model class type")
        sys.exit()

    # override parameter grid
    if args.param_grid:
        manual_param_grid = json.load(open(args.param_grid, "rU"))
        args.n_model_search = len(manual_param_grid[u'params'])
        logger.info("Using manual parameter grid, setting n_model_search={}".format(args.n_model_search))
    else:
        manual_param_grid = {}

    # # custom model parameters
    # if args.params:
    #     params = convert_param_string(args.params)
    #     # override any grid search settings
    #     logger.info("Overriding some model hyperparameters")
    #     # override any model_hyperparameter defaults
    #     for name in params:
    #         model_hyperparams[name] = params[name]
    #         # also override in the param grid
    #         if name in model_param_grid:
    #             model_param_grid[name] = [params[name]]

    # override model params from command line
    model_class_params['seed']       = args.seed
    model_class_params['n_threads']  = args.n_procs
    #model_hyperparams['n_epochs']    = args.n_epochs
    model_class_params['host_device'] = args.host_device
    model_param_grid = OrderedDict(sorted(model_param_grid.items()))

    return model_class, model_class_params, model_param_grid #, manual_param_grid


def get_data_config(args, verbose=True):
    """
    Given command line arguments and (optional) JSON configuration file,
    setup data preprocessing and augmentation.

    :param data_config:
    :param verbose:
    :return:
    """
    # load config JSON
    if args.dconfig:
        args.dconfig = json.load(open(args.dconfig,"rU"))
        if (args.series == 3):
            preprocessing = args.dconfig.get(u'Preprocess',
                {"FrameSelector" : {
                    "name" : "FrameSelectionVarMulti"
                }
            })
            augmentation = None
            postprocessing = args.dconfig.get(u'Postprocess',
                {"Name" : "RescaleIntensityMulti"})
            logger.info("Loaded data config from JSON MULTI file...")
        else:
            preprocessing = args.dconfig.get(u'Preprocess',
                {"FrameSelector" : {
                    "name" : "FrameSelectionVar"
                }
            })
            augmentation = args.dconfig.get(u'Augmentation', None)
            postprocessing = args.dconfig.get(u'Postprocess',
                {"Name" : "RescaleIntensity"})
            logger.info("Loaded data config from JSON file...")
    else:
        if (args.series == 3):
            preprocessing = {
                "FrameSelector" : {
                    "name" : "FrameSelectionVarMulti"
                }
            }
            augmentation = None
            postprocessing = {"Name" : "RescaleIntensityMulti"}
            logger.info("Loaded data defaults for MULTI series...")
        else:
            preprocessing = {
                "FrameSelector" : {
                    "name" : "FrameSelectionVar"
                }
            }
            augmentation = None
            postprocessing = {"Name" : "RescaleIntensity"}
            logger.info("Loaded data defaults for SINGLE series...")

    return preprocessing, augmentation, postprocessing
