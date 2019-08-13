import argparse
import pandas as pd
from os import makedirs
from os.path import isdir

from ensemble import *

def main(args):
    pd.set_option("display.width", 100)

    if args.pids_csv is not None:
        pids = list(pd.read_csv(args.pids_csv)[args.pids_key])
    else:
        pids = None

    if args.output_dir is None:
        output_dir = "{}/ensemble".format(args.results_dir)
    else:
        output_dir = args.output_dir
        
    if not isdir(output_dir):
        makedirs(output_dir)

    if args.output_name is None:
        output_name = "ensemble"
    else:
        output_name = args.output_name

    experiment = Ensemble.from_folder(args.results_dir, args.dev_dir, pids=pids)
    _ = experiment.median_vote(metric=args.metric)
    _ = experiment.mv_vote()

    if experiment.score:
        print(experiment.score_dataframe)
        experiment.score_dataframe.to_csv("{}/{}_score.csv".format(output_dir, output_name), index=False)

    experiment.proba_dataframe.to_csv("{}/{}_proba.csv".format(output_dir, output_name), index=False)
    experiment.pred_dataframe.to_csv("{}/{}_pred.csv".format(output_dir, output_name), index=False)
    print("Ensembled results are saved into {}.".format(output_dir))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--results_dir", type=str, required=True, help="the folder where the results are")
    argparser.add_argument("--dev_dir", type=str, required=True, help="the folder where the devset results are")
    argparser.add_argument("--metric", type=str, default="f1_score", help="the metric for tuning threshold")
    argparser.add_argument("--pids_csv", type=str, default=None, help="the csv of pids to filter the results")
    argparser.add_argument("--pids_key", type=str, default="ID", help="the label for pids in the csv: ID/PID/etc.")
    argparser.add_argument("--output_dir", type=str, default=None, help="folder to save the ensembled results")
    argparser.add_argument("--output_name", type=str, default=None, help="name used to save the ensembled results")

    args = argparser.parse_args()
    main(args)
