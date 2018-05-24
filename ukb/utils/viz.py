import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import seaborn as sns
sns.set_style("darkgrid")


def set_seed(seed, use_cuda):
    np.random.seed(seed=int(seed))
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def embed(model, data_loader, seed, use_cuda):
    """
    Generated learned representation
    :param model:
    :param data_loader:
    :return:
    """
    set_seed(seed,use_cuda)
    embs = []
    for i, data in enumerate(data_loader):
        x,y = data
        if isinstance(x, list):
            X = [Variable(x_) for x_ in x]
            hidden = model.init_hidden(X[0].size(0))
        else:
            X = Variable(x)
            hidden = model.init_hidden(X.size(0))
        embs.append(model.embedding(X, hidden).data.numpy())
    return np.vstack(embs)


def tsne_plot(model, data_loader, outfpath, seed, use_cuda, 
              threshold=0.5, fmt="pdf", width=16, height=8,
              topSelection=None, save_coords=False, pred_only=False,
              save_embeds=False, classes=None):
    """
    Generate TSNE (2D stochastic neighbor embedding) for visualization

    :param model:
    :param data_loader:
    :param width:
    :param height:
    :return:
    """
    # generate labels and embeddings for the provided data loader
    embs = embed(model, data_loader, seed, use_cuda)
    if save_embeds:
        np.save("{}embeds.npy".format(outfpath), embs)
    set_seed(seed, use_cuda)
    y_proba, y_pred = model.predict(data_loader, threshold=threshold, 
                                    return_proba=True, topSelection=topSelection)
    y_true = np.hstack([y.numpy() for x,y in data_loader])

    X_emb = TSNE(n_components=2).fit_transform(embs)
    classes = np.unique(y_true) if classes is None else np.unique(classes)
    colors = cm.rainbow(np.linspace(0, 1, len(classes)))

    def scatter_tsne(ax_, y_label, ax_title):
        if save_coords:
            # save the tsne coordinates
            pnts_indexes = [[index for index, label in zip(range(y_label.shape[0]), y_label) if label == class_name] for class_name in classes]
            pnts_indexes = [np.vstack(p) if p else np.array([]) for p in pnts_indexes]

        pnts = [[pnt for pnt, label in zip(X_emb, y_label) if label == class_name] for class_name in classes]
        pnts = [np.vstack(p) if p else np.array([]) for p in pnts]

        for p, c in zip(pnts, colors):
            if p.size == 0:
                continue
            xs, ys = zip(*p)
            ax_.scatter(xs, ys, color=c)
        ax_.set_title(ax_title)

        if save_coords:
            # save the tsne coordinates
            pnt_table = []
            for i,p in zip(pnts_indexes, pnts):
                if p.size == 0:
                    continue
                xs, ys = zip(*p)
                pnt_table += [(index_[0],x_,y_) for index_, x_, y_ in zip(i, xs, ys)]

            pnt_table = sorted(pnt_table, key=lambda x: x[0])
            coords = np.array([(x_,y_) for i_,x_,y_ in pnt_table])
            xs_,ys_ = zip(*coords)
            pids = [data[0] for i,data in enumerate(data_loader.dataset.get_labels())]

            tsne_csv = "PID,X,Y,PROBA,{}\n".format(ax_title.upper())
            tsne_csv += "\n".join(["{},{},{},{},{}".format(pid_,x_,y_,proba_,label_) for pid_,x_,y_,proba_,label_ in zip(pids, xs_, ys_, y_proba, y_label)])
            open("{}tsne_{}_coords.csv".format(outfpath, ax_title), "w").write(tsne_csv)

    if pred_only:
        width /= 2
        # setup 1 subplots
        f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(width, height))

        # 1) Predicted Labels
        scatter_tsne(ax1, y_pred, "y_pred")

        # Save the fig
        plt.savefig("{}tsne_pred.{}".format(outfpath, fmt))
    else:
        # setup 2 subplots
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(width, height))

        # 1) True Labels
        scatter_tsne(ax1, y_true, "y_true")
        # 2) Predicted Labels
        scatter_tsne(ax2, y_pred, "y_pred")

        # Save the fig
        plt.savefig("{}tsne.{}".format(outfpath, fmt))



def analysis_plot(**kwargs):
    """
    Generate Histograms of probabilities.
    
    Params
    ------
    outfpath:
    types:
    fmt:

    1). y_true:
    1). y_proba:

    2). model:
    2). data_loader:
    2). seed:
    2). use_cuda:

    Return
    ------
    None
    """
    outfpath    = kwargs.get("outfpath")
    types       = kwargs.get("types")
    fmt         = kwargs.get("fmt", "pdf")

    def _plt_hist_plot(y_true, y_proba, bins='auto'):
        plt.figure(figsize=(5,5))
        plt.hist(y_proba, bins=bins)

        plt.xlim([-0.01, 1.01])
        plt.title("Probabilities histogram")
        plt.xlabel("Probabilities")
        plt.ylabel("Sample count")

        plt.savefig("{}plt_hist.{}".format(outfpath, fmt))

    def _hist_plot(y_true, y_proba):
        plt.figure(figsize=(5,5))
        positive_index = np.flatnonzero(y_true)
        negative_index = np.flatnonzero(1-y_true)

        sns.distplot(y_proba[positive_index], label="positive_class")
        sns.distplot(y_proba[negative_index], label="negative_class")

        plt.xlim([-0.01, 1.01])
        plt.title("Probabilities histogram")
        plt.xlabel("Probabilities")
        plt.ylabel("Sample count")
        plt.legend(bbox_to_anchor=(0.95,0.05),loc=4, borderaxespad=0.)

        plt.savefig("{}hist.{}".format(outfpath, fmt))

    def _roc_curve(y_true, y_proba):
        plt.figure(figsize=(5,5))

        fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=1)
        roc_score = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=4)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.title("ROC_Curve: {:4.2f}".format(roc_score))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(bbox_to_anchor=(0.95,0.05),loc=4, borderaxespad=0.)

        plt.savefig("{}roc_curve.{}".format(outfpath, fmt))

    def _prc_curve(y_true, y_proba):
        plt.figure(figsize=(5,5))

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=1)
        prc_score = auc(recall, precision)

        plt.step(recall, precision, color="b", alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.title("PRC_Curve: {:4.2f}".format(prc_score))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(0.95,0.05),loc=4, borderaxespad=0.)

        plt.savefig("{}prc_curve.{}".format(outfpath, fmt))

    def generate_plots(y_true, y_proba, image_type):
        function_dict = dict(hist_plot=_hist_plot,
                             plt_hist_plot=_plt_hist_plot,
                             roc_curve=_roc_curve,
                             prc_curve=_prc_curve)
        function_dict[image_type](y_true, y_proba)

    

    if not 'y_true' in kwargs.keys() or not 'y_proba' in kwargs.keys():
        model       = kwargs.get("model")
        data_loader = kwargs.get("data_loader")
        seed        = kwargs.get("seed")
        use_cuda    = kwargs.get("use_cuda")

        set_seed(seed, use_cuda)
        y_proba = model.predict_proba(data_loader)
        y_true  = np.hstack([y.numpy() for x,y in data_loader])
    else:
        y_true      = kwargs.get("y_true")
        y_proba     = kwargs.get("y_proba")


    for image_type in types:
        try:
            generate_plots(y_true, y_proba, image_type)
        except:
            pass







