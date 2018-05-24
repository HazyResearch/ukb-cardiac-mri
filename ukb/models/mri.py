import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .frame import LeNetFrameEncoder, FNNFrameEncoder, DenseNet121, vgg16_bn, densenet121, densenet_40_12_bc
from .sequence import RNN, MetaRNN, SeqSumPoolingEncoder

logger = logging.getLogger(__name__)

################################################################################
# Image Container Models (each image is independantly classified)
################################################################################


class MRINet(nn.Module):
    """
    Simple container class for MRI net. This module consists of:
        1) A frame encoder, e.g., a ConvNet/CNN
        2) Linear output layer

    """
    def __init__(self, frame_encoder, n_classes, output_size, layers, dropout, 
                 vote_opt='mean', use_cuda=False):
        super(MRINet, self).__init__()

        self.n_classes  = n_classes
        self.fenc       = frame_encoder
        self.classifier = self._make_classifier(output_size, n_classes, layers, dropout)
        self.vote_opt   = vote_opt
        self.use_cuda   = use_cuda

    def _make_classifier(self, output_size, n_classes, layers=[64,32], dropout=0.2):
        layers = [output_size] + layers + [n_classes]
        classifier = []
        for i, size in enumerate(layers[:-1]):
            classifier.append(nn.Linear(layers[i], layers[i+1]))
            if size != layers[-1]:
                classifier.append(nn.ReLU(True))
                classifier.append(nn.Dropout(p=dropout))

        return nn.Sequential(*classifier)


    def init_hidden(self, batch_size):
        return None

    def embedding(self, x, hidden=None):
        """Get learned representation of MRI sequence"""
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        batch_size, num_frames, num_channels, width, height = x.size()
        self.num_frames = num_frames
        x = x.view(-1, num_channels, width, height)
        x = self.fenc(x)
        x = self.classifier(x)
        if self.use_cuda:
            return x.cpu()
        else:
            return x

    def forward(self, x, hidden=None):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        # collapse all frames into new batch = batch_size * num_frames
        batch_size, num_frames, num_channels, width, height = x.size()
        self.num_frames = num_frames
        x = x.view(-1, num_channels, width, height)
        # encode frames
        x = self.fenc(x)
        # feed-forward-classifier
        x = self.classifier(x)
        return x

    def vote(self, y_pred, threshold=None):
        if threshold is not None:
            y_pred = (y_pred > threshold).astype(float)
        num_frames  = self.num_frames
        num_samples = int(y_pred.shape[0]/num_frames)
        ex_shape = y_pred.shape[1:]
        y_pred = np.reshape(y_pred, (num_samples, num_frames,)+ex_shape)
        y_pred = np.mean(y_pred, axis=1)
        return y_pred

    def predict_proba(self, data_loader, binary=True, pos_label=1, threshold=0.5):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = Variable(x) if not self.use_cuda else Variable(x).cuda()
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x.size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)

        if self.vote_opt=='mean':
            y_pred = self.vote(y_pred)
        elif self.vote_opt=='vote':
            y_pred = self.vote(y_pred, threshold)
        return y_pred[:, pos_label] if binary else y_pred

    def predict(self, data_loader, binary=True, pos_label=1, threshold=0.5, return_proba=False):
        """
        If binary classification, use threshold on positive class
        If multinomial, just select the max probability as the predicted class
        :param data_loader:
        :param binary:
        :param pos_label:
        :param threshold:
        :return:
        """
        proba = self.predict_proba(data_loader, binary, pos_label, threshold)
        if binary:
            pred = np.array([1 if p > threshold else 0 for p in proba])
        else:
            pred = np.argmax(proba, 1)

        if return_proba:
            return (proba, pred)
        else:
            return pred


class DenseNet121Net(MRINet):
    def __init__(self, n_classes, output_size, use_cuda, **kwargs):
        super(DenseNet121Net, self).__init__(frame_encoder=None, n_classes=n_classes,
                                             output_size=output_size, use_cuda=use_cuda)
        self.name = "DenseNet121Net"
        self.fenc = DenseNet121()


class VGG16Net(MRINet):
    def __init__(self, n_classes, use_cuda, **kwargs):

        input_shape         = kwargs.get("input_shape", (3, 32, 32))
        layers              = kwargs.get("layers", [64, 32])
        dropout             = kwargs.get("dropout", 0.2)
        vote_opt            = kwargs.get("vote_opt", "mean")
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)
        frm_output_size     = self.get_frm_output_size(input_shape)

        super(VGG16Net, self).__init__(frame_encoder=None, n_classes=n_classes,
                                       output_size=frm_output_size,
                                       layers=layers, dropout=dropout,
                                       vote_opt=vote_opt, use_cuda=use_cuda)

        self.name = "VGG16Net"
        self.fenc = vgg16_bn(pretrained=pretrained, requires_grad=requires_grad)

    def get_frm_output_size(self, input_shape):
        feature_output  = int(min(input_shape[-1], input_shape[-2])/32)
        feature_output  = 1 if feature_output == 0 else feature_output
        frm_output_size = pow(feature_output, 2) * 512
        return frm_output_size


class LeNet(MRINet):
    def __init__(self, n_classes, n_channels, output_size, use_cuda, **kwargs):
        super(LeNet, self).__init__(frame_encoder=None, n_classes=n_classes,
                                    output_size=output_size, use_cuda=use_cuda)
        self.name = "LeNet"
        self.fenc = LeNetFrameEncoder(n_channels=n_channels, output_size=output_size)


################################################################################
# Sequence Container Models
################################################################################

class MRISequenceNet(nn.Module):
    """
    Simple container network for MRI sequence classification. This module consists of:

        1) A frame encoder, e.g., a ConvNet/CNN
        2) A sequence encoder for merging frame representations, e.g., an RNN

    """
    def __init__(self, frame_encoder, seq_encoder, use_cuda=False):
        super(MRISequenceNet, self).__init__()
        self.fenc     = frame_encoder
        self.senc     = seq_encoder
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        return self.senc.init_hidden(batch_size)

    def embedding(self, x, hidden):
        """Get learned representation of MRI sequence"""
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        x = self.senc.embedding(x, hidden)
        if self.use_cuda:
            return x.cpu()
        else:
            return x

    def forward(self, x, hidden=None):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        # collapse all frames into new batch = batch_size * num_frames
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        # encode frames
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        # encode sequence
        x = self.senc(x, hidden)
        return x

    def predict_proba(self, data_loader, binary=True, pos_label=1):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = Variable(x) if not self.use_cuda else Variable(x).cuda()
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x.size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)
        return y_pred[:, pos_label] if binary else y_pred

    def predict(self, data_loader, binary=True, pos_label=1, threshold=0.5, return_proba=False, topSelection=None):
        """
        If binary classification, use threshold on positive class
        If multinomial, just select the max probability as the predicted class
        :param data_loader:
        :param binary:
        :param pos_label:
        :param threshold:
        :return:
        """
        proba = self.predict_proba(data_loader, binary, pos_label)
        if topSelection is not None and topSelection < proba.shape[0]:
            threshold = proba[np.argsort(proba)[-topSelection-1]]
        if binary:
            pred = np.array([1 if p > threshold else 0 for p in proba])
        else:
            pred = np.argmax(proba, 1)

        if return_proba:
            return (proba, pred)
        else:
            return pred



################################################################################
# FNN Models
################################################################################

class FNNFrameSum(MRISequenceNet):

    def __init__(self, n_classes, use_cuda, **kwargs):
        super(FNNFrameSum, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "FNNFrameSum"
        self.n_classes  = n_classes
        frm_layers      = kwargs.get("frm_layers", [64, 32])
        input_shape     = kwargs.get("input_shape", (1, 32, 32))
        frm_input_size  = input_shape[0]*input_shape[1]*input_shape[2]

        self.fenc = FNNFrameEncoder(input_size=frm_input_size, layers=list(frm_layers))
        self.senc = SeqSumPoolingEncoder(n_classes=n_classes, input_size=frm_layers[-1])

class FNNFrameRNN(MRISequenceNet):

    def __init__(self, n_classes, use_cuda, **kwargs):
        super(FNNFrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "FNNFrameRNN"
        self.n_classes      = n_classes
        frm_layers          = kwargs.get("frm_layers", [64, 32])
        input_shape         = kwargs.get("input_shape", (1, 32, 32))
        frm_input_size      = input_shape[0]*input_shape[1]*input_shape[2]
        frm_output_size     = frm_layers[-1]

        seq_output_size     = kwargs.get("seq_output_size", 128)
        seq_dropout         = kwargs.get("seq_dropout", 0.1)
        seq_attention       = kwargs.get("seq_attention", True)
        seq_bidirectional   = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len     = kwargs.get("seq_max_seq_len", 30)
        seq_rnn_type        = kwargs.get("rnn_type", "LSTM")

        self.fenc = FNNFrameEncoder(input_size=frm_input_size, layers=frm_layers)
        self.senc = RNN(n_classes=2, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)


################################################################################
# LeNet Models
################################################################################

class LeNetFrameSum(MRISequenceNet):

    def __init__(self, n_classes, use_cuda, **kwargs):
        super(LeNetFrameSum, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "LeNetFrameSum"
        self.n_classes  = n_classes
        frm_output_size = kwargs.get("frm_output_size", 84)
        input_shape     = kwargs.get("input_shape", (1, 32, 32))

        self.fenc = LeNetFrameEncoder(input_shape=input_shape, output_size=frm_output_size)
        self.senc = SeqSumPoolingEncoder(n_classes=n_classes, input_size=frm_output_size)


class LeNetFrameRNN(MRISequenceNet):

    def __init__(self, n_classes, use_cuda, **kwargs):
        super(LeNetFrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "LeNetFrameRNN"
        self.n_classes  = n_classes
        frm_output_size     = kwargs.get("frm_output_size", 84)
        input_shape         = kwargs.get("input_shape", (1, 32, 32))

        seq_output_size     = kwargs.get("seq_output_size", 128)
        seq_dropout         = kwargs.get("seq_dropout", 0.1)
        seq_attention       = kwargs.get("seq_attention", True)
        seq_bidirectional   = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len     = kwargs.get("seq_max_seq_len", 15)
        seq_rnn_type        = kwargs.get("rnn_type", "LSTM")

        self.fenc = LeNetFrameEncoder(input_shape=input_shape, output_size=frm_output_size)
        self.senc = RNN(n_classes=2, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)


################################################################################
# DenseNet 3-channel Models
################################################################################

class DenseNet121FrameSum(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(DenseNet121FrameSum, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name           = "DenseNet121FrameSum"
        self.n_classes      = n_classes
        input_shape         = kwargs.get("input_shape", (3, 32, 32))
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)
        frm_output_size     = pow(int(input_shape[-1]/32), 2) * 1024

        #self.fenc = DenseNet121()
        self.fenc = densenet121(pretrained=pretrained, requires_grad=requires_grad)
        self.senc = SeqSumPoolingEncoder(n_classes=n_classes, input_size=frm_output_size)


class DenseNet121FrameRNN(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(DenseNet121FrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name           = "DenseNet121FrameRNN"
        self.n_classes      = n_classes
        input_shape         = kwargs.get("input_shape", (3, 32, 32))
        frm_output_size     = pow(int(input_shape[-1]/32), 2) * 1024

        seq_output_size     = kwargs.get("seq_output_size", 128)
        seq_dropout         = kwargs.get("seq_dropout", 0.1)
        seq_attention       = kwargs.get("seq_attention", True)
        seq_bidirectional   = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len     = kwargs.get("seq_max_seq_len", 15)
        seq_rnn_type        = kwargs.get("rnn_type", "LSTM")
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)

        #self.fenc = DenseNet121()
        self.fenc = densenet121(pretrained=pretrained, requires_grad=requires_grad)
        self.senc = RNN(n_classes=2, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)



################################################################################
# VGG 3-channel Models
################################################################################

class VGG16FrameSum(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(VGG16FrameSum, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "VGG16FrameSum"
        self.n_classes  = n_classes
        input_shape     = kwargs.get("input_shape", (3, 32, 32))
        pretrained      = kwargs.get("pretrained", True)
        requires_grad   = kwargs.get("requires_grad", False)

        self.fenc       = vgg16_bn(pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)

        self.senc = SeqSumPoolingEncoder(n_classes=n_classes, input_size=frm_output_size)

    def get_frm_output_size(self, input_shape):
        feature_output  = int(min(input_shape[-1], input_shape[-2])/32)
        feature_output  = 1 if feature_output == 0 else feature_output
        frm_output_size = pow(feature_output, 2) * 512
        return frm_output_size


class VGG16FrameRNN(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(VGG16FrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "VGG16FrameRNN"
        self.n_classes  = n_classes
        input_shape     = kwargs.get("input_shape", (3, 32, 32))
        pretrained      = kwargs.get("pretrained", True)
        requires_grad   = kwargs.get("requires_grad", False)

        self.fenc       = vgg16_bn(pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)

        #print(kwargs)
        #print("seq_bidirectional" in kwargs)

        seq_output_size   = kwargs.get("seq_output_size", 128)
        seq_dropout       = kwargs.get("seq_dropout", 0.1)
        seq_attention     = kwargs.get("seq_attention", True)
        seq_bidirectional = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len   = kwargs.get("seq_max_seq_len", 15)
        seq_rnn_type      = kwargs.get("seq_rnn_type", "LSTM")
        self.senc = RNN(n_classes=n_classes, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0,1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size =  self.fenc.forward(x).size()[1]
        return frm_output_size

class Dense4012FrameRNN(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(Dense4012FrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "Dense4012FrameRNN"
        input_shape         = kwargs.get("input_shape", (3, 32, 32))
  
        seq_output_size     = kwargs.get("seq_output_size", 128)
        seq_dropout         = kwargs.get("seq_dropout", 0.1)
        seq_attention       = kwargs.get("seq_attention", True)
        seq_bidirectional   = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len     = kwargs.get("seq_max_seq_len", 15)
        seq_rnn_type        = kwargs.get("rnn_type", "LSTM")
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)

        logger.info("============================")
        logger.info("Dense4012FrameRNN parameters")
        logger.info("============================")
        logger.info("seq_output_size:   {}".format(seq_output_size))
        logger.info("seq_dropout:       {}".format(seq_dropout))
        logger.info("seq_attention:     {}".format(seq_attention))
        logger.info("seq_bidirectional: {}".format(seq_bidirectional))
        logger.info("seq_max_seq_len:   {}".format(seq_max_seq_len))
        logger.info("seq_rnn_type:      {}".format(seq_rnn_type))
        logger.info("pretrained:        {}".format(pretrained))
        logger.info("requires_grad:     {}\n".format(requires_grad))

        self.fenc           = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size     = self.get_frm_output_size(input_shape)

        self.senc = RNN(n_classes=2, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0,1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size =  self.fenc.forward(x).size()[1]
        return frm_output_size


################################################################################
# Sequence Container Meta Models
################################################################################

class MRIMetaSequenceRNN(MRISequenceNet):
    def __init__(self, frame_encoder, n_classes, use_cuda, **kwargs):
        super(MRIMetaSequenceRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.n_classes      = n_classes
        input_shape         = kwargs.get("input_shape", (3, 32, 32))

        self.fenc           = frame_encoder
        frm_output_size     = self.get_frm_output_size(input_shape)

        #print(kwargs)
        #print("seq_bidirectional" in kwargs)

        seq_output_size     = kwargs.get("seq_output_size", 128)
        seq_dropout         = kwargs.get("seq_dropout", 0.1)
        seq_attention       = kwargs.get("seq_attention", True)
        seq_bidirectional   = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len     = kwargs.get("seq_max_seq_len", 15)
        seq_rnn_type        = kwargs.get("seq_rnn_type", "LSTM")
        self.senc           = MetaRNN(n_classes=n_classes, input_size=frm_output_size, hidden_size=seq_output_size,
                                      dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                                      rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

        meta_input_shape    = kwargs.get("meta_input_shape", 3)
        self.classifier     = self.get_classifier(seq_output_size, n_classes, seq_bidirectional, meta_input_shape)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0,1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size =  self.fenc.forward(x).size()[1]
        return frm_output_size

    def get_classifier(self, seq_output_size, n_classes, seq_bidirectional,
                       meta_input_shape):
        b = 2 if seq_bidirectional else 1
        meta_input_shape = np.prod([meta_input_shape])
        classifier = nn.Linear(int(b * seq_output_size + meta_input_shape), int(n_classes))
        return classifier

    def embedding(self, x, hidden):
        """Get learned representation of MRI sequence"""
        x, meta = x
        return super(MRIMetaSequenceRNN, self).embedding(x, hidden)

    def forward(self, x, hidden=None):
        x, meta = x
        if self.use_cuda and not meta.is_cuda:
            meta = meta.cuda()
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()

        x = super(MRIMetaSequenceRNN, self).forward(x, hidden)
        concats = torch.cat((x.view(x.size(0), -1).float(),
                             meta.view(meta.size(0), -1).float()), 1)
        outputs = self.classifier(concats)

        return outputs

    def predict_proba(self, data_loader, binary=True, pos_label=1):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = [Variable(x_) if not self.use_cuda else Variable(x_).cuda() for x_ in x]
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x[0].size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)
        return y_pred[:, pos_label] if binary else y_pred


class MetaVGG16FrameRNN(MRIMetaSequenceRNN):
    def __init__(self, n_classes, use_cuda, **kwargs):
        self.name           = "MetaVGG16FrameRNN"
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)
        frame_encoder       = vgg16_bn(pretrained=pretrained, requires_grad=requires_grad)

        super(MetaVGG16FrameRNN, self).__init__(frame_encoder=frame_encoder, 
                                                n_classes=n_classes, 
                                                use_cuda=use_cuda, 
                                                **kwargs)


class MetaDense4012FrameRNN(MRIMetaSequenceRNN):
    def __init__(self, n_classes, use_cuda, **kwargs):
        self.name           = "MetaDense4012FrameRNN"
        pretrained          = kwargs.get("pretrained", True)
        requires_grad       = kwargs.get("requires_grad", False)
        frame_encoder       = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)

        super(MetaDense4012FrameRNN, self).__init__(frame_encoder=frame_encoder,
                                                    n_classes=n_classes,
                                                    use_cuda=use_cuda,
                                                    **kwargs)
