from torch.optim import Optimizer
import os 
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn import metrics
import hparams as hp
import math



def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=hp.adam_beta1, b2=hp.adam_beta2, e=hp.epsilon,
                 weight_decay_rate=hp.adam_weight_decay_rate,
                 max_grad_norm=1.0):
        assert lr > 0.0, "Learning rate: %f - should be > 0.0" % (lr)
        assert schedule in SCHEDULES, "Invalid schedule : %s" % (schedule)
        assert 0.0 <= warmup < 1.0 or warmup == -1.0, \
            "Warmup %f - should be in 0.0 ~ 1.0 or -1 (no warm up)" % (warmup)
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        """ get learning rate in training """
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                                       'please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

def optim4GPU(model, total_steps):
    """ optimizer for GPU training """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
    return BertAdam(optimizer_grouped_parameters,
                    lr=hp.lr,
                    warmup=hp.warmup,
                    t_total=total_steps)

class Paths():
    def __init__(self, output_path, result_path):
        self.output_path = output_path
        self.bert_path = f'{output_path}/model_bert'
        self.sim_path = f'{output_path}/model_sim'
        self.bert_checkpoints_path = f'{output_path}/bert_checkpoints_path'
        self.runs_path = f'{output_path}/runs'
        self.train_metric_fp = f'{output_path}/metric.train.{result_path}'
        self.test_metric_fp = f'{output_path}/metric.test.{result_path}'
        self.test_pred_fp = f'{output_path}/pred.test.{result_path}'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path, exist_ok=True)
        os.makedirs(self.sim_path, exist_ok=True)
        os.makedirs(self.bert_checkpoints_path, exist_ok=True)
        os.makedirs(self.runs_path, exist_ok=True)

    def set_result_path(self, result_path):
        output_path = self.output_path
        self.train_metric_fp = f'{output_path}/metric.train.{result_path}'
        self.test_metric_fp = f'{output_path}/metric.test.{result_path}'
        self.test_pred_fp = f'{output_path}/pred.test.{result_path}'

def compute_prediction_metric(pred, obsv, avg='binary'):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(obsv, pred,
                                                                       average=avg)
    acc = metrics.accuracy_score(obsv, pred)
    fpr, tpr, thresholds = metrics.roc_curve(obsv, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr':fpr,
        'tpr':tpr,
        'auc':auc
    }


def write_pred_results(fp, y_pred, y_true, lines, score):
    with open(fp, 'w') as f, open(fp+"_all", 'w') as f2:
        for p, t, l,s in zip(y_pred, y_true, lines,score):
            if not t == p:
                f.write('{}///{}///{}///{}\n'.format(l, p, t,s))
            f2.write('{}///{}///{}///{}\n'.format(l, p, t,s))
