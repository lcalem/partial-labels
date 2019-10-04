from model.relabel.base import ClassifRelabelator

from model import priors

from config.config import cfg


class PriorRelabeling(ClassifRelabelator):

    def __init__(self, exp_folder, p, nb_classes):
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes
        self.prior = self.load_prior(cfg.RELABEL.PRIOR)

    def load_prior(self, name):
        if name == 'conditional':
            prior_path = cfg.RELABEL.PRIOR_PATH
            prior_path = prior_path.replace('$PROP', str(self.p))
            return priors.ConditionalPrior(prior_path, nb_classes=self.nb_classes)

    def relabel(self, x_batch, y_batch, y_pred):
        '''
        y_pred: (1, batch_size, K) -> have to take y_pred[0]
        x_batch is used for image ids

        Steps for relabeling:
        - compute the relevant prior given the ground truth
        - combine the prior with the predictions (-> y_k)
        - Use this y_k and the original predictions y_pred to picj the relabeled examples
        '''
        # print('shape of y_pred %s' % str(y_pred.shape))
        p_k = self.prior.compute_pk(y_batch[0])

        y_k = self.prior.combine(y_pred, p_k)
        relabeling, nb_added = self.prior.pick_relabel(y_pred, y_k, y_batch[0])  # (BS, K)
        self.total_added += nb_added

        # print('y batch')
        # print(y_batch)

        # print('y pred')
        # print(y_pred)

        # print('pk')
        # print(p_k)

        # print('y_k')
        # print(y_k)

        # print('relabeling')
        # print(relabeling)

        # write batch to relabel csv
        for i in range(len(relabeling)):
            parts = relabeling[i]
            img_id = x_batch[1][i][0]

            # for last batch we have duplicates to fill the remaining batch size, we don't want to write those
            if img_id not in self.seen_keys:
                relabel_line = '%s,%s,%s\n' % (img_id, str(parts[0]), ','.join([str(elt) for elt in parts[1:]]))
                self.f_relabel.write(relabel_line)

            self.seen_keys.add(img_id)
