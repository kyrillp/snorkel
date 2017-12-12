import copy
import numpy as np
import tensorflow as tf

from time import time
from rnn_base import RNNBase
from snorkel.models import Candidate
from utils import SymbolTable, get_bi_rnn_output, get_bi_rnn_seq_output


SD = 0.1


class CRFTextRNN(RNNBase):
    """RNN for sequence labeling of strings of text."""

    def _preprocess_data(self, candidates, marginals=None, dev_labels=None, extend=False, shuffle_data=True):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train),
            or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()

        if not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()

        max_word_len = 0
        data, ends, sent_buf, words, word_buf = [], [], [], [], []
        for candidate in candidates:
            tok = candidate.get_contexts()[1].text
            index = candidate.get_contexts()[2].text

            if sent_buf and index == '0':
                f = self.word_dict.get if extend else self.word_dict.lookup
                data.append(np.array(map(f, sent_buf)))
                ends.append(len(sent_buf))
                sent_buf = []

                c = self.char_dict.get if extend else self.char_dict.lookup
                sent_words = [np.array(map(c, chars)) for chars in word_buf]
                words.append(np.array(sent_words))
                word_buf = []

            sent_buf.append(tok)
            word_buf.append(list(tok))
            max_word_len = max(max_word_len, len(tok))

        marg = []
        if marginals is not None:
            cand_idx = 0
            for sent_len in ends:
                end_idx = cand_idx + sent_len
                marg.append(marginals[cand_idx:end_idx, :])
                cand_idx = end_idx
            marg = np.array(marg)

        aligned_dev_labels = []
        if dev_labels is not None:
            cand_idx = 0
            for sent_len in ends:
                end_idx = cand_idx + sent_len
                aligned_dev_labels.append(dev_labels[cand_idx:end_idx])
                cand_idx = end_idx
            aligned_dev_labels = np.array(aligned_dev_labels)

        if shuffle_data:
            indexes = np.arange(len(data))
            np.random.shuffle(indexes)
            data = np.array(data)[indexes]
            ends = np.array(ends)[indexes]
            if marginals is not None:
                marg = marg[indexes]
            if dev_labels is not None:
                aligned_dev_labels = aligned_dev_labels[indexes]
            if words:
                words = np.array(words)[indexes]
            print('Shuffled data for LSTM')

        words = words if len(words) > 0 else None
        return data, ends, marg, aligned_dev_labels, words, max_word_len

    def _build_model(self, dim=50, dim_char=50, attn_window=None, max_len=20,
                     cell_type=tf.contrib.rnn.BasicLSTMCell, max_word_len=10,
                     word_dict=SymbolTable(), char_dict=SymbolTable(), **kwargs):

        # Set the word dictionary passed in as the word_dict for the instance
        self.max_len = max_len
        self.word_dict = word_dict
        vocab_size = word_dict.len()

        self.max_word_len = max_word_len
        self.char_dict = char_dict
        n_chars = char_dict.len()

        # Define input layers
        self.sentences = tf.placeholder(tf.int32, [None, None])
        self.sentence_lengths = tf.placeholder(tf.int32, [None])

        # Seeds
        s = self.seed
        s1, s2, s3, s4 = [None] * 4 if s is None else [s + i for i in range(4)]

        # Embedding layer
        emb_var = tf.Variable(
            tf.random_normal((vocab_size - 1, dim), stddev=SD, seed=s1))
        embedding = tf.concat([tf.zeros([1, dim]), emb_var], axis=0)
        inputs = tf.nn.embedding_lookup(embedding, self.sentences)

        # Character embedding
        # shape = (batch_size, max_sent_len, max_word_len)
        self.words = tf.placeholder(tf.int32, [None, None, None])
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None])

        char_var = tf.get_variable(name='char_embeddings',
                                   dtype=tf.float32, shape=[n_chars, dim_char])
        char_embedding = tf.nn.embedding_lookup(char_var, self.words)

        char_s = tf.shape(char_embedding)
        # shape = (batch x sentence, word, dim of char embeddings)
        char_embedding = tf.reshape(char_embedding, shape=[
                                    char_s[0] * char_s[1], char_s[-2], dim_char])
        word_lengths = tf.reshape(self.word_lengths, shape=[char_s[0] * char_s[1]])

        init = tf.contrib.layers.xavier_initializer(seed=s2)
        with tf.variable_scope(self.name + '_char', reuse=False, initializer=init):
            char_fw_cell = cell_type(dim_char, state_is_tuple=True)
            char_bw_cell = cell_type(dim_char, state_is_tuple=True)

            _, ((_, char_fw_out), (_, char_bw_out)) = tf.nn.bidirectional_dynamic_rnn(
                char_fw_cell, char_bw_cell, char_embedding,
                sequence_length=word_lengths,
                dtype=tf.float32
            )
        char_out = tf.concat([char_fw_out, char_bw_out], axis=-1)
        char_rep = tf.reshape(char_out, shape=[-1, char_s[1], 2 * dim_char])
        inputs = tf.concat([inputs, char_rep], axis=-1)

        # Add dropout layer
        self.keep_prob = tf.placeholder(tf.float32)
        inputs_dropout = tf.nn.dropout(inputs, self.keep_prob, seed=s3)

        # Build RNN graph
        batch_size = tf.shape(self.sentences)[0]
        init = tf.contrib.layers.xavier_initializer(seed=s2)
        with tf.variable_scope(self.name, reuse=False, initializer=init):
            # Build RNN cells
            fw_cell = cell_type(dim)
            bw_cell = cell_type(dim)
            # Add attention if needed
            if attn_window:
                fw_cell = tf.contrib.rnn.AttentionCellWrapper(
                    fw_cell, attn_window, state_is_tuple=True
                )
                bw_cell = tf.contrib.rnn.AttentionCellWrapper(
                    bw_cell, attn_window, state_is_tuple=True
                )
            # Construct RNN
            initial_state_fw = fw_cell.zero_state(batch_size, tf.float32)
            initial_state_bw = bw_cell.zero_state(batch_size, tf.float32)
            # rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(
            #     fw_cell, bw_cell, inputs,
            #     sequence_length=self.sentence_lengths,
            #     initial_state_fw=initial_state_fw,
            #     initial_state_bw=initial_state_bw,
            #     time_major=False
            # )
            rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs_dropout,
                sequence_length=self.sentence_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=False
            )
        # potentials = get_bi_rnn_output(rnn_out, dim, self.sentence_lengths)
        potentials, ntime_steps = get_bi_rnn_seq_output(
            rnn_out, dim, self.sentence_lengths)

        # Add dropout layer
        # self.keep_prob = tf.placeholder(tf.float32)
        # potentials_dropout = tf.nn.dropout(potentials, self.keep_prob, seed=s3)

        # Build activation layer
        # self.Y = tf.placeholder(tf.float32, [None, self.cardinality])
        self.Y = tf.placeholder(tf.float32, [None, None, self.cardinality])
        # self.train_labels = tf.placeholder(tf.int32, [None, self.cardinality])
        self.train_labels = tf.placeholder(tf.int32, [None, self.max_len])

        W = tf.Variable(tf.random_normal((2 * dim, self.cardinality),
                                         stddev=SD, seed=s4))
        b = tf.Variable(np.zeros(self.cardinality), dtype=tf.float32)
        self.logits = tf.matmul(potentials, W) + b
        # self.logits = tf.matmul(potentials_dropout, W) + b
        self.logits = tf.reshape(
            self.logits, [-1, ntime_steps, self.cardinality])
        # self.marginals_op = tf.nn.softmax(self.logits)

        self.pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def _build_training_ops(self, **training_kwargs):

        # batch_size = tf.shape(self.logits)[0]
        # seq_len = tf.shape(self.logits)[1]
        # self.Y = tf.cast(tf.argmax(self.Y, axis=2), tf.int32)
        # self.Y = tf.reshape(self.Y, [batch_size, seq_len])
        #
        # log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        #     self.logits, self.Y, self.sentence_lengths)
        # self.loss = tf.reduce_mean(-log_likelihood)

        # self.pred, viterbi_score = tf.contrib.crf.viterbi_decode(
        #     self.logits, self.transition_params)

        # losses = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.logits, labels=self.Y)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.train_labels)

        mask = tf.sequence_mask(self.sentence_lengths)
        losses = tf.boolean_mask(losses, mask)

        self.loss = tf.reduce_mean(losses)

        # Build training op
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, dropout=None, train_labels=None,
                             chars=None, **kwargs):
        X_b, len_b, Y_b, L_b, C_b, len_c = self._make_tensor(X_b, Y_b, train_labels, chars)

        return {
            self.sentences:        X_b,
            self.sentence_lengths: len_b,
            self.Y:                Y_b,
            self.keep_prob:        dropout or 1.0,
            self.lr:               lr,
            self.train_labels:     L_b,
            self.words:            C_b,
            self.word_lengths:     len_c
        }

    def _make_tensor(self, x, y=None, z=None, c=None):
        """Construct input tensor with padding
            Builds a matrix of symbols corresponding to @self.word_dict for the
            current batch and an array of true sentence lengths
        """
        batch_size = len(x)
        x_batch = np.zeros((batch_size, self.max_len), dtype=np.int32)
        y_batch = np.zeros((batch_size, self.max_len, self.cardinality))
        z_batch = np.zeros((batch_size, self.max_len), dtype=np.int32)
        c_batch = np.zeros((batch_size, self.max_len, self.max_word_len), dtype=np.int32)
        len_batch = np.zeros(batch_size, dtype=np.int32)
        len_words = np.zeros((batch_size, self.max_len), dtype=np.int32)

        if c is not None and y is None and z is None:
            for j, (token_ids, words) in enumerate(zip(x, c)):
                t = min(len(token_ids), self.max_len)
                x_batch[j, 0:t] = token_ids[0:t]
                len_batch[j] = t

                for x, y in enumerate(words[0:t]):
                    c_batch[j][x][0:len(y)] = y

                char_t = np.array([min(len(word_ids), self.max_word_len) for word_ids in words])
                len_words[j][0:len(char_t)] = char_t

        elif c is not None:
            for j, (token_ids, marginals, labels, words) in enumerate(zip(x, y, z, c)):
                t = min(len(token_ids), self.max_len)
                x_batch[j, 0:t] = token_ids[0:t]
                y_batch[j, 0:t] = marginals[0:t]
                z_batch[j, 0:t] = labels[0:t]
                len_batch[j] = t

                for x, y in enumerate(words[0:t]):
                    c_batch[j][x][0:len(y)] = y

                char_t = np.array([min(len(word_ids), self.max_word_len) for word_ids in words])
                len_words[j][0:len(char_t)] = char_t

        elif z is not None:
            for j, (token_ids, marginals, labels) in enumerate(zip(x, y, z)):
                t = min(len(token_ids), self.max_len)
                x_batch[j, 0:t] = token_ids[0:t]
                y_batch[j, 0:t] = marginals[0:t]
                z_batch[j, 0:t] = labels[0:t]
                len_batch[j] = t

        elif y is not None:
            for j, (token_ids, marginals) in enumerate(zip(x, y)):
                t = min(len(token_ids), self.max_len)
                x_batch[j, 0:t] = token_ids[0:t]
                y_batch[j, 0:t] = marginals[0:t]
                len_batch[j] = t

        else:
            for j, token_ids in enumerate(x):
                t = min(len(token_ids), self.max_len)
                x_batch[j, 0:t] = token_ids[0:t]
                len_batch[j] = t

        return x_batch, len_batch, y_batch, z_batch, c_batch, len_words

    def predictions(self, X, b=0.5, batch_size=None, words=None):

        if isinstance(X[0], Candidate):
            X_test, ends, _, _, words, _ = self._preprocess_data(X, extend=False)
            self._check_max_sentence_length(ends)
        else:
            X_test = X

        # Make tensor and run prediction op
        x, x_len, _, _, _words, _words_len = self._make_tensor(X_test, c=words)
        pred = self.session.run(self.pred, {
            self.sentences:        x,
            self.sentence_lengths: x_len,
            self.keep_prob:        1.0,
            self.words:            _words,
            self.word_lengths:     _words_len
        })

        # logit_scores = self.session.run(self.logits, {
        #     self.sentences: x,
        #     self.sentence_lengths: x_len,
        #     self.keep_prob: 1.0
        # })
        #
        # preds = []
        # for logits in logit_scores:
        #     pred_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logits,
        #                                                             self.transition_params)
        #     preds.append(pred_seq)

        return pred
        # return preds

    def score(self, X_test, Y_test, b=0.5, set_unlabeled_as_neg=True, beta=1,
              batch_size=None, other_id=-1):

        # predictions, viterbi_score = self.predictions(X_test, b, batch_size)
        # pred_words = [self.word_dict.reverse()[i] for i in predictions]
        # try:
        #     Y_test = np.array(Y_test.todense()).reshape(-1)
        # except:
        #     Y_test = np.array(Y_test)

        # correct = np.where([predictions == Y_test])[0].shape[0]
        # return correct / float(Y_test.shape[0])

        X_test, ends, _, _, words, _ = self._preprocess_data(X_test, extend=False)
        self._check_max_sentence_length(ends)
        predictions = self.predictions(X_test, b=b, batch_size=batch_size, words=words)

        # # Convert Y_test to dense numpy array
        # try:
        #     Y_test = np.array(Y_test.todense()).reshape(-1)
        # except:
        #     Y_test = np.array(Y_test)

        labels = []
        cand_idx = 0
        for sent_len in ends:
            end_idx = cand_idx + sent_len
            labels.append(Y_test[cand_idx:end_idx])
            cand_idx = end_idx
        Y_test = np.array(labels)

        correct = 0
        # correct = np.where([predictions == Y_test])[0].shape[0]
        # return correct / float(Y_test.shape[0])

        token_err, sent_err = 0, 0
        token_num, sent_num = 0, len(Y_test)
        gold_other_num, gold_other_err = 0, 0
        pred_other_num, pred_other_err = 0, 0

        for sent_pred, sent_gold in zip(predictions, Y_test):
            pred_err = 0

            for tag_pred, tag_gold in zip(sent_pred, sent_gold):
                token_num += 1

                if tag_pred == tag_gold:
                    correct += 1

                if tag_pred != tag_gold:
                    pred_err += 1

                    if tag_pred == other_id:
                        pred_other_err += 1
                    if tag_gold == other_id:
                        gold_other_err += 1

                if tag_pred == other_id:
                    pred_other_num += 1
                if tag_gold == other_id:
                    gold_other_num += 1

                if tag_pred > self.cardinality:
                    print('PREDICTION ({}) / CARDINALITY MISMATCH ({})'
                          .format(tag_pred, self.cardinality))

            token_err += pred_err
            if pred_err != 0:
                sent_err += 1

        if gold_other_num == 0:
            gold_other_num = 1
        if pred_other_num == 0:
            pred_other_num = 1

        return float(correct) / token_num, \
            float(token_err) / token_num, float(sent_err) / sent_num, \
            float(gold_other_err) / gold_other_num, float(pred_other_err) / pred_other_num

    def train(self, X_train, Y_train, dev_labels=None, X_dev=None, max_sentence_length=None,
              shuffle=True, max_word_length=None, **kwargs):
        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """
        # Text preprocessing
        X_train, ends, Y_train, train_labels, train_words, max_word_len = self._preprocess_data(
            X_train, Y_train, dev_labels=dev_labels, extend=True, shuffle_data=shuffle)
        if X_dev is not None:
            X_dev, _, _, _, _, _ = self._preprocess_data(X_dev, [], extend=False)

        # Get max sentence size
        max_len = max_sentence_length or max(len(x) for x in X_train)
        self._check_max_sentence_length(ends, max_len=max_len)
        max_word_len = max_word_length or max_word_len

        # Train model- note we pass word_dict through here so it gets saved...
        # super(RNNBase, self).train(X_train, Y_train, X_dev=X_dev,
        #                            word_dict=self.word_dict, max_len=max_len, train_labels=train_labels, **kwargs)
        self._train(X_train, Y_train, X_dev=X_dev, words=train_words, char_dict=self.char_dict,
                    word_dict=self.word_dict, max_len=max_len, dev_labels=train_labels,
                    max_word_len=max_word_len, **kwargs)

    def _train(self, X_train, Y_train, dev_labels=None, words=None, n_epochs=25, lr=0.01, batch_size=256,
               rebalance=False, X_dev=None, Y_dev=None, print_freq=5, dev_ckpt=True,
               dev_ckpt_delay=0.75, save_dir='checkpoints', **kwargs):
        """
        Generic training procedure for TF model

        :param X_train: The training Candidates. If self.representation is True, then
            this is a list of Candidate objects; else is a csr_AnnotationMatrix
            with rows corresponding to training candidates and columns
            corresponding to features.
        :param Y_train: Array of marginal probabilities for each Candidate
        :param n_epochs: Number of training epochs
        :param lr: Learning rate
        :param batch_size: Batch size for SGD
        :param rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        :param X_dev: Candidates for evaluation, same format as X_train
        :param Y_dev: Labels for evaluation, same format as Y_train
        :param print_freq: number of epochs at which to print status, and if present,
            evaluate the dev set (X_dev, Y_dev).
        :param dev_ckpt: If True, save a checkpoint whenever highest score
            on (X_dev, Y_dev) reached. Note: currently only evaluates at
            every @print_freq epochs.
        :param dev_ckpt_delay: Start dev checkpointing after this portion
            of n_epochs.
        :param save_dir: Save dir path for checkpointing.
        :param kwargs: All hyperparameters that change how the graph is built
            must be passed through here to be saved and reloaded to save /
            reload model. *NOTE: If a parameter needed to build the
            network and/or is needed at test time is not included here, the
            model will not be able to be reloaded!*
        """
        self._check_input(X_train)
        verbose = print_freq > 0

        # Set random seed for all numpy operations
        self.rand_state.seed(self.seed)

        # If the data passed in is a feature matrix (representation=False),
        # set the dimensionality here; else assume this is done by sub-class
        if not self.representation:
            kwargs['d'] = X_train.shape[1]

        if dev_labels is not None:
            if len(dev_labels) > 0:
                train_labels = copy.deepcopy(dev_labels)
            else:
                train_labels = None
        else:
            train_labels = None

        # Create new graph, build network, and start session
        self._build_new_graph_session(**kwargs)

        # Build training ops
        # Note that training_kwargs and model_kwargs are mixed together; ideally
        # would be separated but no negative effect
        with self.graph.as_default():
            self._build_training_ops(**kwargs)

        # Initialize variables
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        # Run mini-batch SGD
        n = len(X_train) if self.representation else X_train.shape[0]
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] n_train={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))
        dev_score_opt = 0.0
        for t in range(n_epochs):
            epoch_losses = []
            for i in range(0, n, batch_size):
                if train_labels is not None:
                    batch_labels = train_labels[i:min(n, i + batch_size)]
                else:
                    batch_labels = None

                if words is not None:
                    batch_words = words[i:min(n, i + batch_size)]
                else:
                    batch_words = None

                feed_dict = self._construct_feed_dict(
                    X_train[i:min(n, i + batch_size)],
                    Y_train[i:min(n, i + batch_size)],
                    train_labels=batch_labels,
                    chars=batch_words,
                    lr=lr,
                    **kwargs
                )
                # Run training step and evaluate loss function
                epoch_loss, _ = self.session.run(
                    [self.loss, self.optimizer], feed_dict=feed_dict)
                epoch_losses.append(epoch_loss)

            # Reshuffle training data
            train_idxs = range(n)
            self.rand_state.shuffle(train_idxs)
            X_train = [X_train[j] for j in train_idxs] if self.representation \
                else X_train[train_idxs, :]
            Y_train = Y_train[train_idxs]

            if train_labels is not None:
                train_labels = [train_labels[j] for j in train_idxs]

            # Print training stats and optionally checkpoint model
            if verbose and (t % print_freq == 0 or t in [0, (n_epochs - 1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name, t, time() - st, np.mean(epoch_losses))
                if X_dev is not None:
                    scores = self.score(X_dev, Y_dev, batch_size=batch_size)
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += '\tDev {0}={1:.2f}'.format(
                        score_label, 100. * score)
                print(msg)

                # If best score on dev set so far and dev checkpointing is
                # active, save checkpoint
                if X_dev is not None and dev_ckpt and \
                        t > dev_ckpt_delay * n_epochs and score > dev_score_opt:
                    dev_score_opt = score
                    self.save(save_dir=save_dir, global_step=t)

        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(
                self.name, time() - st))

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir)
