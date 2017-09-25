import numpy as np
import tensorflow as tf

from rnn_base import RNNBase
from snorkel.models import Candidate
from utils import SymbolTable, get_bi_rnn_output, get_bi_rnn_seq_output


SD = 0.1


class CRFTextRNN(RNNBase):
    """RNN for sequence labeling of strings of text."""

    def _preprocess_data(self, candidates, marginals=None, extend=False):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train),
            or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()

        data, ends, sent_buf = [], [], []
        for candidate in candidates:
            tok = candidate.get_contexts()[1].text
            index = candidate.get_contexts()[2].text

            if sent_buf and index == '0':
                f = self.word_dict.get if extend else self.word_dict.lookup
                data.append(np.array(map(f, sent_buf)))
                ends.append(len(sent_buf))
                sent_buf = []

            sent_buf.append(tok)

        marg = []
        if marginals is not None:
            cand_idx = 0
            for sent_len in ends:
                end_idx = cand_idx + sent_len
                marg.append(marginals[cand_idx:end_idx, :])
                cand_idx = end_idx
            marg = np.array(marg)

        return data, ends, marg

    def _build_model(self, dim=50, attn_window=None, max_len=20,
                     cell_type=tf.contrib.rnn.BasicLSTMCell,
                     word_dict=SymbolTable(), **kwargs):

        # Set the word dictionary passed in as the word_dict for the instance
        self.max_len = max_len
        self.word_dict = word_dict
        vocab_size = word_dict.len()

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
            rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs,
                sequence_length=self.sentence_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=False
            )
        # potentials = get_bi_rnn_output(rnn_out, dim, self.sentence_lengths)
        potentials, ntime_steps = get_bi_rnn_seq_output(
            rnn_out, dim, self.sentence_lengths)

        # Add dropout layer
        self.keep_prob = tf.placeholder(tf.float32)
        potentials_dropout = tf.nn.dropout(potentials, self.keep_prob, seed=s3)

        # Build activation layer
        # self.Y = tf.placeholder(tf.float32, [None, self.cardinality])
        self.Y = tf.placeholder(tf.float32, [None, None, self.cardinality])
        W = tf.Variable(tf.random_normal((2 * dim, self.cardinality),
                                         stddev=SD, seed=s4))
        b = tf.Variable(np.zeros(self.cardinality), dtype=tf.float32)
        self.logits = tf.matmul(potentials, W) + b
        self.logits = tf.reshape(
            self.logits, [-1, ntime_steps, self.cardinality])
        # self.marginals_op = tf.nn.softmax(self.logits)

        # self.pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def _build_training_ops(self, **training_kwargs):

        batch_size = tf.shape(self.logits)[0]
        seq_len = tf.shape(self.logits)[1]
        self.Y = tf.cast(tf.argmax(self.Y, axis=2), tf.int32)
        self.Y = tf.reshape(self.Y, [batch_size, seq_len])

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.Y, self.sentence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)

        # self.pred, viterbi_score = tf.contrib.crf.viterbi_decode(
        #     self.logits, self.transition_params)

        # losses = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.logits, labels=self.Y)
        # self.loss = tf.reduce_mean(losses)

        # Build training op
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_feed_dict(self, X_b, Y_b, lr=0.01, dropout=None, **kwargs):
        X_b, len_b, Y_b = self._make_tensor(X_b, Y_b)
        # print()
        # print(X_b)
        # print()
        # print(len_b)
        # print()
        # print(Y_b)
        return {
            self.sentences:        X_b,
            self.sentence_lengths: len_b,
            self.Y:                Y_b,
            self.keep_prob:        dropout or 1.0,
            self.lr:               lr
        }

    def _make_tensor(self, x, y=None):
        """Construct input tensor with padding
            Builds a matrix of symbols corresponding to @self.word_dict for the
            current batch and an array of true sentence lengths
        """
        batch_size = len(x)
        x_batch = np.zeros((batch_size, self.max_len), dtype=np.int32)
        y_batch = np.zeros((batch_size, self.max_len, self.cardinality))
        len_batch = np.zeros(batch_size, dtype=np.int32)

        if y is not None:
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

        return x_batch, len_batch, y_batch

    def predictions(self, X, b=0.5, batch_size=None):

        if isinstance(X[0], Candidate):
            X_test, ends, _ = self._preprocess_data(X, extend=False)
            self._check_max_sentence_length(ends)
        else:
            X_test = X

        # Make tensor and run prediction op
        x, x_len, _ = self._make_tensor(X_test)
        # pred = self.session.run(self.pred, {
        #     self.sentences:        x,
        #     self.sentence_lengths: x_len,
        #     self.keep_prob:        1.0,
        # })

        logit_scores = self.session.run(self.logits, {
            self.sentences: x,
            self.sentence_lengths: x_len,
            self.keep_prob: 1.0
        })

        preds = []
        for logits in logit_scores:
            pred_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logits,
                                                                    self.transition_params)
            preds.append(pred_seq)

        # return pred
        return preds

    def score(self, X_test, Y_test, b=0.5, set_unlabeled_as_neg=True, beta=1,
              batch_size=None):

        # predictions, viterbi_score = self.predictions(X_test, b, batch_size)
        # pred_words = [self.word_dict.reverse()[i] for i in predictions]
        # try:
        #     Y_test = np.array(Y_test.todense()).reshape(-1)
        # except:
        #     Y_test = np.array(Y_test)

        # correct = np.where([predictions == Y_test])[0].shape[0]
        # return correct / float(Y_test.shape[0])

        X_test, ends, _ = self._preprocess_data(X_test, extend=False)
        self._check_max_sentence_length(ends)
        predictions = self.predictions(X_test, b=b, batch_size=batch_size)

        # # Convert Y_test to dense numpy array
        # try:
        #     Y_test = np.array(Y_test.todense()).reshape(-1)
        # except:
        #     Y_test = np.array(Y_test)

        labels = []
        cand_idx = 0
        for sent_len in ends:
            end_idx = cand_idx + sent_len
            # print('{}/{}'.format(cand_idx, end_idx))
            labels.append(Y_test[cand_idx:end_idx])
            cand_idx = end_idx
        Y_test = np.array(labels)

        # print(X_test)
        # print(Y_test)
        # print(predictions)

        # correct = np.where([predictions == Y_test])[0].shape[0]
        # return correct / float(Y_test.shape[0])

        token_err, sent_err = 0, 0
        token_num, sent_num = 0, len(Y_test)
        for sent_pred, sent_gold in zip(predictions, Y_test):
            pred_err = 0

            for tag_pred, tag_gold in zip(sent_pred, sent_gold):
                token_num += 1
                if tag_pred != tag_gold:
                    pred_err += 1

                if tag_pred > self.cardinality:
                    print('PREDICTION ({}) / CARDINALITY MISMATCH ({})'
                          .format(tag_pred, self.cardinality))

            token_err += pred_err
            if pred_err != 0:
                sent_err += 1

        return float(token_err) / token_num, float(sent_err) / sent_num

    def train(self, X_train, Y_train, X_dev=None, max_sentence_length=None,
              **kwargs):
        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """
        # Text preprocessing
        X_train, ends, Y_train = self._preprocess_data(
            X_train, Y_train, extend=True)
        if X_dev is not None:
            X_dev, _ = self._preprocess_data(X_dev, [], extend=False)

        # Get max sentence size
        max_len = max_sentence_length or max(len(x) for x in X_train)
        self._check_max_sentence_length(ends, max_len=max_len)

        # Train model- note we pass word_dict through here so it gets saved...
        super(RNNBase, self).train(X_train, Y_train, X_dev=X_dev,
                                   word_dict=self.word_dict, max_len=max_len, **kwargs)
