import argparse
import os
import tensorflow as tf
import numpy as np
from model1 import Model
from data_utils1 import download_dbpedia, build_word_dict, build_dataset, batch_iter


def train(train_x, train_xnli1, train_xnli2, train_xsts1, train_xsts2, train_lm_y, train_clf_y, train_clf_sts, train_clf_nli, test_x, test_xnli1, test_xnli2, test_xsts1, test_xsts2, test_lm_y, test_clf_y, test_clf_nli, test_clf_sts, word_dict, args):
    with tf.Session() as sess:
        model = Model(len(word_dict), num_class=14, args=args)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.total_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        lm_loss_summary = tf.summary.scalar("lm_loss", model.lm_loss)
        clf_loss_summary = tf.summary.scalar("clf_loss", model.clf_loss)
        clf_nliloss_summary = tf.summary.scalar("clf_nliloss", model.clf_loss_nli)
        clf_stsloss_summary = tf.summary.scalar("clf_stsloss", model.clf_loss_sts)
        total_loss_summary = tf.summary.scalar("total_loss", model.total_loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("summary", sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x, batch_xnli1, batch_xnli2, batch_xsts1, batch_xsts2, batch_lm_y, batch_clf_y, batch_clf_sts, batch_clf_nli):
            feed_dict = {model.x: batch_x, model.xnli11: batch_xnli1, model.xnli2:batch_xnli2, model.clf_nli:batch_clf_nli, model.clf_sts:batch_clf_sts, model.lm_y: batch_lm_y, model.clf_y: batch_clf_y, model.keep_prob: args.keep_prob}
            _, step, summaries, total_loss, lm_loss, clf_loss,clf_loss_sts, clf_loss_nli = \
                sess.run([train_op, global_step, summary_op, model.total_loss, model.lm_loss, model.clf_loss,model.clf_loss_sts,model.clf_loss_nli], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("step {0}: loss={1} (lm_loss={2}, clf_loss={3})".format(step, total_loss, lm_loss, clf_loss))

        def eval(test_x,test_xnli1, test_xnli2, test_xsts1, test_xsts2, test_lm_y, test_clf_nli, test_clf_sts, test_clf_y):
            test_batches = batch_iter(test_x, test_lm_y, test_clf_y, args.batch_size, 1)
            losses, accuracies, iters = 0, 0, 0

            for batch_x, batch_lm_y, batch_clf_y in test_batches:
                feed_dict = {model.x: batch_x, model.xnli11: batch_xnli1, model.xnli2:batch_xnli2, model.clf_nli:batch_clf_nli, model.clf_sts:batch_clf_sts, model.lm_y: batch_lm_y, model.clf_y: batch_clf_y, model.keep_prob: args.keep_prob}
                lm_loss, accuracy = sess.run([model.lm_loss, model.accuracy], feed_dict=feed_dict)
                losses += lm_loss
                accuracies += accuracy
                iters += 1

            print("\ntest perplexity = {0}".format(np.exp(losses / iters)))
            print("test accuracy = {0}\n".format(accuracies / iters))

        batches = batch_iter(train_x, train_xnli1, train_xnli2, train_xsts1, train_xsts2, train_lm_y, train_clf_y, train_clf_sts, train_clf_nli, args.batch_size, args.num_epochs)
        for (batch_x, batch_xnli1, batch_xnli2, batch_xsts1, batch_xsts2, batch_lm_y, batch_clf_y, batch_clf_sts, batch_clf_nli in batches:
            train_step(batch_x, batch_xnli1, batch_xnli2, batch_xsts1, batch_xsts2, batch_lm_y, batch_clf_y, batch_clf_sts, batch_clf_nli)
            step = tf.train.global_step(sess, global_step)

            if step % 1000 == 0:
                eval(test_x, test_lm_y, test_clf_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=150, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=150, help="RNN network size.")

    parser.add_argument("--keep_prob", type=float, default=0.8, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs.")
    parser.add_argument("--max_document_len", type=int, default=100, help="max document length.")
    args = parser.parse_args()

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()
    print("Preprocessing dataset..")
    train_x, train_xnli1, train_xnli2, train_xsts1, train_xsts2, train_lm_y, train_clf_y , train_clf_sts, train_clf_nli = build_dataset("train", word_dict, args.max_document_len)
    test_x, test_xnli1, test_xnli2, test_xsts1, test_xsts2, test_lm_y, test_clf_y , test_clf_sts, test_clf_nli  = build_dataset("test", word_dict, args.max_document_len)

    train(train_x, train_xnli1, train_xnli2, train_xsts1, train_xsts2, train_lm_y, train_clf_y, train_clf_nli, train_clf_sts, test_x, test_xnli1, test_xnli2, test_xsts1, test_xsts2, test_lm_y, test_clf_y,test_clf_nli, test_clf_sts ,word_dict, args)
