import os
import random
import time

import inference
import tensorflow as tf
import model_helper
import misc_utils as utils
import nmt_utils
import vocab_utils

__all__ = ["run_sample_decode", "run_internal_eval", "run_external_eval", "run_full_eval"]

def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)


def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer):
  """Compute internal evaluation (perplexity) for both dev / test."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_eval_iterator_feed_dict = {
      eval_model.src_file_placeholder: dev_src_file,
      eval_model.tgt_file_placeholder: dev_tgt_file
  }

  dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                           eval_model.iterator, dev_eval_iterator_feed_dict,
                           summary_writer, "dev")
  test_ppl = None
  if hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: test_src_file,
        eval_model.tgt_file_placeholder: test_tgt_file
    }
    test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                              eval_model.iterator, test_eval_iterator_feed_dict,
                              summary_writer, "test")
  return dev_ppl, test_ppl

def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True):

  """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_infer_iterator_feed_dict = {
      infer_model.src_placeholder: inference.load_data(dev_src_file),
      infer_model.batch_size_placeholder: hparams.infer_batch_size,
  }
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev)

  test_scores = None
  if hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(test_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        "test",
        summary_writer,
        save_on_best=False)
  return dev_scores, test_scores, global_step

def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data):
  """Wrapper for running sample_decode, internal_eval and external_eval."""
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data)
  dev_ppl, test_ppl = run_internal_eval(
      eval_model, eval_sess, model_dir, hparams, summary_writer)
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model, infer_sess, model_dir, hparams, summary_writer)

  eval_results = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
  if hparams.test_prefix:
    eval_results += ", " + _format_results("test", test_ppl, test_scores,
                                           hparams.metrics)

  return eval_results, global_step

def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      result_str += ", %s %s %.1f" % (name, metric, scores[metric])
  return result_str


def _get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(src_data) - 1)
  utils.print_out("  # %d" % decode_id)

  iterator_feed_dict = {
      iterator_src_placeholder: [src_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  nmt_outputs, attention_summary = model.decode(sess)

  if hparams.beam_width > 0:
    # get the top translation.
    nmt_outputs = nmt_outputs[0]

  translation = nmt_utils.get_translation(
      nmt_outputs,
      sent_id=0,
      tgt_eos=hparams.eos,
      bpe_delimiter=hparams.bpe_delimiter)
  #utils.print_out("    src: %s" % src_data[decode_id])
  #utils.print_out("    ref: %s" % tgt_data[decode_id])
  #utils.print_out("    nmt: %s" % translation)
  print("    src: %s" % src_data[decode_id])
  print("    ref: %s" % tgt_data[decode_id])
  print("    nmt: %s" % translation.decode())

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)


def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0
  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  output = os.path.join(out_dir, "output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      output,
      ref_file=tgt_file,
      metrics=hparams.metrics,
      bpe_delimiter=hparams.bpe_delimiter,
      beam_width=hparams.beam_width,
      tgt_eos=hparams.eos,
      decode=decode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, "best_" + metric):
        setattr(hparams, "best_" + metric, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, "best_" + metric + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores