import mxnet as mx
import logging


def one_off_callback(train_iter, test_iter, one_off, context):
    def _callback(iter_num, symbol, arg, aux):
        # construct a model for the symbol so can make predictions on our data
        model = mx.mod.Module(symbol=symbol, context=context)
        model.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
        model.set_params(arg_params=arg, aux_params=aux)

        # compute one-off metric for both the training and testing data
        train_mae = _compute_one_off(model, train_iter, one_off)
        test_mae = _compute_one_off(model, test_iter, one_off)

        # log the values
        logging.info('Epoch[{}] Train-one-off={:.5f}'.format(iter_num, train_mae))
        logging.info('Epoch[{}] Test-one-off={:.5f}'.format(iter_num, test_mae))

    return _callback


def _compute_one_off(model, data_iter, one_off):
    total = 0
    correct = 0

    for preds, _, batch in model.iter_predict(data_iter):
        # convert the batch of predictions and labels to NumPy arrays
        predictions = preds[0].asnumpy().argmax(axis=1)  # largest probability
        labels = batch.label[0].asnumpy().astype('int')

        for pred, label in zip(predictions, labels):
            # if correct label is in the set of 'one off' predictions
            # then update the correct counter
            if label in one_off[pred]:
                correct += 1

            # increment the total number of samples
            total += 1

        return correct / float(total)
