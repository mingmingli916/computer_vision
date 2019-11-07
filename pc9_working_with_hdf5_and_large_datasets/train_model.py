from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py
from pyimagesearch.messages import info

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
                help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open the HDF5 dataset for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled prior to writing it to disk
db = h5py.File(args['db'], 'r')
# index before i is for training and after i for testing
i = int(db['labels'].shape[0] * .75)
features = db['features']
labels = db['labels']
label_names = db['label_names']

# define the set of parameter that we want to tune
# then start a grid search where we evaluate our model
# for each value of C
print(info.tuning_hyperparameters)
params = {'C': [.0001, .001, .01, .1, 1.]}
model = GridSearchCV(LogisticRegression(), param_grid=params, cv=3, n_jobs=args['jobs'])
model.fit(features[:i], labels[:i])
print('[INFO] best hyperparameters: {}'.format(model.best_params_))

# generate a classification report for the model
print(info.evaluating_model)
preds = model.predict(features[i:])
print(classification_report(labels[i:], preds, target_names=label_names))

# compute the raw accuracy with extra precision
acc = accuracy_score(labels[i:], preds)
print('[INFO] score: {}'.format(acc))

# serialize the model to disk
print(info.saving_model)
with open(args['model'], 'wb') as f:
    f.write(pickle.dumps(model))

db.close()

# [INFO] best hyperparameters: {'C': 0.01}
# [INFO] evaluating model...
#               precision    recall  f1-score   support
#
#          cat       0.99      0.99      0.99      3091
#          dog       0.99      0.99      0.99      3159
#
#     accuracy                           0.99      6250
#    macro avg       0.99      0.99      0.99      6250
# weighted avg       0.99      0.99      0.99      6250
#
# [INFO] score: 0.98992
# [INFO] saving model...
