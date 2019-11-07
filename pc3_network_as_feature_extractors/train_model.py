from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
params = {'C': [.1, 1., 10., 100., 1000., 10000.]}
model = GridSearchCV(LogisticRegression(), param_grid=params, cv=3, n_jobs=args['jobs'])
model.fit(features[:i], labels[:i])
print('[INFO] best hyperparameters: {}'.format(model.best_params_))

print(info.evaluating_model)
preds = model.predict(features[i:])
print(classification_report(labels[i:], preds, target_names=label_names))

# serialize the model to disk
print(info.saving_model)
with open(args['model'], 'wb') as f:
    f.write(pickle.dumps(model))

db.close()

# flowers17
#               precision    recall  f1-score   support
#
#     bluebell       0.78      1.00      0.88        14
#    buttercup       0.94      0.94      0.94        18
#   colts_foot       0.93      0.81      0.87        16
#      cowslip       0.56      0.88      0.68        17
#       crocus       1.00      0.94      0.97        18
#     daffodil       0.88      0.79      0.83        19
#        daisy       1.00      1.00      1.00        28
#    dandelion       1.00      0.88      0.93        24
#   fritillary       0.96      0.96      0.96        28
#         iris       1.00      0.86      0.93        22
#  lily_valley       0.90      0.90      0.90        21
#        pansy       1.00      0.95      0.97        19
#     snowdrop       0.89      0.85      0.87        20
#    sunflower       1.00      1.00      1.00        21
#    tigerlily       1.00      1.00      1.00        18
#        tulip       0.95      0.86      0.90        21
#   windflower       0.94      1.00      0.97        16
#
#     accuracy                           0.92       340
#    macro avg       0.93      0.92      0.92       340
# weighted avg       0.93      0.92      0.92       340


# animals
#               precision    recall  f1-score   support
#
#         cats       0.98      1.00      0.99       245
#         dogs       0.99      0.98      0.99       253
#        panda       1.00      0.99      1.00       252
#
#     accuracy                           0.99       750
#    macro avg       0.99      0.99      0.99       750
# weighted avg       0.99      0.99      0.99       750

# caltech101
#                    precision    recall  f1-score   support
#
# BACKGROUND_Google       0.75      0.70      0.72       128
#             Faces       0.95      0.99      0.97       102
#        Faces_easy       0.98      0.96      0.97       107
#          Leopards       1.00      1.00      1.00        46
#        Motorbikes       1.00      1.00      1.00       178
#         accordion       0.92      1.00      0.96        11
#         airplanes       0.99      1.00      1.00       226
#            anchor       0.88      1.00      0.93         7
#               ant       0.88      0.88      0.88         8
#            barrel       1.00      1.00      1.00        13
#              bass       0.90      0.90      0.90        10
#            beaver       0.67      0.50      0.57         8
#         binocular       1.00      0.80      0.89         5
#            bonsai       0.91      1.00      0.96        32
#             brain       0.85      1.00      0.92        17
#      brontosaurus       1.00      0.75      0.86        12
#            buddha       0.88      1.00      0.94        22
#         butterfly       0.90      0.90      0.90        20
#            camera       1.00      0.94      0.97        16
#            cannon       1.00      0.86      0.92        14
#          car_side       1.00      1.00      1.00        25
#       ceiling_fan       0.93      1.00      0.97        14
#         cellphone       1.00      1.00      1.00        15
#             chair       1.00      0.94      0.97        16
#        chandelier       1.00      0.92      0.96        24
#       cougar_body       0.92      0.73      0.81        15
#       cougar_face       0.81      0.89      0.85        19
#              crab       0.62      0.77      0.69        13
#          crayfish       0.82      0.90      0.86        20
#         crocodile       0.92      0.85      0.88        13
#    crocodile_head       0.82      0.82      0.82        11
#               cup       1.00      1.00      1.00         9
#         dalmatian       1.00      1.00      1.00        18
#       dollar_bill       0.90      0.82      0.86        11
#           dolphin       0.89      1.00      0.94        17
#         dragonfly       1.00      1.00      1.00        16
#   electric_guitar       0.95      1.00      0.98        21
#          elephant       0.90      0.95      0.92        19
#               emu       1.00      0.93      0.97        15
#         euphonium       1.00      1.00      1.00        13
#              ewer       1.00      1.00      1.00        21
#             ferry       1.00      1.00      1.00        14
#          flamingo       1.00      0.95      0.97        20
#     flamingo_head       0.91      0.91      0.91        11
#          garfield       0.89      1.00      0.94         8
#           gerenuk       1.00      0.90      0.95        10
#        gramophone       1.00      1.00      1.00        17
#       grand_piano       1.00      0.97      0.98        33
#         hawksbill       0.71      1.00      0.83        12
#         headphone       1.00      1.00      1.00        13
#          hedgehog       0.93      1.00      0.97        14
#        helicopter       0.92      1.00      0.96        23
#              ibis       0.95      0.95      0.95        22
#      inline_skate       1.00      1.00      1.00         5
#       joshua_tree       0.70      1.00      0.82        14
#          kangaroo       0.89      1.00      0.94        25
#             ketch       0.88      0.90      0.89        31
#              lamp       0.79      1.00      0.88        15
#            laptop       0.94      1.00      0.97        16
#             llama       0.92      0.70      0.79        33
#           lobster       1.00      0.50      0.67        14
#             lotus       0.70      0.73      0.72        26
#          mandolin       1.00      1.00      1.00        13
#            mayfly       0.89      0.89      0.89         9
#           menorah       1.00      0.95      0.98        22
#         metronome       1.00      0.83      0.91         6
#           minaret       0.96      1.00      0.98        24
#          nautilus       0.90      1.00      0.95         9
#           octopus       0.78      0.88      0.82         8
#             okapi       1.00      0.92      0.96        12
#            pagoda       1.00      0.92      0.96        12
#             panda       0.78      1.00      0.88         7
#            pigeon       1.00      0.88      0.93         8
#             pizza       0.93      0.93      0.93        15
#          platypus       1.00      0.70      0.82        10
#           pyramid       0.89      1.00      0.94        16
#          revolver       0.88      1.00      0.94        22
#             rhino       0.90      1.00      0.95        19
#           rooster       1.00      1.00      1.00        13
#         saxophone       1.00      1.00      1.00         7
#          schooner       0.80      0.80      0.80        15
#          scissors       1.00      0.93      0.96        14
#          scorpion       1.00      0.95      0.97        20
#         sea_horse       0.83      0.88      0.86        17
#            snoopy       1.00      0.88      0.93         8
#       soccer_ball       1.00      1.00      1.00        16
#           stapler       1.00      0.90      0.95        10
#          starfish       0.94      0.94      0.94        18
#       stegosaurus       0.94      0.89      0.91        18
#         stop_sign       0.95      1.00      0.97        19
#        strawberry       0.90      1.00      0.95         9
#         sunflower       1.00      1.00      1.00        21
#              tick       1.00      1.00      1.00        14
#         trilobite       1.00      1.00      1.00        14
#          umbrella       0.94      0.84      0.89        19
#             watch       0.98      0.94      0.96        50
#       water_lilly       0.58      0.50      0.54        14
#        wheelchair       1.00      1.00      1.00         9
#          wild_cat       0.75      1.00      0.86         3
#     windsor_chair       1.00      1.00      1.00        15
#            wrench       0.86      0.75      0.80         8
#          yin_yang       1.00      0.90      0.95        20
#
#          accuracy                           0.93      2286
#         macro avg       0.93      0.92      0.92      2286
#      weighted avg       0.94      0.93      0.93      2286
