from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Report evaluation statistics


def cnn_statistics(model, ds_test):
    # compute predictions (we need to transform probabilities into the predicted class using argmax, e.g. the position with the maximum value)
    # axis = 1 ensures that we do this for all rows in the dataset
    y_hat_test = np.argmax(model.predict(ds_test), axis=1)
    y_val = np.concatenate([y for x, y in ds_test], axis=0)

    print(f'Print classification report on validation(test) dataset')
    print(f'-----'*10)
    print(classification_report(y_val, y_hat_test))


# Plot confusion matrix

def plot_confusion_matrix(cm,
                          classes, 
                          title,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # we quickly create a decent looking CM for many classes using pandas and seaborn
    # print(f'\nPlot confusion matrix on validation(test) dataset')
    # print(f'-----'*10)
    # df_cm = pd.DataFrame(confusion_matrix(y_val,y_hat_test))
    # f, ax = plt.subplots(1,figsize = (10,10))
    # sns.heatmap(df_cm, annot=True, annot_kws={"fontsize":12}, ax= ax, vmax=50, cbar=False, cmap='Blues', fmt='d') # we use this value of vmax to emphasize misclassifications
    # ax.set_title(f'{title}', fontsize=16, fontweight='bold')
    # ax.set_ylabel('True class');
    # ax.set_xlabel('Predicted class');

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def ConfusionMatrix(title, model, ds_test, num_classes):
    y_test = np.concatenate([y for x, y in ds_test], axis=0)
    x_test = np.concatenate([x for x, y in ds_test], axis=0)
    p_test = model.predict(ds_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(num_classes)), title)