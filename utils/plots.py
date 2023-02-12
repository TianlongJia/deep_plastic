import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_results_loss(r, training_title, label_name_loss, label_name_val_loss, colors, path):
    plt.figure(figsize=(13, 6))

    for key, c in zip(r, colors):
        plt.plot(r[key].history['loss'], label=f'{label_name_loss}: {key}', color=c)
        plt.plot(r[key].history['val_loss'], label=f'{label_name_val_loss}: {key}', ls='--', color=c)

    plt.xlabel('Epochs')
    plt.ylabel('Loss [-]')
    plt.title(training_title, fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(path, bbox_inches='tight')


def plot_results_acc(r, training_title, label_name_acc, label_name_val_acc, colors, path):
    plt.figure(figsize=(13, 6))

    for key, c in zip(r, colors):
        plt.plot(r[key].history['accuracy'], label=f'{label_name_acc}: {key}', color=c)
        plt.plot(r[key].history['val_accuracy'], label=f'{label_name_val_acc}: {key}', ls='--', color=c)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [-]')
    plt.title(training_title, fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(path, bbox_inches='tight')

def plot_hist(hist, model_name):
    # plot model accuracy
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy_"+ model_name)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    # plot model loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss_"+ model_name)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def save_plot_hist(hist, save_fig_acc_path, save_fig_loss_path):
    # plot model accuracy
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(save_fig_acc_path)
    plt.close()
    
    # plot model loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(save_fig_loss_path)
    plt.close()
   
