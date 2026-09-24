# ==============================================================
# CIFAR-100 FOOD CONTAINER CLASSIFICATION
# Classes:
# 0 = bottle
# 1 = bowl
# 2 = can
# 3 = cup
# 4 = plate
# ==============================================================

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================
# 1. BASIC INFORMATION
# ==============================================================

print("\n========================================")
print("CIFAR-100 FOOD CONTAINER CLASSIFICATION")
print("========================================")

print("TensorFlow version:", tf.__version__)


# Food container superclass in CIFAR-100
COARSE_CLASS = 3

class_names = [
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate"
]


# ==============================================================
# 2. LOAD CIFAR-100
# ==============================================================

print("\nLoading CIFAR-100...")

# Load coarse labels
(train_x_coarse, train_coarse_labels), (
    test_x_coarse,
    test_coarse_labels
) = keras.datasets.cifar100.load_data(
    label_mode="coarse"
)


# Load fine labels
(train_images_all, train_fine_labels), (
    test_images_all,
    test_fine_labels
) = keras.datasets.cifar100.load_data(
    label_mode="fine"
)


print("Dataset loaded successfully.")


# ==============================================================
# 3. EXTRACT FOOD CONTAINER IMAGES
# ==============================================================

print("\nExtracting food-container images...")


# Find training images belonging to coarse class 3
train_indices = np.where(
    train_coarse_labels.flatten() == COARSE_CLASS
)[0]


# Find testing images belonging to coarse class 3
test_indices = np.where(
    test_coarse_labels.flatten() == COARSE_CLASS
)[0]


train_images = train_images_all[train_indices]

train_labels_original = train_fine_labels[
    train_indices
].flatten()


test_images = test_images_all[test_indices]

test_labels_original = test_fine_labels[
    test_indices
].flatten()


print("Training images:", len(train_images))
print("Testing images:", len(test_images))

print(
    "Original CIFAR-100 fine labels:",
    np.unique(train_labels_original)
)


# ==============================================================
# 4. RELABEL CLASSES TO 0–4
# ==============================================================

fine_classes = np.unique(train_labels_original)

print("\nFine-class mapping:")

for new_label, original_label in enumerate(fine_classes):

    print(
        original_label,
        "->",
        new_label,
        "->",
        class_names[new_label]
    )


label_map = {
    old_label: new_label
    for new_label, old_label
    in enumerate(fine_classes)
}


train_labels = np.array(
    [label_map[label] for label in train_labels_original]
)


test_labels = np.array(
    [label_map[label] for label in test_labels_original]
)


# ==============================================================
# 5. SHOW CLASS DISTRIBUTION
# ==============================================================

class_counts = []

for i in range(5):

    count = np.sum(train_labels == i)

    class_counts.append(count)


plt.figure(figsize=(8, 5))

plt.bar(
    class_names,
    class_counts
)

plt.title(
    "Training Images per Food-Container Class"
)

plt.xlabel("Class")

plt.ylabel("Number of Images")

plt.tight_layout()

plt.show()


# ==============================================================
# 6. DISPLAY SAMPLE IMAGES
# ==============================================================

plt.figure(figsize=(12, 6))


for class_number in range(5):

    class_indexes = np.where(
        test_labels == class_number
    )[0]

    # show 2 examples for every class
    for example_number in range(2):

        image_index = class_indexes[
            example_number
        ]

        plot_number = (
            class_number * 2
            + example_number
            + 1
        )

        plt.subplot(
            5,
            2,
            plot_number
        )

        plt.imshow(
            test_images[image_index]
        )

        plt.title(
            class_names[class_number]
        )

        plt.axis("off")


plt.suptitle(
    "Sample CIFAR-100 Food Container Images"
)

plt.tight_layout()

plt.show()


# ==============================================================
# 7. NORMALIZE IMAGE VALUES
# ==============================================================

# Original RGB values range from 0 to 255.
#
# Dividing by 255 changes the range to 0–1.
#
# Neural networks generally train more easily when
# inputs are on a smaller numerical scale.

train_images = (
    train_images.astype("float32")
    / 255.0
)

test_images = (
    test_images.astype("float32")
    / 255.0
)


print("\nImage values normalized to 0–1.")


# ==============================================================
# 8. BUILD CNN
# ==============================================================

print("\nBuilding neural network...")


model = keras.Sequential([

    # ------------------------------
    # CONVOLUTION BLOCK 1
    # ------------------------------

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3)
    ),

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.25),


    # ------------------------------
    # CONVOLUTION BLOCK 2
    # ------------------------------

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.25),


    # ------------------------------
    # CLASSIFICATION SECTION
    # ------------------------------

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        5,
        activation="softmax"
    )
])


# Show neural network structure
model.summary()


# ==============================================================
# 9. COMPILE MODEL
# ==============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==============================================================
# 10. TRAIN MODEL
# ==============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")


history = model.fit(

    train_images,

    train_labels,

    epochs=20,

    batch_size=32,

    validation_split=0.20,

    verbose=1
)


print("\nTraining finished.")


# ==============================================================
# 11. ACCURACY GRAPH
# ==============================================================

training_accuracy = history.history[
    "accuracy"
]

validation_accuracy = history.history[
    "val_accuracy"
]


epochs = range(
    1,
    len(training_accuracy) + 1
)


plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    training_accuracy,
    label="Training Accuracy"
)


plt.plot(
    epochs,
    validation_accuracy,
    label="Validation Accuracy"
)


plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==============================================================
# 12. LOSS GRAPH
# ==============================================================

training_loss = history.history[
    "loss"
]

validation_loss = history.history[
    "val_loss"
]


plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    training_loss,
    label="Training Loss"
)


plt.plot(
    epochs,
    validation_loss,
    label="Validation Loss"
)


plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==============================================================
# 13. TEST THE MODEL
# ==============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")


test_loss, test_accuracy = model.evaluate(

    test_images,

    test_labels,

    verbose=1

)


print(
    "\nTest Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Accuracy (%):",
    test_accuracy * 100
)


# ==============================================================
# 14. MAKE PREDICTIONS
# ==============================================================

predictions = model.predict(

    test_images,

    verbose=0

)


predicted_labels = np.argmax(

    predictions,

    axis=1

)


# ==============================================================
# 15. CONFUSION MATRIX
# ==============================================================

confusion_matrix = tf.math.confusion_matrix(

    test_labels,

    predicted_labels,

    num_classes=5

).numpy()


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    confusion_matrix
)


plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)


plt.xticks(
    range(5),
    class_names,
    rotation=45
)

plt.yticks(
    range(5),
    class_names
)


# Put numbers inside each box
for i in range(5):

    for j in range(5):

        plt.text(

            j,

            i,

            confusion_matrix[i, j],

            ha="center",

            va="center"

        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ==============================================================
# 16. PER-CLASS ACCURACY
# ==============================================================

class_accuracy = []


print(
    "\n========================================"
)

print(
    "ACCURACY PER CLASS"
)

print(
    "========================================"
)


for i in range(5):

    correct = confusion_matrix[i, i]

    total = np.sum(
        confusion_matrix[i]
    )

    accuracy = (
        correct / total
    )

    class_accuracy.append(
        accuracy
    )


    print(

        class_names[i],

        ":",

        round(
            accuracy * 100,
            2
        ),

        "%"

    )


# Graph per-class accuracy

plt.figure(figsize=(8, 5))


plt.bar(

    class_names,

    np.array(class_accuracy) * 100

)


plt.title(
    "Classification Accuracy per Class"
)

plt.xlabel(
    "Food Container"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.show()


# ==============================================================
# 17. DISPLAY RANDOM PREDICTIONS
# ==============================================================

random_indices = np.random.choice(

    len(test_images),

    10,

    replace=False

)


plt.figure(
    figsize=(15, 6)
)


for i, image_index in enumerate(
    random_indices
):


    plt.subplot(
        2,
        5,
        i + 1
    )


    plt.imshow(
        test_images[image_index]
    )


    predicted_class = (
        predicted_labels[image_index]
    )


    true_class = (
        test_labels[image_index]
    )


    confidence = (

        predictions[
            image_index
        ][
            predicted_class
        ]

        * 100

    )


    title = (

        "Pred: "
        + class_names[
            predicted_class
        ]

        + "\nTrue: "
        + class_names[
            true_class
        ]

        + "\n"
        + str(
            round(
                confidence,
                1
            )
        )

        + "%"

    )


    plt.title(
        title,
        fontsize=9
    )

    plt.axis(
        "off"
    )


plt.suptitle(
    "Example Model Predictions"
)

plt.tight_layout()

plt.show()


# ==============================================================
# 18. FINAL RESULTS
# ==============================================================

print(
    "\n========================================"
)

print(
    "FINAL RESULTS"
)

print(
    "========================================"
)


print(
    "Training accuracy:",
    round(
        training_accuracy[-1] * 100,
        2
    ),
    "%"
)


print(
    "Validation accuracy:",
    round(
        validation_accuracy[-1] * 100,
        2
    ),
    "%"
)


print(
    "Testing accuracy:",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


print(
    "\nProgram finished."
)