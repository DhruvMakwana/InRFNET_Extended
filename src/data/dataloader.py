import tensorflow as tf
import numpy as np
import cv2

# data loader
class ctScanDataLoader(tf.keras.utils.Sequence):
    """Dataloader to iterate over the data. This class will return batch of 
    images and labels."""

    def __init__(self, batchSize, imgSize, inputImgPaths, inputLabels, classes):
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.inputImgPaths = inputImgPaths
        self.inputLabels = inputLabels
        self.classes = classes
        
    def __len__(self):
        return len(self.inputImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batchInputImgPaths = self.inputImgPaths[i : i + self.batchSize]
        batchInputLabels = self.inputLabels[i : i + self.batchSize]
        
        images = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype="float32")
        labels = np.zeros((self.batchSize,) + (self.classes,), dtype="float32")
        for j, (input_image, label) in enumerate(zip(batchInputImgPaths, batchInputLabels)):
            img = cv2.imread(str(input_image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # random crop
            max_x = img.shape[1] - self.imgSize[1]
            max_y = img.shape[0] - self.imgSize[0]

            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)

            img = img[y: y + self.imgSize[1], x: x + self.imgSize[0]]
            images[j] = img.astype("float32") / 255.0
            
            lbl = tf.keras.utils.to_categorical(label, num_classes=self.classes)
            labels[j] = lbl

        return images, labels