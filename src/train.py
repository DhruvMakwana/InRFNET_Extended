# -*- coding: utf-8 -*-
from data.dataloader import ctScanDataLoader
from models.model import InRFNet
import tensorflow as tf
from config import *
from utils import *
import logging
import sys
import os

def main():
    logging.info("Creating Dataset")
    X_train, X_test, y_train, y_test = get_data(DATA_DIR)

    trainGen = ctScanDataLoader(batchSize = BATCH_SIZE, imgSize = IMG_SIZE, inputImgPaths = X_train, inputLabels = y_train, classes = CLASSES)
    testGen = ctScanDataLoader(batchSize = BATCH_SIZE, imgSize = IMG_SIZE, inputImgPaths = X_test, inputLabels = y_test, classes = CLASSES)

    logging.info("creating Model")
    net = InRFNet()
    model = net.build(IMG_SIZE[0], IMG_SIZE[1], 3, CLASSES)
    logging.info(model.summary())

    logging.info("Compiling Model")
    model.compile(optimizer = tf.keras.optimizers.Adam(LEARNING_RATE), loss = "categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('inrfnet_best', monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=0.00001)
    ]
    history = model.fit(trainGen, validation_data=testGen, epochs=NUMBER_EPOCHS, callbacks=callbacks)

if __name__ == '__main__':
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join('app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    random_seed(SEED)
    main()