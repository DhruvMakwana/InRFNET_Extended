# -*- coding: utf-8 -*-
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging
import click
import os

@click.command()
@click.option('--input_filepath', type=click.Path(exists=True), help="input path for raw data")
@click.option('--output_filepath', type=click.Path(), help="output path for processed data")
@click.option('--image_size', default=256, help="resize to this")
def main(input_filepath, output_filepath, image_size):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # create lists of images
    p = Path(input_filepath)
    cap_images = list(p.glob('3CAP/*'))
    pos_images = list(p.glob('2COVID/*'))
    neg_images = list(p.glob('1NonCOVID/*'))

    # check if output folders exists
    if not os.path.exists(os.path.join(output_filepath, "CAP")):
        os.makedirs(os.path.join(output_filepath, "CAP"))
    if not os.path.exists(os.path.join(output_filepath, "COVID")):
        os.makedirs(os.path.join(output_filepath, "COVID"))
    if not os.path.exists(os.path.join(output_filepath, "NonCOVID")):
        os.makedirs(os.path.join(output_filepath, "NonCOVID"))

    for i, image in enumerate(tqdm(cap_images)):
        img = Image.open(str(image))
        img = img.resize((image_size, image_size))
        img.save(os.path.join(output_filepath, "CAP", f"image_{i}.png"))
    click.echo("All CAP images are processed")

    for i, image in enumerate(tqdm(pos_images)):
        img = Image.open(str(image))
        img = img.resize((image_size, image_size))
        img.save(os.path.join(output_filepath, "COVID", f"image_{i}.png"))
    click.echo("All COVID images are processed")

    for i, image in enumerate(tqdm(neg_images)):
        img = Image.open(str(image))
        img = img.resize((image_size, image_size))
        img.save(os.path.join(output_filepath, "NonCOVID", f"image_{i}.png"))
    click.echo("All NonCOVID images are processed")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
