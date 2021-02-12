#!/usr/bin/env python3

# Does neural style transfer for each content image and style image pair
# Run from root of project

import subprocess
from pathlib import Path

DEBUG=False

def run_command(args):
    if DEBUG:
        print(args)
    else:
        subprocess.run(args)

run_command(['touch', 'test'])

data_dir = Path('data')
content_dir = data_dir / 'content-images'
style_dir = data_dir / 'style-images'

content_images = [f.name for f in list(content_dir.iterdir())]
style_images = [f.name for f in list(style_dir.iterdir())]

i = 0
for content_image in content_images:
    for style_image in style_images:
        run_command(['python3', 'neural_style_transfer.py',
            '--content_img_name', content_image,
            '--style_img_name', style_image,
            '--height', '720'])
        i += 1
        print('Finished image ' + str(i))

