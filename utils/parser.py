import os
import logging

def setup(args, log):
    args.name = 'hetero'
    ldir = args.savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)

    lfile = args.name + '_' + log + '.txt' if log else args.name + '.txt'
    lfile = os.path.join(ldir, lfile)

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s', filename=lfile)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
