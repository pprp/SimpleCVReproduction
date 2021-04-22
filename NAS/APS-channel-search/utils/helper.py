import logging
import os
import random
import string


def init_logging(log_path):

  if not os.path.isdir(os.path.dirname(log_path)):
    print("Log path does not exist. Create a new one.")
    os.makedirs(os.path.dirname(log_path))

  log = logging.getLogger()
  log.setLevel(logging.INFO)
  logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

  fileHandler = logging.FileHandler(log_path)
  fileHandler.setFormatter(logFormatter)
  log.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  log.addHandler(consoleHandler)


def print_args(args):
  for k, v in zip(args.keys(), args.values()):
    logging.info("{0}: {1}".format(k, v))


def generate_job_id():
  return ''.join(random.sample(string.ascii_letters+string.digits, 8))
