from argparse import ArgumentParser
from sys import argv
from datetime import datetime
from os import makedirs, path
import logging
from torch import cuda
from torch.backends import cudnn
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from bohb_worker import PyTorchWorker as worker

arg_parser = ArgumentParser(
    description="Perform HPO with BOHB to train a generative model"
)
if cuda.device_count() > 1:
    arg_parser.add_argument(
        "-o",
        "--output-dir",
        default="/srv/nfs/data/mengze/vae/bohb/",
        required=False,
        help="Directory for log files, save states and BOHB output",
    )
else:
    arg_parser.add_argument(
        "-o",
        "--output-dir",
        default="C:/Users/OGMENGLI/Projects/HyperFair/fair_hpo_bohb",
        required=False,
        help="Directory for log files, save states and BOHB output",
    )
arg_parser.add_argument(
    "--image_size",
    default=64,
    type=int,
    required=False,
    help="Image size used for loading the dataset",
)
arg_parser.add_argument(
    "--batch_size",
    default=144,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
arg_parser.add_argument(
    "--max_epochs",
    default=3,
    type=int,
    required=False,
    help="maximum epochs used for training the model",
)
arg_parser.add_argument(
    "--min_epochs",
    default=1,
    type=int,
    required=False,
    help="minimum epochs used for training the model",
)

arg_parser.add_argument(
    '--worker',
    help='Flag to turn this into a worker process',
    action='store_true'
)
args = arg_parser.parse_args(argv[1:])
log_file_path = path.join(args.output_dir, "log.txt")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, force=True)
print(f"Logging started with Output Directory { args.output_dir}")




logging.info(f"Script started")

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
    logging.info(
        f"Memory allocation was selected to be performed on {device_count} "
        f"CUDA device{'s' if device_count > 1 else ''}"
    )
else:
    device = "cpu"
    device_count = 1
    logging.info("Memory allocation was selected to be performed on the CPU device")

if cudnn.is_available():
    cudnn.benchmark = True
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)
logging.info(
    f"Data will be loaded with image size {args.image_size} and batch size {args.batch_size}"
)
logging.info(f"Generative models will be trained for maximum {args.max_epochs} epochs and miminum {args.min_epochs} epochs")

if __name__ == "__main__":
    host = '127.0.0.1'

    if args.worker:
        w = worker(run_id="example1", nameserver=host)
        #w.load_nameserver_credentials(working_directory=output_directory)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=args.output_dir, overwrite=True)
    '''
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1',  port=0, working_directory=args.output_dir)
    ns_host, ns_port = NS.start()
    w = worker(run_id='example1', host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    w.run(background=True)
    bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = 'example1', host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port,
                  result_logger=result_logger, min_budget=3, max_budget=5
               )
    res = bohb.run(n_iterations=10)
    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    '''
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    w = worker(nameserver='127.0.0.1', run_id='example1')
    w.run(background=True)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(configspace=w.get_configspace(),
                run_id='example1', nameserver='127.0.0.1',
                result_logger=result_logger,
                min_budget=1, max_budget=3
                )
    res = bohb.run(n_iterations=2)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    
