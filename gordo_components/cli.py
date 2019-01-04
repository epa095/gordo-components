# -*- coding: utf-8 -*-

'''
CLI interfaces
'''

import logging
import os
from ast import literal_eval

import yaml
import click
from gordo_components.builder import build_model
from gordo_components.server import server

import dateutil.parser

# Set log level, defaulting to DEBUG
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level),
                    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)


@click.group('gordo-components')
def gordo():
    """
    The main entry point for the CLI interface
    """
    pass


DEFAULT_MODEL_CONFIG = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': 'feedforward_symetric'}}"


@click.command()
@click.argument('output-dir',
                default='/data',
                envvar='OUTPUT_DIR')
@click.argument('model-config',
                envvar='MODEL_CONFIG',
                default=DEFAULT_MODEL_CONFIG,
                type=yaml.load)
@click.argument('data-config',
                envvar='DATA_CONFIG',
                default='{"type": "influx"}',
                type=literal_eval)
def build(output_dir, model_config, data_config):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.
    """

    # TODO: Move all data related input from environment variable to data_config,
    # TODO: thereby removing all these data_config['variable'] lines

    data_config['tag_list'] = literal_eval(os.environ.get('TAGS', '[]'))

    # TODO: Move parsing from here, into the InfluxDataSet class
    data_config['from_ts'] = dateutil.parser.isoparse(os.environ['TRAIN_START_DATE'])

    # TODO: Move parsing from here, into the InfluxDataSet class
    data_config['to_ts'] = dateutil.parser.isoparse(os.environ['TRAIN_END_DATE'])

    logger.info('Building, output will be at: {}'.format(output_dir))
    logger.info('Model config: {}'.format(model_config))
    logger.info('Data config: {}'.format(data_config))

    build_model(output_dir=output_dir,
                model_config=model_config,
                data_config=data_config)
    logger.info('Successfully built model, and deposited at {}'.format(output_dir))
    return 0


@click.command('run-server')
@click.option('--host', type=str, help='The host to run the server on.')
@click.option('--port', type=int, help='The port to run the server on.')
def run_server_cli(host, port):
    server.run_server(host, port)


gordo.add_command(build)
gordo.add_command(run_server_cli)
