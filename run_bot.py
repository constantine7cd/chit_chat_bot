import click

from bot.telegram_bot import ChitChatBot
from bot.utils import read_yaml


@click.command()
@click.option('--config_path', help='Bot configuration path.')
def run(config_path):
    config = read_yaml(config_path)

    bot = ChitChatBot(**config)
    bot.start()


if __name__ == '__main__':
    run()
