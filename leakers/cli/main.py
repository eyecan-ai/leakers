import click
from leakers.cli.init_configuration import init_configuration
from leakers.cli.generate import generate
from leakers.cli.live import live


@click.group()
def leakers():
    pass


leakers.add_command(init_configuration)
leakers.add_command(generate)
leakers.add_command(live)
