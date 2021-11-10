import click


@click.command("init_configuration", help="Compile Configuration file")
@click.option("-f", "--filename", required=True, help="Output filename.")
@click.option(
    "-n", "--configuration_name", default="default", help="Configuration scheme name."
)
def init_configuration(filename: str, configuration_name: str):

    from leakers.trainers.factory import LeakersConfigurationsBucket
    from choixe.configurations import XConfig

    if configuration_name not in LeakersConfigurationsBucket.CONFIGURATIONS_MAP:
        raise ValueError(
            f"Configuration scheme {configuration_name} not found. "
            f"Available schemes: {LeakersConfigurationsBucket.CONFIGURATIONS_MAP.keys()}"
        )

    cfg = LeakersConfigurationsBucket.CONFIGURATIONS_MAP[configuration_name]
    cfg = XConfig.from_dict(cfg)
    cfg.save_to(filename)
