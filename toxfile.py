"""
tox plugin which allows adding an entry called ``alternative_commands`` to a
tox environment configuration section with different commands to run instead of
the main ``commands`` entry. The plugin also adds an ``--alt`` command line
option for selecting the alternative command sets. To use the alternative
commands, one must specify the runner as ``alternative_commands``.

For example, one might have an environment for running ``black --check`` but
set up some alternate commands to check the ``black`` help or version (these
are not exciting use cases but illustrate the plugin features).  To do this one
would have an environment config entry like

    [testenv:black]
    runner = alternative_commands
    skip_install = true
    setenv =
        ENVVAR1 = yes
    deps =
      black
    commands = black --check
    alternative_commands =
      help =
        add_setenv ENVVAR2 = no
        black --help
      version =
        black --version
        python --version

Then instead of the usual invocation of

    tox run -eblack

one could use

    tox run -eblack --alt help

to print the black help text.

In addition to overriding commands, ``alternative_commands`` entries can also
include ``add_setenv`` lines which get filtered out from the commands and used
to add extra variables to the ``setenv`` configuration of the test environment.
For example, in the ``--alt help`` example, in addition to ``ENVVAR1`` being
set to ``yes``, ``ENVVAR2`` will be set to ``no``.
"""
from __future__ import annotations

from tox.config.cli.parser import ToxParser
from tox.config.loader.api import Override
from tox.config.loader.stringify import stringify
from tox.plugin import impl
from tox.tox_env.python.virtual_env.runner import VirtualEnvRunner
from tox.tox_env.register import ToxEnvRegister


@impl
def tox_register_tox_env(register: ToxEnvRegister) -> None:
    """Register the ``AlternativeCommandsRunner`` tox environment type"""
    register.add_run_env(AlternativeCommandsRunner)


@impl
def tox_add_option(parser: ToxParser):
    """Add the ``--alt-command`` command line option"""
    parser.add_argument("--alt", "--alt-command", default="", dest="alt_command")


class AlternativeCommandsRunner(VirtualEnvRunner):
    """tox environment type that can run alternative commands

    Modified ``VirtualEnvRunner`` that can override settings if the
    ``--alt-command`` option is used with an argument that matches an
    alternative command set in the test environment configuration. See the
    module docstring for more information.
    """

    def register_config(self):
        self.conf.add_config(
            keys=["alternative_commands"],
            of_type=str,
            default="",
            desc="Alternative command sets that can be selected from command line",
        )
        super().register_config()
        alt_cmd = self.options.alt_command
        if alt_cmd:
            alt_commands = self.parse_alternative_commands()
            if alt_cmd in alt_commands:
                # It would be nice to warn here if alt_cmd is not in
                # alt_commands, but register_config gets called for every
                # environment before the selected environments are chosen. So
                # register_config could be called for an environment that does
                # not define alt_cmd. More work would be needed to make sure a
                # warning were only emitted when alt_cmd is not defined for the
                # selected environments.
                self.conf.loaders[0].overrides["commands"] = Override(
                    "commands=" + stringify(alt_commands[alt_cmd]["commands"])[0]
                )
                if alt_commands[alt_cmd].get("add_setenv"):
                    orig_setenv = self.conf.load("setenv")
                    orig_setenv.update(alt_commands[alt_cmd]["add_setenv"])
                    self.conf.loaders[0].overrides["setenv"] = Override(
                        "setenv=" + stringify(orig_setenv)[0]
                    )

    @staticmethod
    def id() -> str:
        return "alternative_commands"

    def parse_alternative_commands(self) -> dict[str, list[str]]:
        """Parse the alternative_commands config entry

        We turn an entry like

            alternative_commands =
              set1 =
                add_setenv OMP_NUM_THREADS = 1
                add_setenv RAYON_NUM_THREADS = 1
                cmd1 arg1
                cmd2 arg2
              set2 =
                cmd3 arg3
                cmd4 arg4

        which gets rendered a newline separated string into a dictionary like

            {
                "set1": {
                    "add_setenv": {
                        "OMP_NUM_THREADS": "1",
                        "RAYON_NUM_THREADS": "1",
                    },
                    "commands": [
                        "cmd1 arg1",
                        "cmd2 arg2",
                    ],
                },
                "set2": {
                    "commands": [
                        "cmd3 arg3",
                        "cmd4 arg4",
                    ],
                },
            }

        We don't parse the commands because we pass them as a config override
        and the config loader wants to load the override as a single string,
        not as a parsed list of Command objects.
        """
        alternative_commands = self.conf["alternative_commands"]

        output = {}
        name = ""
        for line in alternative_commands.splitlines():
            line = line.strip()
            if line.endswith("="):
                name = line[:-1].strip()
                continue

            if not name:
                raise ValueError(f"Badly formed alternative_commands: {alternative_commands}")
            if line:
                alt_entry = output.setdefault(name, {})
                if line.startswith("add_setenv"):
                    subline = line[len("add_setenv") :].strip()
                    key, sep, value = subline.partition("=")
                    if not sep:
                        raise ValueError(f"Badly formed add_setenv: {subline}")
                    alt_entry.setdefault("add_setenv", {})[key.strip()] = value.strip()
                else:
                    alt_entry.setdefault("commands", []).append(line)

        return output
