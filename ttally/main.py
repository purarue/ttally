import sys
import json
from typing import (
    NamedTuple,
    Optional,
    List,
    Sequence,
    Iterable,
    Any,
    Literal,
    Union,
    Generator,
)
from datetime import timedelta, datetime
from contextlib import contextmanager

import click
import autotui.exceptions

from .core import Extension, CacheError


@contextmanager
def handle_autotui_errors() -> Generator[None, None, None]:
    try:
        yield
    except autotui.exceptions.AutoTUIException as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _parse_recent(value: Union[str, int]) -> Union[int, timedelta, Literal["all"]]:
    if isinstance(value, int):
        return value
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        pass

    import re
    from datetime import timedelta

    timedelta_regex = re.compile(
        r"^((?P<weeks>[\.\d]+?)w)?((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?$"
    )

    # This uses a syntax similar to the 'GNU sleep' command
    # e.g.: 1w5d5h10m50s means '1 week, 5 days, 5 hours, 10 minutes, 50 seconds'
    parts = timedelta_regex.match(value)
    if parts is not None:
        time_params = {
            name: float(param) for name, param in parts.groupdict().items() if param
        }
        return timedelta(**time_params)

    raise click.BadParameter(
        f"{value} is not 'all', a valid integer, or a timedelta (e.g. 2d, 5h, 20m)"
    )


def wrap_accessor(*, extension: Extension) -> click.Group:
    @click.group()
    def call_main() -> None:
        """
        Tally things that I do often!

        Given a few namedtuples, this creates serializers/deserializers
        and an interactive interface using 'autotui', and aliases
        to:

        prompt using default autotui behavior, writing to the ttally datafile,
        same as above, but if the model has a datetime, set it to now,
        query the 10 most recent items for a model
        """
        pass

    @call_main.command(short_help="generate shell aliases")
    def generate() -> None:
        """
        Generate the shell aliases!
        """
        for a in extension.generate_shell_aliases():
            print(a)

    def _model_complete(
        ctx: click.Context, args: Sequence[str], incomplete: str
    ) -> List[str]:
        return [
            m for m in extension._autocomplete_model_names() if m.startswith(incomplete)
        ]

    model_with_completion = click.argument("MODEL", shell_complete=_model_complete)

    @call_main.command(short_help="add item by piping JSON")
    @model_with_completion
    @click.option(
        "-p",
        "--partial",
        default=False,
        is_flag=True,
        help="Allow partial input -- prompt any fields which aren't provided",
    )
    @click.option(
        "-f",
        "--file",
        default=None,
        type=click.Path(exists=True),
        help="Read from file instead of STDIN",
    )
    def from_json(model: str, partial: bool, file: Optional[str]) -> None:
        """
        A way to allow external programs to save JSON data to the current file for the model

        Provide a list of JSON from STDIN, and the corresponding model to parse it to
        (in lowercase) as the first argument, and this parses (validates)
        and saves it to the file
        """

        with handle_autotui_errors():
            if file is None:
                extension.save_from(
                    extension._model_from_string(model),
                    use_input=sys.stdin,
                    partial=partial,
                )
            else:
                with open(file, "r") as f:
                    extension.save_from(
                        extension._model_from_string(model),
                        use_input=f,
                        partial=partial,
                    )

    @call_main.command(short_help="print the datafile location")
    @model_with_completion
    @click.argument(
        "PATH_TYPE",
        type=click.Choice(["datafile", "merged", "cached"]),
        default="datafile",
    )
    def datafile(
        model: str, path_type: Literal["datafile", "merged", "cached"]
    ) -> None:
        """
        Print the location of the current datafile for some model
        """
        extension._model_from_string(model)
        if path_type == "cached":
            click.echo(extension.cache_file(model))
        elif path_type == "merged":
            click.echo(extension.ttally_merged_path(model))
        else:
            click.echo(extension.datafile(model))

    @call_main.command(name="prompt", help="tally an item")
    @model_with_completion
    def _prompt(model: str) -> None:
        """
        Prompt for every field in the given model
        """
        with handle_autotui_errors():
            extension.prompt(extension._model_from_string(model))

    @call_main.command(name="models", help="list models")
    def _models_cmd() -> None:
        """
        List all ttally models
        """
        click.echo("\n".join(extension._autocomplete_model_names()))

    @call_main.command(name="prompt-now", help="tally an item (now)")
    @model_with_completion
    def _prompt_now(model: str) -> None:
        """
        Prompt for every field in the model, except datetime, which should default to now
        """
        with handle_autotui_errors():
            extension.prompt_now(extension._model_from_string(model))

    @call_main.command(name="recent", short_help="print recently tallied items")
    @model_with_completion
    @click.option(
        "-r",
        "--remove-attrs",
        type=str,
        default="",
        help="comma separated list of attributes to remove while printing",
    )
    @click.option(
        "-o",
        "--output-format",
        type=click.Choice(["json", "table"]),
        default="table",
        help="how to print output",
    )
    @click.option(
        "-h/-d",
        "--human-readable/--exact-dates",
        is_flag=True,
        default=False,
        help="print dates in a human readable format",
    )
    @click.argument(
        "COUNT",
        default=10,
        type=click.UNPROCESSED,
        callback=lambda ctx, arg, value: _parse_recent(value),
    )
    def _recent(
        model: str,
        remove_attrs: str,
        count: Union[int, timedelta, Literal["all"]],
        output_format: Literal["json", "table"],
        human_readable: bool,
    ) -> None:
        """
        List recent items logged for this model

        Can provide 'all' for COUNT to list all items
        A number for COUNT to list that many items
        Or a timedelta (e.g. 2d, 5h, 20m) to list all items in that time range
        """
        nt = extension._model_from_string(model)

        # try to load cached data
        res: Optional[List[NamedTuple]] = None
        try:
            from autotui.serialize import deserialize_namedtuple

            # reverse so it is ordered for query properly
            res_iter = list(reversed(extension.read_cache_json(model=model)))
            res_items = extension.take_items(res_iter, count, nt)
            res = [
                deserialize_namedtuple(o, to=extension.MODELS[model]) for o in res_items
            ]
        except CacheError:
            pass

        attrs = [a.strip() for a in remove_attrs.split(",") if a.strip()]
        extension.query_print(
            extension._model_from_string(model),
            count,
            output_format=output_format,
            remove_attrs=attrs,
            cached_data=res,
            human_readable=human_readable,
        )

    @call_main.command(short_help="export all data from a model")
    @model_with_completion
    @click.option(
        "-s",
        "--stream",
        default=False,
        is_flag=True,
        help="Stream objects as they're read, instead of a list",
    )
    def export(model: str, stream: bool) -> None:
        """
        List all the data from a model as JSON
        """

        # read from cache if cache isn't stale
        itr: Optional[Iterable[Any]] = None
        try:
            itr = extension.read_cache_json(model=model)
        except CacheError:
            pass

        # cache was stale, read from datafiles
        if itr is None:
            from autotui.fileio import namedtuple_sequence_dumps

            itr = json.loads(
                namedtuple_sequence_dumps(
                    list(extension.glob_namedtuple(extension._model_from_string(model)))
                )
            )

        assert itr is not None

        if stream:
            for blob in itr:
                sys.stdout.write(json.dumps(blob))
                sys.stdout.write("\n")
        else:
            sys.stdout.write(json.dumps(list(itr)))
            sys.stdout.write("\n")
        sys.stdout.flush()

    @call_main.command(short_help="merge all data for a model into one file")
    @model_with_completion
    @click.option(
        "--sort-key", default=None, help="Sort resulting merged data by JSON key"
    )
    @click.option(
        "-R",
        "--remove-duplicates",
        default=False,
        is_flag=True,
        help="Remove duplicate entries from the merged data (might occur if there are errors syncing files)",
    )
    def merge(model: str, sort_key: Optional[str], remove_duplicates: bool) -> None:
        """
        Merge all datafiles for one model into a single '-merged.json' file
        """
        from pathlib import Path
        from datetime import datetime

        from autotui.fileio import namedtuple_sequence_dumps
        from more_itertools import unique_everseen

        datafiles: List[Path] = list(extension.glob_datafiles(model))
        if len(datafiles) == 0:
            click.echo(f"No datafiles for model {model}", err=True)
            return

        data = json.loads(
            namedtuple_sequence_dumps(
                list(extension.glob_namedtuple(extension._model_from_string(model)))
            )
        )

        # write backup before sorting/removing datafiles
        epoch = int(datetime.now().timestamp())
        cachefile = extension.temp_dir() / f"{model}-{epoch}-merged.json"

        click.echo(f"Writing backup to '{cachefile}'", err=True)
        with cachefile.open("w") as backup_f:
            json.dump(data, backup_f)

        # if provided, use sort key
        if sort_key is not None and len(data) > 0:
            assert sort_key in data[0], f"Could not find {sort_key} in {data[0]}"
            data = list(sorted(data, key=lambda obj: obj[sort_key]))  # type: ignore[no-any-return]

        if remove_duplicates:
            new_data = list(unique_everseen(data, key=lambda obj: json.dumps(obj)))
            if len(new_data) != len(data):
                click.echo(f"Removed {len(data) - len(new_data)} duplicates", err=True)
            else:
                click.echo("No duplicates found", err=True)
            data = new_data

        # remove current datafiles
        for rmf in datafiles:
            click.echo(f"Removing '{rmf}'", err=True)
            rmf.unlink()

        merge_target = extension.ttally_merged_path(model)
        with merge_target.open("w") as merged_f:
            json.dump(data, merged_f)

        click.echo(f"Wrote merged file to '{merge_target}'", err=True)

    @call_main.command(short_help="cache export data", name="update-cache")
    @click.option(
        "--print-hashes",
        is_flag=True,
        default=False,
        help="print current filehash debug info",
    )
    def update_cache(print_hashes: bool) -> None:
        """
        Caches data for 'export' and 'recent' by saving
        the current data and an index to ~/.cache/ttally

        exit code 0 if cache was updated, 2 if it was already up to date
        """
        was_stale = extension.cache_sorted_exports()
        ret = 0
        if was_stale:
            click.echo("Cache was stale, updated", err=True)
        else:
            click.echo("Cache is already up to date", err=True)
            ret = 2
        if print_hashes:
            click.echo(json.dumps(extension.file_hashes()))
        sys.exit(ret)

    @call_main.command(short_help="edit the datafile")
    @model_with_completion
    def edit(model: str) -> None:
        """
        Edit the current datafile with your editor
        """
        extension._model_from_string(model)
        f = extension.datafile(model)
        if not f.exists():
            click.secho(f"Warning: {f} doesn't exist. ", err=True, fg="red")
            if not click.confirm("Open anyways?"):
                return

        click.edit(filename=str(f))

    @call_main.command(short_help="fuzzy select/edit recent items")
    @click.option(
        "-l",
        "--loop",
        is_flag=True,
        default=False,
        help="prompt fields to edit multiple times",
    )
    @model_with_completion
    def edit_recent(loop: bool, model: str) -> None:
        """
        Edit recent items from a model, fuzzy selecting and then selecting fields to edit
        """
        nt = extension._model_from_string(model)
        f = extension.datafile(model)
        if not f.exists():
            click.secho(f"Error: {f} doesn't exist. ", err=True, fg="red")
            return

        from autotui.shortcuts import load_from

        data = list(load_from(to=nt, path=f))
        if len(data) == 0:
            click.secho(f"Error: No data for {model}", err=True, fg="red")
            return

        from autotui.pick import pick_namedtuple
        from autotui.edit import edit_namedtuple
        from autotui.shortcuts import dump_to

        def _nt_string(d: NamedTuple) -> str:
            return ", ".join([f"{k}: {v}" for k, v in d._asdict().items()])

        # pick data from current file
        selected = pick_namedtuple(
            data,
            fzf_options=("--tac",),
            key_func=_nt_string,
        )
        if selected is None:
            return

        # choose a field to edit and writeback
        idx = data.index(selected)
        print(f"Editing item: {_nt_string(selected)}", file=sys.stderr)
        edited = edit_namedtuple(selected, loop=loop, print_namedtuple=True)
        data[idx] = edited

        click.echo(
            f"Edited at index {idx}:\nFrom:\t{_nt_string(selected)}\nTo:\t{_nt_string(edited)}",
            err=True,
        )

        dump_to(data, f)

    @call_main.command(short_help="drop the last n items")
    @model_with_completion
    @click.argument("COUNT", type=int, default=1)
    def drop_last(model: str, count: int) -> None:
        """
        Drop the last n items from the datafile
        """
        nt = extension._model_from_string(model)
        f = extension.datafile(model)
        if not f.exists():
            click.secho(f"Error: {f} doesn't exist. ", err=True, fg="red")
            return

        from autotui.shortcuts import load_from, dump_to

        data = list(load_from(to=nt, path=f))
        if len(data) == 0:
            click.secho(f"Error: No data for {model}", err=True, fg="red")
            return

        removed = data[-count:]
        data = data[:-count]

        click.echo("Removing:", err=True)
        for r in removed:
            click.echo("-" * 20)
            for k, v in r._asdict().items():
                v = (
                    datetime.fromtimestamp(v.timestamp())
                    if isinstance(v, datetime)
                    else v
                )
                click.echo(f"{k}: {v}")

        if len(data) == 0:
            click.secho(f"Warning: No data left for {model}", err=True, fg="yellow")
            if click.confirm("Remove file?"):
                f.unlink()
        else:
            dump_to(data, f)

    return call_main
