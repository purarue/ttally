**TL;DR**: This converts a file like this (config file at `~/.config/ttally.py`):

```python
from datetime import datetime
from typing import NamedTuple, Optional


class Weight(NamedTuple):
    when: datetime
    pounds: float


class Food(NamedTuple):
    when: datetime
    calories: int
    food: str
    quantity: float
    water: int  # how much ml of water was in this

    # specify a special way to prompt for quantity
    @staticmethod
    def attr_validators() -> dict:
        # https://purarue.xyz/d/ttally_types.py?redirect
        from my.config.pura.ttally_types import prompt_float_default  # type: ignore

        # if I don't supply a quantity, default to 1
        return {"quantity": lambda: prompt_float_default("quantity")}


class Event(NamedTuple):
    """e.g. a concert or something"""

    event_type: str
    when: datetime
    description: str
    score: Optional[int]
    comments: Optional[str]

    @staticmethod
    def attr_validators() -> dict:
        from my.config.pura.ttally_types import edit_in_vim  # type: ignore

        return {"comments": edit_in_vim}


import os
from enum import Enum

# dynamically create an enum using each line of the file as an option
with open(os.path.join(os.environ["HPIDATA"], ".self_types.txt")) as f:
    SelfT = Enum("SelfT", [s.rstrip().upper() for s in f])


class Self(NamedTuple):
    when: datetime
    what: SelfT  # type: ignore
```

to (shell aliases)...

```
alias event='python3 -m ttally prompt event'
alias event-now='python3 -m ttally prompt-now event'
alias event-recent='python3 -m ttally recent event'
alias food='python3 -m ttally prompt food'
alias food-now='python3 -m ttally prompt-now food'
alias food-recent='python3 -m ttally recent food'
alias self='python3 -m ttally prompt self'
alias self-now='python3 -m ttally prompt-now self'
alias self-recent='python3 -m ttally recent self'
alias weight='python3 -m ttally prompt weight'
alias weight-now='python3 -m ttally prompt-now weight'
alias weight-recent='python3 -m ttally recent weight'
```

Whenever I run any of those aliases, it inspects the model in the config file, and on-the-fly creates and runs an interactive interface like this:

<img src="https://raw.githubusercontent.com/purarue/autotui/master/.assets/builtin_demo.gif">

... which saves what I enter to a file:

```yaml
- when: 1598856786,
  glasses": 2.0
```

## ttally

`ttally` is an interactive module using [`autotui`](https://github.com/purarue/autotui) to save things I do often to YAML/JSON

Currently, I use this to store info like whenever I eat something/drink water/my current weight/thoughts on concerts

Given a `NamedTuple` defined in [`~/.config/ttally.py`](https://purarue.xyz/d/ttally.py?redirect), this creates interactive interfaces which validates my input and saves it to a file

The `{tuple}-now` aliases set the any `datetime` values for the prompted tuple to now

This also gives me `{tuple}-recent` aliases, which print recent items I've logged. For example:

```
$ water-recent 5
2021-03-20 18:23:24     2.0
2021-03-20 01:28:27     1.0
2021-03-19 23:34:12     1.0
2021-03-19 22:49:05     1.5
2021-03-19 16:05:34     1.0
```

The `-recent` aliases can accept `all` to print all items, or a duration like `1d` or `6h` to print data from the last few hours/days.

## Why/How

### Goals

- validates my user input to basic types
- stores it as a user-editable format (YAML)
- can be loaded into python as typed objects
- minimal boilerplate to add a new model
- can be synced across multiple machines without conflicts
- allow completely custom types or prompts - see [autotui docs](https://github.com/purarue/autotui#custom-types), [my custom prompts](https://purarue.xyz/d/ttally_types.py?redirect)

This intentionally uses YAML and doesn't store the info into a single "merged" database. That way:

- you can just open the YAML file and quickly change/edit some item, no need to re-invent a CRUD interface (though `ttally edit-recent` does exist)
- files can be synced across machines and to my phone using [syncthing](https://syncthing.net/) without file conflicts
- prevents issues with trying to merge multiple databases from different machines together ([I've tried](https://github.com/purarue/calories-scripts/blob/master/calmerge))

The YAML files are versioned with the date/OS/platform, so I'm able to add items on my linux, mac, or android (using [`termux`](https://termux.com/)) and sync them across all my devices using [`SyncThing`](https://syncthing.net/). Each device creates its own file it adds items to, like:

```
food-darwin-mbp.localdomain-2021-03.yaml
food-linux-bastion-2021-03.yaml
food-linux-localhost-2021-04.yaml
```

... which can then be combined back into python, like:

```python
>>> from more_itertools import take  # just to grab a few items
>>> from ttally.__main__ import ext
>>> from ttally.config import Food
>>> take(3, ext.glob_namedtuple(Food))

[Food(when=datetime.datetime(2020, 9, 27, 6, 49, 34, tzinfo=datetime.timezone.utc), calories=440, food='ramen, egg'),
Food(when=datetime.datetime(2020, 9, 27, 6, 52, 16, tzinfo=datetime.timezone.utc), calories=160, food='2 eggs'),
Food(when=datetime.datetime(2020, 9, 27, 6, 53, 44, tzinfo=datetime.timezone.utc), calories=50, food='ginger chai')]
```

... or into JSON using `ttally export food`

The `from-json` command can be used to send this JSON which matches a model, i.e. providing a non-interactive interface to add items, in case I want to [call this from a script](bin/cz)

`hpi query` from [`HPI`](https://github.com/purarue/HPI) can be used with the `ttally.__main__` module, like:

```bash
# how many calories in the last day
$ hpi query ttally.__main__.food --recent 1d -s | jq -r '(.quantity)*(.calories)' | datamash sum 1
2252
```

If you'd prefer to use JSON files, you can set the `TTALLY_EXT=json` environment variable.

This can load data from YAML or JSON (or both at the same time), every couple months I'll combine all the versioned files to a single merged file using the `merge` command:

```
ttally merge food
```

## Installation

```bash
pip install ttally  # can use 'pip install ttally[optional]' for extra features
```

```
Usage: ttally [OPTIONS] COMMAND [ARGS]...

  Tally things that I do often!

  Given a few namedtuples, this creates serializers/deserializers and an
  interactive interface using 'autotui', and aliases to:

  prompt using default autotui behavior, writing to the ttally datafile, same
  as above, but if the model has a datetime, set it to now, query the 10 most
  recent items for a model

Options:
  --help  Show this message and exit.

Commands:
  datafile      print the datafile location
  drop-last     drop the last n items
  edit          edit the datafile
  edit-recent   fuzzy select/edit recent items
  export        export all data from a model
  from-json     add item by piping JSON
  generate      generate shell aliases
  merge         merge all data for a model into one file
  models        list models
  prompt        tally an item
  prompt-now    tally an item (now)
  recent        print recently tallied items
  update-cache  cache export data
```

### Configuration

You need to setup a `~/.config/ttally.py` file. You can use the block above as a starting point, or with mine:

```bash
curl -s 'https://purarue.xyz/d/ttally.py' > ~/.config/ttally.py
```

To setup aliases; You can do it each time you launch you terminal like:

```bash
eval "$(python3 -m ttally generate)"
```

Or, 'cache' the generated aliases by putting a block like this in your shell config:

```bash
TTALLY_ALIASES="${HOME}/.cache/ttally_aliases"
if [[ ! -e "${TTALLY_ALIASES}" ]]; then  # alias file doesn't exist
	python3 -m ttally generate >"${TTALLY_ALIASES}"  # generate and save the aliases
fi
source "${TTALLY_ALIASES}"  # make aliases available in your shell
```

i.e., it runs the first time I open a terminal, but then stays the same until I remove the file

You can set the `TTALLY_DATA_DIR` environment variable to the directory that `ttally` should save data to, defaults to `~/.local/share/ttally`. If you want to use a different path for configuration, you can set the `TTALLY_CFG` to the absolute path to the file.

For shell completion to autocomplete options/model names:

```
eval "$(_TTALLY_COMPLETE=bash_source ttally)"  # in ~/.bashrc
eval "$(_TTALLY_COMPLETE=zsh_source ttally)"  # in ~/.zshrc
eval "$(_TTALLY_COMPLETE=fish_source ttally)"  # in ~/.config/fish/config.fish
```

### Caching

`ttally update-cache` can be used to speedup the `export` and `recent` commands:

```
Usage: ttally update-cache [OPTIONS]

  Caches data for 'export' and 'recent' by saving the current data and an
  index to ~/.cache/ttally

  exit code 0 if cache was updated, 2 if it was already up to date

Options:
  --print-hashes  print current filehash debug info
  --help          Show this message and exit.
```

I run this using [entr](https://github.com/eradman/entr) whenever the data files change. In the background, like:

```
find ~/data/ttally -type f | entr -n ttally update-cache
```

Default cache directory can be overwritten with the `TTALLY_CACHE_DIR` environment variable

### Subclassing/Extension

The entire `ttally` library/CLI can also be subclassed/extended for custom usage, by using `ttally.core.Extension` class and `wrap_cli` to add additional [click](https://click.palletsprojects.com/en/8.1.x) commands. For an example, see [flipflop.py](https://purarue.xyz/d/flipflop.py?redirect)

### Shell Scripts

[`cz`](bin/cz) lets me fuzzy select something I've eaten in the past using [`fzf`](https://github.com/junegunn/fzf), like:

![](https://raw.githubusercontent.com/purarue/calories-fzf/master/demo.gif)
