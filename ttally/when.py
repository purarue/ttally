#!/usr/bin/env python3

"""
Auxiliary code that's used in CLI scripts to query ttally in complex ways

See bin/ttally-when for example usage
"""

import sys
import json
import time
from typing import (
    cast,
    Any,
    Union,
    NamedTuple,
    TypeGuard,
    Iterator,
    Type,
    Callable,
    Optional,
    Literal,
)

if sys.version_info >= (3, 11):
    from typing import assert_never  # Python 3.11+
else:
    from typing_extensions import assert_never  # Python < 3.11

from functools import cached_property
from datetime import datetime, timedelta

import click
import ttally.core as ttally
import more_itertools


class CachedExtension(ttally.Extension):
    @cached_property
    def _cache(self) -> dict[str, list[NamedTuple]]:
        return {}

    def glob_namedtuple(self, nt: Type[NamedTuple]) -> Iterator[NamedTuple]:
        type_name = nt.__name__
        if type_name not in self._cache:
            self._cache[type_name] = list(super().glob_namedtuple(nt))
        return more_itertools.always_iterable(self._cache[type_name])


# TODO: does this actually speed anything up to cache?
# @cache
def _extract_dt_attr(item: Type[NamedTuple]) -> str:
    return CachedExtension.namedtuple_extract_from_annotation(item, datetime)


def when(item: NamedTuple) -> datetime:
    dt_attr = _extract_dt_attr(type(item))
    dt_val = cast(datetime, getattr(item, dt_attr))
    return dt_val.astimezone()


def since(item: NamedTuple) -> timedelta:
    return datetime.now().astimezone() - when(item)


def recent(results: list[NamedTuple]) -> Optional[NamedTuple]:
    if len(results) == 0:
        return None
    return max(results, key=when)


QueryFunc = Callable[[NamedTuple], bool]


def _infer_model(query: QueryFunc, *, ext: ttally.Extension) -> Type[NamedTuple]:
    import inspect

    # inspect the callable name
    # to determine the model type

    params = list(inspect.signature(query).parameters)
    if len(params) != 1:
        raise ValueError(
            f"Query must take exactly one argument, the name of the model, known={list(ext.MODELS)}"
        )
    if params[0] not in ext.MODELS:
        raise ValueError(
            f"Unknown model type, known={list(ext.MODELS)}, got={params[0]}"
        )

    model_type = ext.MODELS[params[0]]
    return model_type


MINUTE_IN_HOURS = 1 / 60


def dt_humanize(dt: datetime) -> str:
    import arrow

    # if more than two days away, then just say in 'days'
    # otherwise, say hours
    hours_distance = abs((dt.timestamp() - time.time()) / 3600)
    if hours_distance > 48:
        return arrow.get(dt).humanize(granularity=["day"])
    else:
        if hours_distance < MINUTE_IN_HOURS:
            return arrow.get(dt).humanize()
        elif hours_distance < 2:
            return arrow.get(dt).humanize(granularity=["minute"])
        else:
            return arrow.get(dt).humanize(granularity=["hour"])


# TODO: add table
LineFormat = Literal["human", "json"]


def format_dt(dt: datetime, date_fmt: str) -> str:
    match date_fmt:
        case "epoch":
            return str(dt.timestamp())
        case "human":
            return dt_humanize(dt)
        case "iso":
            return dt.isoformat()
        case "date":
            return dt.strftime("%Y-%m-%d")
        case _:
            try:
                return dt.strftime(date_fmt)
            except ValueError as e:
                raise ValueError(
                    "Invalid date format, should be one of epoch, human, iso, date, or a valid strftime format"
                ) from e


def desc(
    item: Optional[NamedTuple] = None,
    *,
    date_fmt: str = "human",
    name: Optional[Union[str, Callable[[Optional[NamedTuple]], str]]] = None,
    line_format: LineFormat = "human",
    with_timedelta: Optional[timedelta] = None,
) -> str | None:
    """
    a helper that lets me print a description of an item in a more useful way

    with_timedelta: if provided, will also include some fields that add the timedelta to the date.
    name: if provided, will use string. A callable can also be passed, or an 'attribute string', like
            'food.food' or 'food.when' to get the value of that attribute on the item

    this lets me see the last time I did something, and when I should do it next
    """

    use_name: str
    if name is None and item is not None:
        use_name = item.__class__.__name__.casefold()
    elif callable(name):
        use_name = name(item)
    else:
        use_name = name or "Untitled"

    if item is None:
        match line_format:
            case "human":
                return None
            case "json":
                return json.dumps({"name": name, "when": None})
            case _:
                assert_never(line_format)

    dt: datetime
    with_timedelta_dt: datetime | None = None
    td_str: str | None = None

    if item is not None:
        dt = when(item)
        use_dt: str = format_dt(dt, date_fmt)

        if with_timedelta:
            with_timedelta_dt = dt + with_timedelta
            td_str = format_dt(with_timedelta_dt, date_fmt)

    buf: str
    match line_format:
        case "human":
            buf = f"{use_name}: {use_dt}"
            if td_str:
                buf += f" (next {td_str})"
        case "json":
            d = {
                "name": use_name,
                "when": use_dt,
                "epoch": int(dt.timestamp()),
            }
            if td_str and with_timedelta_dt:
                d["next"] = td_str
                d["next_epoch"] = int(with_timedelta_dt.timestamp())
                d["expired"] = with_timedelta_dt < datetime.now().astimezone()

            buf = json.dumps(d)
        case _:
            assert_never(line_format)

    return buf


def descs(items: list[Optional[NamedTuple]], **kwargs: Any) -> list[str | None]:
    return [desc(item, **kwargs) for item in items]


class Query(NamedTuple):
    filter: QueryFunc
    raw_str: str
    model_type: Type[NamedTuple]
    action: Optional[Callable[[Union[NamedTuple, list[NamedTuple]]], None]]
    action_on_results: bool = False

    @classmethod
    def validate_query(cls, s: Any) -> Callable[[NamedTuple], bool]:
        if s.strip() == "":
            raise ValueError("Expected query (lambda function), got empty string")
        try:
            query = eval(s)
            if cls._validate_type(query):
                return query
            else:
                raise ValueError(f"query is not valid {s=} {query=}")
        except Exception as e:
            raise ValueError(f"Could not eval {s}") from e

    @staticmethod
    def _validate_type(s: Any) -> TypeGuard[Callable[[NamedTuple], bool]]:
        return callable(s)
        # todo: stricter validation by inspecting signature/arg count?

    @classmethod
    def from_str(cls, s: str, ext: ttally.Extension) -> "Query":

        if ">>>" in s:
            query_str, _, action_str = s.partition(">>>")
            query = cls.validate_query(query_str)
            Model = _infer_model(query, ext=ext)
            action = f"lambda results: {action_str}"

            return Query(
                filter=query,
                raw_str=s,
                model_type=Model,
                action=eval(action),
                action_on_results=True,
            )

        elif ">>" in s:
            query_str, _, action_str = s.partition(">>")
            query = cls.validate_query(query_str)
            Model = _infer_model(query, ext=ext)
            action = f"lambda {Model.__name__.casefold()}: {action_str}"
            return Query(
                filter=query,
                raw_str=s,
                model_type=Model,
                action=eval(action),
                action_on_results=False,
            )

        else:
            query = cls.validate_query(s)
            Model = _infer_model(query, ext=ext)
            return Query(
                filter=query,
                raw_str=s,
                model_type=Model,
                action=None,
                action_on_results=False,
            )

    def run_action(self, item: Union[list[NamedTuple], NamedTuple]) -> None:
        if self.action:
            try:
                self.action(item)
            except NameError as ne:
                if ne.name == "results":
                    if ">>>" not in self.raw_str:
                        click.echo(
                            f"Error: For '{self.raw_str}', to use the 'results' variable, you must have >>> instead of >> in your query",
                            err=True,
                        )
                        exit(1)
                elif ne.name == self.model_type.__name__.casefold():
                    if ">>>" in self.raw_str:
                        click.echo(
                            f"Error: For '{self.raw_str}', when using >>>, you must use the variable 'results' to refer to the list of results, Use >> to access each item individually",
                            err=True,
                        )
                        exit(1)
                raise ne

    def run(self, ext: ttally.Extension) -> None:
        items = []
        for item in ext.glob_namedtuple(self.model_type):
            if self.filter(item):
                if not self.action:
                    print(item)
                else:
                    if self.action_on_results:
                        items.append(item)
                    else:
                        self.run_action(item)

        if self.action_on_results and self.action:
            self.run_action(items)
