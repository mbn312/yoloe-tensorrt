from __future__ import annotations

import argparse
from typing import TypeAlias, cast

import yaml

from ._shapes import HWShape

YamlValue: TypeAlias = (
    str | int | float | bool | None | list["YamlValue"] | tuple["YamlValue", ...] | dict[str, "YamlValue"]
)


def parse_positive_int(value: str, option_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must contain integer value(s).") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be greater than zero.")
    return parsed


def parse_imgsz(value: str, option_name: str = "--imgsz") -> int | HWShape:
    raw_value = value.strip().lower()
    if not raw_value:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty.")

    for separator in ("x", ","):
        if separator in raw_value:
            parts = [part.strip() for part in raw_value.split(separator)]
            if len(parts) != 2 or not all(parts):
                raise argparse.ArgumentTypeError(f"{option_name} must be one integer or two integers as HxW or H,W.")
            height, width = (parse_positive_int(part, option_name) for part in parts)
            return (height, width)
    return parse_positive_int(raw_value, option_name)


def load_yaml_value(value: str) -> YamlValue:
    try:
        return cast(YamlValue, yaml.safe_load(value))
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"could not parse YAML scalar {value!r}: {exc}") from exc


def parse_key_value_args(
    entries: list[str] | None,
    *,
    option_name: str,
    error_type: type[Exception] = ValueError,
) -> dict[str, YamlValue]:
    values: dict[str, YamlValue] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise error_type(f"{option_name} must use KEY=VALUE syntax.")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise error_type(f"{option_name} keys must not be empty.")
        if key in values:
            raise error_type(f"{option_name} key {key!r} was provided more than once.")
        try:
            values[key] = "" if raw_value == "" else load_yaml_value(raw_value)
        except argparse.ArgumentTypeError as exc:
            raise error_type(str(exc)) from exc
    return values
