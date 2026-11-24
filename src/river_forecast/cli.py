"""Команды загрузки данных и оценки бейзлайна."""

import argparse
import json
from pathlib import Path

import pandas as pd

from river_forecast.artifacts import frame_sha256, write_json
from river_forecast.config import load_config
from river_forecast.demo import make_demo
from river_forecast.errors import describe_error
from river_forecast.pipeline import config_metadata, evaluate_dataset, fetch_dataset, journal


class CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if message.startswith("the following arguments are required:"):
            hint = "Укажите команду demo, fetch или evaluate."
        elif message.startswith("unrecognized arguments: "):
            hint = f"Неизвестные аргументы: {message.split(': ', 1)[1]}"
        else:
            argument, _, detail = message.removeprefix("argument ").partition(": ")
            if detail.startswith("invalid choice: "):
                value = detail.removeprefix("invalid choice: ").split(" (choose from", 1)[0]
                hint = f"Неизвестная команда {value}. Доступны demo, fetch и evaluate."
            elif detail == "expected one argument":
                hint = f"После {argument} нужно указать значение."
            elif detail.startswith("invalid int value: "):
                hint = f"Для {argument} нужно целое число."
            else:
                hint = f"Проверьте аргументы. Детали: {message}"
        self.exit(2, f"{hint}\nСправка: --help\n")


def main():
    parser = CommandParser(description="Прогноз уровня воды по данным USGS и Open-Meteo")
    commands = parser.add_subparsers(dest="command", metavar="{demo,fetch,evaluate}", required=True)
    demo = commands.add_parser("demo", help="Прогнать расчёт на синтетических данных без сети")
    demo.add_argument("--seed", type=int, default=42, help="Seed генератора данных")
    for name, help_text in (
        ("fetch", "Загрузить гидрологию и погоду"),
        ("evaluate", "Посчитать бейзлайн на загруженных данных"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--config", type=Path, default=Path("config/suwannee_branford.toml"), help="Путь к TOML"
        )
    args = parser.parse_args()
    try:
        parameters = (
            {"kind": "synthetic_demo", "seed": args.seed}
            if args.command == "demo"
            else {"config": str(args.config)}
        )
        with journal(Path("artifacts"), args.command, parameters):
            if args.command == "demo":
                processed, artifacts = Path("data/processed/demo"), Path("artifacts/demo")
                metadata = parameters
                daily = make_demo(seed=args.seed)
                processed.mkdir(parents=True, exist_ok=True)
                daily.to_csv(processed / "daily.csv")
                write_json(processed / "dataset.json", metadata)
                metrics = evaluate_dataset(daily, processed, artifacts, metadata)
            else:
                config = load_config(args.config)
                processed = Path("data/processed") / args.config.stem
                artifacts = Path("artifacts") / args.config.stem
                if args.command == "fetch":
                    daily = fetch_dataset(config, processed, Path("data/raw"))
                    print(f"Загружено дней: {len(daily)}. Данные в {processed}")
                    return
                metadata = json.loads((processed / "dataset.json").read_text(encoding="utf-8"))
                if metadata["config"] != config_metadata(config):
                    raise ValueError("Конфиг изменился. Перезагрузите данные командой fetch.")
                if metadata["daily_sha256"] != frame_sha256(processed / "daily.csv"):
                    raise ValueError("Контрольная сумма daily.csv не совпала. Запустите fetch.")
                daily = pd.read_csv(processed / "daily.csv", index_col="date", parse_dates=["date"])
                metrics = evaluate_dataset(
                    daily,
                    processed,
                    artifacts,
                    metadata,
                    config.weather_delay_days,
                    config.holdout_start,
                )
        print(json.dumps(metrics, ensure_ascii=True, indent=2, allow_nan=False))
        print(f"Результаты в {artifacts}")
    except Exception as exc:
        parser.exit(1, f"Ошибка. {describe_error(exc)}\n")
