"""Wandb-related functionality."""
from __future__ import annotations
from functools import lru_cache
from typing import Sequence

from loguru import logger
import pandas as pd
import wandb

from ranzen.misc import flatten_dict


@lru_cache(None)
def get_api() -> wandb.Api:
    return wandb.Api()


class RunsDownloader:
    """Download logged runs from W&B."""

    def __init__(self, *, project: str, entity: str) -> None:
        self.project = project
        self.entity = entity
        self.api = get_api()

    def runs(self, *run_ids: str) -> pd.DataFrame:
        """Download runs given the run IDs (e.g., "qvlp96vk").

        :param run_ids: IDs for the runs to download.

        :returns: the downloaded runs as a pandas DataFrame
        """
        runs = []
        for run_id in run_ids:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            runs.append(run)
        return self._runs_to_df(runs)

    def groups(self, *groups_: str) -> pd.DataFrame:
        """Download all runs in a group."""
        path = f"{self.entity}/{self.project}"
        dfs = []
        for group in groups_:
            runs = self.api.runs(path, filters={"group": group})
            logger.info(f"'{group}': found {len(runs)} runs.")
            dfs.append(self._runs_to_df(runs))
        return pd.concat(dfs, axis=0, sort=False, keys=list(groups_))

    def modify_config(
        self, *, group: str, config_key: str, new_value: bool | int | float | str
    ) -> None:
        """Modify the config value of runs logged on W&B.

        This is not possible with the web UI.
        """
        path = f"{self.entity}/{self.project}"
        runs = self.api.runs(path, filters={"group": group})
        i = 0
        for i, run in enumerate(runs, start=1):
            run.config[config_key] = new_value
            run.update()
        logger.info(f"Changed config for {i} runs.")

    @staticmethod
    def _runs_to_df(runs: Sequence[wandb.wandb_run.Run]) -> pd.DataFrame:  # type: ignore
        summary_list = []
        config_list = []
        name_list = []
        for run in runs:
            # run.summary are the output key/values like accuracy.
            # We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)
            # run.config is the input metrics.  We remove special values that start with _.
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            config_list.append(flatten_dict(config, sep="."))
            # run.name is the name of the run.
            name_list.append(run.name)
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame({"name": name_list})
        return pd.concat([name_df, config_df, summary_df], axis=1)
