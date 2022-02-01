# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fake service class for tests."""

from typing import Optional, List, Dict, Type, Any, Union, Tuple
import json
from datetime import datetime, timedelta
import pandas as pd

from qiskit_experiments.test.fake_backend import FakeBackend

from qiskit_experiments.database_service import DatabaseServiceV1
from qiskit_experiments.database_service.device_component import DeviceComponent


class FakeService(DatabaseServiceV1):
    """
    This extremely simple database is designated for testing and as a playground for developers.
    It does not support multi-threading.
    It is not guaranteed to perform well for a large amount of data.
    It implements most of the methods of `DatabaseService`.
    """

    def __init__(self):
        self.exps = pd.DataFrame(
            columns=[
                "experiment_type",
                "backend_name",
                "metadata",
                "experiment_id",
                "parent_id",
                "job_ids",
                "tags",
                "notes",
                "figure_names",
            ]
        )
        self.results = pd.DataFrame(
            columns=[
                "experiment_id",
                "result_data",
                "result_type",
                "device_components",
                "tags",
                "quality",
                "verified",
                "result_id",
                "chisq",
                "creation_datetime",
                "service",
            ]
        )

    def create_experiment(
        self,
        experiment_type: str,
        backend_name: str,
        metadata: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        """Create a new experiment in the database.

        Args:
            experiment_type: Experiment type.
            backend_name: Name of the backend the experiment ran on.
            metadata: Experiment metadata.
            experiment_id: Experiment ID. It must be in the ``uuid4`` format.
            parent_id: The experiment ID of the parent experiment.
                The parent experiment must exist, must be on the same backend as the child,
                and an experiment cannot be its own parent.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            notes: Freeform notes about the experiment.
            json_encoder: Custom JSON encoder to use to encode the experiment.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Experiment ID.

        Raises:
            ValueError: if the experiment_id parameter is None.
                The fake service requires this parameter, and does not generate it by itself.
        """

        if experiment_id is None:
            raise ValueError("The fake service requires the experiment id parameter")

        # Clarifications about some of the columns:
        # share_level - not a parameter of `DatabaseService.create_experiment` but a parameter of
        #    `IBMExperimentService.create_experiment`. It must be supported because it is used
        #    in `DbExperimentData`.
        # device_components - the user speicifies the device components when adding a result
        #    (this is not a local decision of the fake service but the interface of DatabaseService
        #    and IBMExperimentService). The components of the different results of the same
        #    experiment are aggregated here in the device_components column.
        # start_datetime - not a parameter of `DatabaseService.create_experiment` but a parameter of
        #    `IBMExperimentService.create_experiment`. Since `DbExperimentData` does not set it
        #    via kwargs (as it does with share_level), the user cannot control the time and the
        #    service alone decides about it. Here we've chosen to set a unique time for each
        #    experiment, with the first experiment dated to midnight of January 1st, 2022, the
        #    second exeperiment an hour later, etc.
        # figure_names - the fake service currently does not support figures. The column
        #    (degenerated to []) is required to prevent a flaw in the work with DbExperimentData.
        self.exps = self.exps.append(
            {
                "experiment_type": experiment_type,
                "experiment_id": experiment_id,
                "parent_id": parent_id,
                "backend_name": backend_name,
                "metadata": metadata,
                "job_ids": job_ids,
                "tags": tags,
                "notes": notes,
                "share_level": kwargs.get("share_level", None),
                "device_components": [],
                "start_datetime": datetime(2022, 1, 1) + timedelta(hours=len(self.exps)),
                "figure_names": [],
            },
            ignore_index=True,
        )

        return experiment_id

    def update_experiment(
        self,
        experiment_id: str,
        metadata: Optional[Dict] = None,
        job_ids: Optional[List[str]] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing experiment.

        Args:
            experiment_id: Experiment ID.
            metadata: Experiment metadata.
            job_ids: IDs of experiment jobs.
            notes: Freeform notes about the experiment.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.
        """
        row = self.exps.experiment_id == experiment_id
        if metadata is not None:
            self.exps.loc[row, "metadata"] = metadata
        if job_ids is not None:
            self.exps.loc[row, "job_ids"] = job_ids
        if tags is not None:
            self.exps.loc[row, "tags"] = tags
        if notes is not None:
            self.exps.loc[row, "notes"] = notes
        if "share_level" in kwargs:
            self.exps.loc[row, "share_level"] = kwargs["share_level"]
        if "parent_id" in kwargs:
            self.exps.loc[row, "parent_id"] = kwargs["parent_id"]

    def experiment(
        self, experiment_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.
            json_decoder: Custom JSON decoder to use to decode the retrieved experiment.

        Returns:
            A dictionary containing the retrieved experiment data.
        """
        db_entry = self.exps.loc[self.exps.experiment_id == experiment_id].to_dict("records")[0]

        # DbExperimentData expects an instansiated backend object, and not the backend name
        db_entry["backend"] = FakeBackend(db_entry["backend_name"])

        return db_entry

    def experiments(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        df = self.exps

        if experiment_type is not None:
            df = df.loc[df.experiment_type == experiment_type]

        # TODO: do we have to return the backend itself, as in `experiment`?
        if backend_name is not None:
            df = df.loc[df.backend_name == backend_name]

        # Note a bug in the interface for all services:
        # It is impossible to filter by experiments whose parent id is None
        # (i.e., root experiments)
        if parent_id is not None:
            df = df.loc[df.parent_id == parent_id]

        # Waiting for consistency between provider service and qiskit-experiments service,
        # currently they have different types for `device_components`
        if device_components is not None:
            raise ValueError(
                "The fake service currently does not support filtering on device components"
            )

        if tags is not None:
            if tags_operator == "OR":
                df = df.loc[df.tags.apply(lambda dftags: any(x in dftags for x in tags))]
            elif tags_operator == "AND":
                df = df.loc[df.tags.apply(lambda dftags: all(x in dftags for x in tags))]
            else:
                raise ValueError("Unrecognized tags operator")

        # These are parameters of IBMExperimentService.experiments
        if "start_datetime_before" in filters:
            df = df.loc[df.start_datetime <= filters["start_datetime_before"]]
        if "start_datetime_after" in filters:
            df = df.loc[df.start_datetime >= filters["start_datetime_after"]]

        # This is a parameter of IBMExperimentService.experiments
        sort_by = filters.get("sort_by", "start_datetime:desc")

        if not isinstance(sort_by, list):
            sort_by = [sort_by]

        # TODO: support also experiment_type
        if len(sort_by) != 1:
            raise ValueError("The fake service currently supports only sorting by start_datetime")

        sortby_split = sort_by[0].split(":")
        # TODO: support also experiment_type
        if (
            len(sortby_split) != 2
            or sortby_split[0] != "start_datetime"
            or (sortby_split[1] != "asc" and sortby_split[1] != "desc")
        ):
            raise ValueError(
                "The fake service currently supports only sorting by start_datetime, which can be "
                "either asc or desc"
            )

        df = df.sort_values(
            ["start_datetime", "experiment_id"], ascending=[(sortby_split[1] == "asc"), True]
        )

        df = df.iloc[:limit]

        return df.to_dict("records")

    def delete_experiment(self, experiment_id: str) -> None:
        index = self.exps[self.exps.experiment_id == experiment_id].index
        self.exps.drop(index, inplace=True)

    def create_analysis_result(
        self,
        experiment_id: str,
        result_data: Dict,
        result_type: str,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = False,
        result_id: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        # Clarifications about some of the columns:
        # backend_name - taken from the experiment.
        # creation_datetime - start_datetime - not a parameter of
        #    `DatabaseService.create_analysis_result` but a parameter of
        #    `IBMExperimentService.create_analysis_result`. Since `DbExperimentData` does not set it
        #    via kwargs (as it does with chisq), the user cannot control the time and the service
        #    alone decides about it. Here we've chosen to set the start date of the experiment.
        self.results = self.results.append(
            {
                "result_data": result_data,
                "result_id": result_id,
                "result_type": result_type,
                "device_components": device_components,
                "experiment_id": experiment_id,
                "quality": quality,
                "verified": verified,
                "tags": tags,
                "backend_name": self.exps.loc[self.exps.experiment_id == experiment_id]
                .iloc[0]
                .backend_name,
                "chisq": kwargs.get("chisq", None),
                "creation_datetime": self.exps.loc[self.exps.experiment_id == experiment_id]
                .iloc[0]
                .start_datetime,
            },
            ignore_index=True,
        )

        # a helper method for updating the experiment's device components, see usage below
        def add_new_components(expcomps):
            for dc in device_components:
                if dc not in expcomps:
                    expcomps.append(dc)

        # update the experiment's device components
        self.exps.loc[self.exps.experiment_id == experiment_id, "device_components"].apply(
            add_new_components
        )

        return result_id

    def update_analysis_result(
        self,
        result_id: str,
        result_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = None,
        **kwargs: Any,
    ) -> None:
        row = self.results.result_id == result_id
        if result_data is not None:
            self.results.loc[row, "result_data"] = result_data
        if tags is not None:
            self.results.loc[row, "tags"] = tags
        if quality is not None:
            self.results.loc[row, "quality"] = quality
        if verified is not None:
            self.results.loc[row, "verified"] = verified
        if "chisq" in kwargs:
            self.results.loc[row, "chisq"] = kwargs["chisq"]

    def analysis_result(
        self, result_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        # The `experiment` method implements special handling of the backend, we skip it here.
        # It's a bit strange, so, if not required by `DbExperimentData` then we'd better skip.
        return self.results.loc[self.results.result_id == result_id].to_dict("records")[0]

    def analysis_results(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_id: Optional[str] = None,
        result_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        quality: Optional[str] = None,
        verified: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        df = self.results

        # TODO: skipping device components for now until we conslidate more with the provider service
        # (in the qiskit-experiments service there is no opertor for device components,
        # so the specification for filtering is not clearly defined)

        if experiment_id is not None:
            df = df.loc[df.experiment_id == experiment_id]
        if result_type is not None:
            df = df.loc[df.result_type == result_type]
        if backend_name is not None:
            df = df.loc[df.backend_name == backend_name]
        if quality is not None:
            df = df.loc[df.quality == quality]
        if verified is not None:
            df = df.loc[df.verified == verified]

        if tags is not None:
            if tags_operator == "OR":
                df = df.loc[df.tags.apply(lambda dftags: any(x in dftags for x in tags))]
            elif tags_operator == "AND":
                df = df.loc[df.tags.apply(lambda dftags: all(x in dftags for x in tags))]
            else:
                raise ValueError("Unrecognized tags operator")

        # This is a parameter of IBMExperimentService.experiments
        sort_by = filters.get("sort_by", "creation_datetime:desc")

        if not isinstance(sort_by, list):
            sort_by = [sort_by]

        # TODO: support also device components and result type
        if len(sort_by) != 1:
            raise ValueError(
                "The fake service currently supports only sorting by creation_datetime"
            )

        sortby_split = sort_by[0].split(":")
        # TODO: support also device components and result type
        if (
            len(sortby_split) != 2
            or sortby_split[0] != "creation_datetime"
            or (sortby_split[1] != "asc" and sortby_split[1] != "desc")
        ):
            raise ValueError(
                "The fake service currently supports only sorting by creation_datetime, "
                "which can be either asc or desc"
            )

        df = df.sort_values(
            ["creation_datetime", "result_id"], ascending=[(sortby_split[1] == "asc"), True]
        )

        df = df.iloc[:limit]
        return df.to_dict("records")

    def delete_analysis_result(self, result_id: str) -> None:
        index = self.results[self.results.result_id == result_id].index
        self.results.drop(index, inplace=True)

    def create_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: Optional[str]
    ) -> Tuple[str, int]:
        pass

    def update_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: str
    ) -> Tuple[str, int]:
        pass

    def figure(
        self, experiment_id: str, figure_name: str, file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        pass

    def delete_figure(
        self,
        experiment_id: str,
        figure_name: str,
    ) -> None:
        pass

    @property
    def preferences(self) -> Dict:
        return {"auto_save": False}
