# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Test the fake service
"""

from time import time
from test.base import QiskitExperimentsTestCase
from qiskit_experiments.test.fake_service import FakeService


class TestFakeService(QiskitExperimentsTestCase):
    """
    Test the fake service
    """

    def setUp(self):
        super().setUp()

        self.service = FakeService()
        
        self.expdict = {}
        expid = 0
        for experiment_type in range(2):
            for backend_name in range(2):
                for tags in range(2):
                    expentry = {"experiment_id": str(expid), "experiment_type": str(experiment_type), "backend_name": str(backend_name), "tags": ["a"+str(tags), "b"+str(tags)]}
                    if expid>2:
                        expentry["parent_id"] = str(expid%3)
                    self.service.create_experiment(**expentry)
                    if expid in [0, 1, 6, 7]:
                        expentry["device_components"] = [0, 1]
                    self.expdict[str(expid)] = expentry
                    expid += 1

        self.resdict = {}
        resid = 0
        for experiment_id in [0, 1, 6, 7]:
            for result_type in range(2):
                for samebool4all in range(2):
                    tags = samebool4all
                    quality = samebool4all
                    verified = samebool4all
                    result_data = samebool4all
                    device_components = samebool4all
                    resentry = {"experiment_id": str(experiment_id), "result_type": str(result_type), "result_id": str(resid), "tags": ["a"+str(tags), "b"+str(tags)], "quality": quality, "verified": verified, "result_data": {"value": result_data}, "device_components": [device_components]}
                    self.service.create_analysis_result(**resentry)
                    resentry["backend_name"] = self.expdict[str(experiment_id)]["backend_name"]
                    self.resdict[str(resid)] = resentry
                    resid += 1

    def test_creation(self):
        for df, reference_dict, id_field in zip([self.service.exps, self.service.results], [self.expdict, self.resdict], ["experiment_id", "result_id"]):
            self.assertEqual(len(df), len(reference_dict))
            is_in_frame = []
            for i in range(len(df)):
                full_entry = df.loc[i, :].to_dict()
                id = full_entry[id_field]
                self.assertTrue(id not in is_in_frame)
                is_in_frame.append(id)
                self.assertTrue(id in reference_dict.keys())
                entry = reference_dict[id]
                self.assertTrue(entry.items() <= full_entry.items())

    def test_single_experiment_query(self):
        for expid in range(8):
            full_entry = self.service.experiment(str(expid))
            entry = self.expdict[str(expid)]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_experiments_query(self):
        for experiment_type in range(2):
            expids = sorted([exp["experiment_id"] for exp in self.service.experiments(experiment_type=str(experiment_type))])
            ref_expids = sorted([exp["experiment_id"] for exp in self.expdict.values() if exp["experiment_type"] == str(experiment_type)])
            self.assertEqual(expids, ref_expids)

        for backend_name in range(2):
            expids = sorted([exp["experiment_id"] for exp in self.service.experiments(backend_name=str(backend_name))])
            ref_expids = sorted([exp["experiment_id"] for exp in self.expdict.values() if exp["backend_name"] == str(backend_name)])
            self.assertEqual(expids, ref_expids)

            
