"""
    FeatureCloud Data Distributor Application
    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from FeatureCloud.engine.app import app_state, AppState, Role, LogLevel
from FeatureCloud.engine.app import State as op_state
import pandas as pd
import bios
from utils import save_numpy, load_numpy, sep_feat_from_label, log_send_data, log_data
import ConfigState

from utils import log_dataframe, plot_clients_data, noniid_sampling, unsupervised_iid_sampling, supervised_iid_sampling

name = 'fc_data_distributor'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class DistributeData(ConfigState.State):

    def register(self):
        self.register_transition('WriteResults', Role.BOTH)

    def run(self):
        self.lazy_init()
        if self.is_coordinator:
            self.read_config()
            self.finalize_config()
            self.sanity_check()
            file_name = self.load('input_files')['data'][0]
            df = self.load_dataset(file_name)
            clients_data = self.sample_dataset(df)

            plot_clients_data(clients_data, self.output_dir)
            config_file = bios.read(self.config_file)
            for client in self.clients:
                client_data = clients_data[clients_data.ASSIGNED_CLIENT == client]
                log_send_data([client_data, config_file], self.log)
                self.send_data_to_participant(data=[client_data, config_file], destination=client)
            self.store('config', self.config)
        else:
            self.store('splits', {'temp'})

        return 'WriteResults'

    def sanity_check(self):
        self.config['format'] = self.config['local_dataset']['data'].strip().split(".")[-1].lower()
        if not self.config['format'] in ['txt', 'npy', 'npz', 'csv']:
            self.log(f"Unsupported {self.config['format']} file extension!", LogLevel.ERROR)
            self.update(state=op_state.ERROR)
        self.config['sampling']['type'] = self.config['sampling']['type'].lower()
        if not self.config['sampling']['type'] in ['non-iid', 'noniid', 'non_iid', 'iid']:
            self.log(f"Unsupported {self.config['sampling']['type']} type!", LogLevel.ERROR)
            self.update(state=op_state.ERROR)

    def load_dataset(self, file_name):
        """ load dataset with npy(NumPy), txt(text), and csv(supporting different separators) extensions.


        Parameters
        ----------
        file_name: str

        Returns:
        -------
        df: pandas.DataFrame

        """
        if self.config['format'] in ['txt', 'csv']:
            df = pd.read_csv(file_name, sep=self.config['local_dataset']['sep'])
            if self.config['local_dataset']['task'] == "classification":
                df = df.rename(columns={self.config['local_dataset']['target_value']: 'label'})
                self.log(df.columns, LogLevel.DEBUG)
                self.log(log_dataframe(df), LogLevel.DEBUG)
            return df
        if self.config['format'] in ['npy', 'npz']:
            ds = load_numpy(file_name)

            self.log(f"Number of rows: {len(ds)}\n"
                     f"Number of columns: {len(ds[0])}", LogLevel.DEBUG)
            if self.config['local_dataset']['task'] == 'classification':
                df = sep_feat_from_label(ds, self.config['local_dataset']['target_value'])
                if df is None:
                    self.log(f"{self.config['local_dataset']['target_value']} is not supported", LogLevel.ERROR)
                    self.update(state=op_state.ERROR)
                self.log(f"Number of unique labels: {len(df.label.unique())}", LogLevel.DEBUG)
                return df

            # Regression or Clustering
            return pd.DataFrame({"features": [s for s in ds]})

    def sample_dataset(self, df):
        non_iid_ness = self.config['sampling']['non_iid_ness']
        if self.config['sampling']['type'] in ['non-iid', 'noniid', 'non_iid']:
            labels = df.label.unique()
            if int(non_iid_ness) <= 0 or int(non_iid_ness) > len(labels):
                self.log(f"Level of Non-IID-ness is restricted to the number of classes!\n"
                         f"Number of labels: {len(labels)}"
                         f"\nNon-IID-ness: {non_iid_ness}", LogLevel.FATAL)
                self.update(state=op_state.ACTION)
        df['ASSIGNED_CLIENT'] = None
        if self.config['local_dataset']['task'] == 'classification':
            if self.config['sampling']['type'] == 'iid':
                clients_data = supervised_iid_sampling(df, self.clients)
            else:
                clients_data = noniid_sampling(df, self.clients, non_iid_ness)
        else:
            clients_data = unsupervised_iid_sampling()
        return clients_data


@app_state(name='WriteResults', role=Role.BOTH)
class WriteResults(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str:
        data, config_file = self.await_data(n=1, unwrap=True, is_json=False)
        if self.is_coordinator:
            file_name = self.load('output_files')['data'][0]
            target = self.load('config')['local_dataset']['target_value']
            sep = self.load('config')['local_dataset']['sep']
        else:
            log_data(data, self.log)
            log_data(config_file, self.log)
            output_path = "/mnt/output/"
            file_name = output_path + config_file[name]['result']['data']
            config_filename = output_path + 'config.yml'
            target = config_file[name]['local_dataset']['target_value']
            sep = config_file[name]['local_dataset']['sep']
            bios.write(config_filename, config_file)
        format = file_name.split('.')[-1]
        if format in ['npy', 'npz']:
            save_numpy(file_name, data.features.values, data.label.values, target)
        else:
            data.rename(columns={'label': target}, inplace=True)
            data.to_csv(file_name, sep=sep, index=False)
        return 'terminal'
