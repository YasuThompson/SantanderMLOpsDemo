# Copyright 2022 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from kfp import compiler
from kfp import dsl
from kfp.client import Client


QUERY = open("../../../database/add_cleaned_data_sample.sql", "r").read()

@dsl.container_component
def container_no_input():
    return dsl.ContainerSpec(
        image='localhost:32000/kubeflow-images/mssql-cmd:latest',
        command=[
            "/entrypoint.sh",
        ],
        args=[
            os.environ["MSSQL_SERVER"],
            os.environ["MSSQL_PASSWORD"],
            os.environ["MSSQL_USER"],
            QUERY
        ],
    )


@dsl.pipeline(name='v2-container-component-no-input')
def pipeline_container_no_input():
    container_no_input()


if __name__ == '__main__':
    # execute only if run as a script
    from utils.utils import get_istio_auth_session

    KUBEFLOW_ENDPOINT = "http://10.64.140.43"  # this is default Kubeflow endpoint by juju
    KUBEFLOW_USERNAME = "admin"  # this is default Kubeflow username by juju
    KUBEFLOW_PASSWORD = "admin"  # this is default Kubeflow password by juju

    auth_session = get_istio_auth_session(
        url=KUBEFLOW_ENDPOINT,
        username=KUBEFLOW_USERNAME,
        password=KUBEFLOW_PASSWORD
    )

    client = Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline",
                        cookies=auth_session["session_cookie"])
    print(client.list_experiments(namespace="admin-kubeflow"))

    client.create_run_from_pipeline_func(pipeline_container_no_input,
                                         run_name="clean data component",
                                         namespace="admin-kubeflow")
