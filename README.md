# Gooey GPU

A set of common deep learning models that can be deployed on a Kubernetes cluster with GPU support.

## Setup Instructions

1. Create the `env-values.yaml` file:
    ```bash
    cp env-values.example.yaml env-values.yaml
    ```
   Now, open the `env-values.yaml` file and fill in the values.

2. Connect to your k8s cluster. This will change depending on the cloud provider.

   Eg for azure: 
   ``` bash
   # Set the cluster subscription
   az account set --subscription <subscription-id>
   # Download cluster credentials
   az aks get-credentials --resource-group <resource-group> --name <vm-name>
   # Set the namespace
   kubectl config set-context --current --namespace=gpu
   ```
   
3. Deploy the helm chart
    ```bash
    helm install gooey-gpu-1 chart -f chart/model-values.yaml -f env-values.yaml
    ```

4. Check the status of the deployment
    ```bash
    kubectl get pods -n gpu
    ```

## Development

### New huggingface based models

gooey-gpu includes a standard Dockerfile with common deep learning dependencies like cuda, diffusers & transformers pre-installed.

gooey-gpu also provides a small python helper library to make it easy to write celery tasks.

1. If the dependencies are not enough, you can add more dependencies to the [Dockerfile](common/Dockerfile) & [requirements.txt](common/requirements.txt) file.

2. Create a new file in the `common/` folder that imports the model and defines the load function.

   ```python
   ## common/my_model.py
   from celeryconfig import setup_queues
   from functools import lru_cache
   import os
   
   
   @lru_cache  # this will cache the model in memory and use it across calls
   def load_model(model_id: str):
       ...
   
   
   setup_queues(
       model_ids=os.environ["MY_MODEL_IDS"].split(), # get the model ids from the env
       load_fn=load_model, # this tells the celery worker to load the model when starting
   )
   ```
   
   To load custom models, you can use the provided cache directory. The helm chart includes a nfs provisioner that mounts a shared directory across all the pods. You can use this directory to store the models.

   ```python
   ## common/my_model.py
   import torch
   import os
   import gooey_gpu
   from functools import lru_cache
   
   
   @lru_cache
   def load_model(model_id: str):
       model_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, model_id)
       if not os.path.exists(model_path):
           ... # download the model from huggingface or any other source
       return torch.load(model_path).to(gooey_gpu.DEVICE_ID)
   ```
   
   You can also `kubectl exec` into a running pod a manually copy the model files to the shared directory.

   ```bash
   kubectl exec -it <pod-name> -n gpu -- bash
   cd /root/.cache/gooey-gpu/checkpoints
   ... # copy the model files here
   ```
   
    
3. Define the model inference params and code
   ```python
   ## common/my_model.py
   import gooey_gpu
   from celeryconfig import app, setup_queues
   from pydantic import BaseModel
   
   class MyModelPipeline(BaseModel):
       model_id: str
       ...
   
   class MyModelInputs(BaseModel):
       text: str
       ...
   
   class MyModelOutput(BaseModel):
       image: str
       ...

   @app.task(name="my_model_task")
   @gooey_gpu.endpoint
   def my_model_task(
       pipeline: MyModelPipeline,
       inputs: MyModelInputs,
   ) -> MyModelOutput:
       model = load_model(pipeline.model_id)
       ... # write the inference code here
       return MyModelOutput(...)
   ```
4. Install rabbitmq, redis & docker on the machine with GPUs. 
 
5. Set the Hugging Face Hub Token

   ```bash
   echo "export HUGGING_FACE_HUB_TOKEN=hf_XXXX" >> ~/.bashrc
   ```

6. Add your new model ids as env vars in `scripts/run-dev.sh`
   
   ```bash
   ## scripts/run-dev.sh
   
   docker run \
      -e MY_MODEL_IDS="
         model_id_1 
         model_id_2
      " \
      ...
   ```

7. Run the development script

   ```bash
   ./scripts/run-dev.sh common common.my_model
   ```
   
8. Test the model by sending a request to the celery worker.
   
    If you are using is a vm, Port forward the redis & rabbitmq ports from the vm to your local machine.

    Eg for azure:

    ```bash
    az ssh config --name <vm-name> --resource-group <resource-group> --file ~/.ssh/config.d/<vm-name> --overwrite
    ssh <vm-name> -vN -L 6374:localhost:6379 -L 5674:localhost:5672
    ```
   
   Note how we use ports `6374` and `5674`. This is to avoid potential conflicts with a local redis and rabbitmq.

   ```python
   from celery import Celery

   app = Celery()
   app.conf.broker_url = "amqp://localhost:<port>"
   app.conf.result_backend = "redis://localhost:<port>"
   app.conf.result_extended = True
   
   result = app.send_task(
       "my_model_task",
       kwargs=dict(
           pipeline=dict(model_id="model_id_1"), 
           inputs=dict(text="Hello, World!"),
       ), 
       queue="gooey-gpu/model_id_1",  # by default the queue name is gooey-gpu/<model_id>
   )
   print(result.get())  # { "image": "..." }
   ```

9. During this, try to record the GPU usage using [`nvitop`](https://github.com/XuehaiPan/nvitop) (or `nvidia-smi`) 
 
   This will come handy to define the resource limits in the helm chart.
   Once you have this number, you need to convert this to the equivalent CPU memory limit. 
   
   To do this, you can use the following formula:
   
   ```
   cpu_memory_limit = (cpu_memory_capacity / gpu_memory_capacity) * gpu_memory_limit
   ```
   E.g. for an azure Standard_NC24ads_A100_v4 with 216 Gib CPU memory and 80 Gib GPU memory, and a diffusion model with a max GPU memory usage of 7 Gib, the CPU memory limit would be:
   
   ```
   (216 / 80) * 7 ~= 20Gi
   ```
   
   This helps us put multiple models in the same GPU and avoid CUDA OOM errors.
   

10. Once you are confident that the model is working as expected, upload the docker image to a container registry.

    ```bash
    docker tag gooey-gpu-common <registry>/<image>:<tag>
    docker push <registry>/<image>:<tag>
    ```

    You might need to login to your container registry before pushing the image. Eg for azure:
    ```bash
    az acr login --name <registry>
    ```
11. Update the `model-values.yaml` file with the new image (or create a new file)
  - Under `rabbitmqAutoscaling`, add the new env var name
      ```yaml
      rabbitmqAutoscaling:
         queueNameVars:
            - MY_MODEL_IDS
      ```

  - Under `deployments`, add your new model. 
      ```yaml
      deployments:
      - name: "common-my-model"
        image: "<registry>/<image>:<tag>"
        limits:
          memory: "20Gi"
          cpu: 1
        env:
          IMPORTS: |-
            my_model
          MODEL_IDS: |-
            model_id_1
            model_id_2
      ```

12. Deploy the helm chart
   - Check the diff to make sure the new model is being added
      ```bash
      helm diff upgrade gooey-gpu-1 chart -f chart/model-values.yaml -f env-values.yaml
      ```
   - Deploy the chart
      ```bash
      helm upgrade gooey-gpu-1 chart -f chart/model-values.yaml -f env-values.yaml
      ```

### Older Deep Learning Models / Scripts

We have a `retro/` folder that  contains dependencies for older projects using CUDA 11.6 & PyTorch v1

The recommended way to turn a set of research scripts like [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) into a deployable container are:

1. Add the project as a submodule in the `retro/` folder.
    ```bash
    cd retro/
    git submodule add <project-url>
    ```

2. Create a python module in the `retro/` folder that imports the project's modules.

    If you're lucky and the project has an installable package, you can just add the package to the `requirements.txt` file. 

    Otherwise, you can create a python module that imports the project's modules:
    ```python
    ## retro/my_model.py
    import sys
    
    sys.path.append(os.path.join(os.path.dirname(__file__), "<project-dir>"))
    ```
   
3. Copy the code from the project's inference script, and make a celery task out of it.
    
    If you're lucky the project has a separate load() and predict() function. In that case you can just call the load() fn from `setup_queues()`, and the predict() fn from the celery task.
    
    Otherwise, you will have to decode the inference script and write separate functions for loading the model and running inference.

4. Once you have these written, you can follow the same steps as the common models to deploy the model.

### 💣 Secret Scanning

Gitleaks will automatically run pre-commit (see `pre-commit-config.yaml` for details) to prevent commits with secrets in the first place. To test this without committing, run `pre-commit` from the terminal. To skip this check, use `SKIP=gitleaks git commit -m "message"` to commit changes. Preferably, label false positives with the `#gitleaks:allow` comment instead of skipping the check.

Gitleaks will also run in the CI pipeline as a GitHub action on push and pull request (can also be manually triggered in the actions tab on GitHub). To update the baseline of ignored secrets, run `python ./scripts/create_gitleaks_baseline.py` from the venv and commit the changes to `.gitleaksignore`.
