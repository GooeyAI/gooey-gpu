## Save Code to VM

To save code to the vm, use either `rsync` or [JetBrains Deployment Tutorial](https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html).

## Port Forwarding

Port forward the redis & rabbitmq ports from the vm to your local machine:

(note how we use ports `6374` and `5674` to avoid conflicts with local redis and rabbitmq)

```bash
gcloud compute ssh --zone <zone> <vm> -- -vN -L 6374:localhost:6379 -L 5674:localhost:5672
```

## Set Hugging Face Hub Token

Make sure you have a `HUGGING_FACE_HUB_TOKEN` set. You can also add this to `~/.bashrc`

```bash
export HUGGING_FACE_HUB_TOKEN=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
```

## Add Models for Testing

Add the Models you'd like to test in `run-dev.sh`. You should also specify the modules that the celery worker should load.

```bash
./scripts/run-dev.sh <common|backports|deforum_sd>
```

## Install netadata for monitoring

Refer to the official [Netdata installation guide](https://learn.netdata.cloud/docs/installing/#install-on-linux-with-one-line-installer) for more information.

```bash
wget -O /tmp/netdata-kickstart.sh https://my-netdata.io/kickstart.sh && sh /tmp/netdata-kickstart.sh --stable-channel
```

Then, update the python.d configuration:

```bash
cd /opt/netdata/etc/netdata/
sudo ./edit-config python.d.conf
```

Update the following lines:

```
nvidia_smi: yes
```

Update the main configuration file:

```bash
cd /opt/netdata/etc/netdata/
sudo nano netdata.conf
```

Update these lines:

```
[plugins]
    python.d = yes
```

Change the ownership of the python.d plugin:

```bash
cd /opt/netdata/usr/libexec/netdata/plugins.d/
sudo chown netdata python.d.plugin
```

Finally, restart the netdata service:

```bash
sudo systemctl restart netdata
```
