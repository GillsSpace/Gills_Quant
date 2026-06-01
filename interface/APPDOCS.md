## To set up server:
1. `sudo nvim /etc/systemd/system/gills_quant.service`
2. Paste the following, replacing the paths and user as needed:
```
[Unit]
Description=Gills Quant Server
After=network.target tailscaled.service

[Service]
User=willse
WorkingDirectory=/home/willse/Gills_Quant
ExecStart=/home/willse/Gills_Quant/venv/bin/python -m app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
3. Run the following commands:
```
sudo systemctl daemon-reload
sudo systemctl start gills_quant.service
sudo systemctl enable gills_quant.service
```
## Helpful commands:
- `sudo systemctl status gills_quant.service` - Check the status of the service
- `sudo journalctl -u gills_quant.service -f` - View real-time logs
- `sudo systemctl restart gills_quant.service` - Restart the service (Needed after code changes)