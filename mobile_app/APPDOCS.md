# Schwab Token Update Mobile App (PWA) Service Documentation

## Service Setup (`gills_quant_mobile.service`)

1. Create the systemd service file:
   ```bash
   sudo nvim /etc/systemd/system/gills_quant_mobile.service
   ```

2. Paste the following configuration (adjusting user/paths if needed):
   ```ini
   [Unit]
   Description=Gills Quant Mobile App Service
   After=network.target tailscaled.service

   [Service]
   User=willse
   WorkingDirectory=/home/willse/Gills_Quant/mobile_app
   ExecStart=/home/willse/Gills_Quant/mobile_app/start.sh
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. Reload systemd, start, and enable the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start gills_quant_mobile.service
   sudo systemctl enable gills_quant_mobile.service
   ```

---

## Helpful Commands

- **Check status**: `sudo systemctl status gills_quant_mobile.service`
- **View live logs**: `sudo journalctl -u gills_quant_mobile.service -f`
- **Restart service**: `sudo systemctl restart gills_quant_mobile.service`
- **Stop service**: `sudo systemctl stop gills_quant_mobile.service`

---

## Accessing on Phone via Tailscale (HTTPS Required for PWA)

Mobile web browsers (Chrome & Safari) require **HTTPS** for PWA functionality ("Add to Home Screen" & Service Workers).

1. Enable HTTPS proxy on Tailscale for port 8001:
   ```bash
   sudo tailscale serve --bg --https=8001 8001
   ```

2. Open your mobile web browser and navigate to:
   - **`https://gill-01.taileb5b7d.ts.net:8001`** *(or `https://gill-01:8001`)*

3. Tap **"Add to Home Screen"** in your phone browser menu to install the PWA.
