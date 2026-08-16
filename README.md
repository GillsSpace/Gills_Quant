# Gills_Quant

A personal project focused on statistical research, quantitative trading, and automated execution via the **Charles Schwab Developer API** (`schwabdev`).

---

## Setup Guide

1. **Clone the repo to home directory**:
   ```bash
   git clone https://github.com/willse/Gills_Quant.git
   cd Gills_Quant
   ```

2. **Setup Python environment**:
   ```bash
   source setup.bash
   ```

3. **Initialize directory structure & Zarr DB**:
   ```bash
   python -m tool
   ```

4. **Install Node.js & npm (for Web Dashboard UI)**:
   ```bash
   sudo apt update
   sudo apt install nodejs npm
   ```

5. **Configure API Secrets (`secrets/keys.json`)**:
   Create a `secrets/keys.json` file:
   ```json
   {
       "schwab": {
           "app_key": "YOUR_SCHWAB_APP_KEY",
           "app_secret": "YOUR_SCHWAB_APP_SECRET"
       },
       "alpaca": {
           "key": "YOUR_ALPACA_KEY",
           "secret": "YOUR_ALPACA_SECRET"
       }
   }
   ```

6. **Setup Mobile App & Push Notifications**:
   * Install `ntfy` on your phone and subscribe to `gills_quant_trading`.
   * Start the Mobile PWA API service:
     ```bash
     docker compose up -d --build
     ```

7. **Tailscale & Remote Access**:
   ```bash
   tailscale serve --bg 8000
   ```

8. **Strategy Framework Documentation**:
   For detailed strategy architecture, Schwab API actions, and execution pipeline documentation, see [`strategies/START_HERE.md`](file:///home/willse/Gills_Quant/strategies/START_HERE.md).

---

*Gills_Quant Architecture Team*
