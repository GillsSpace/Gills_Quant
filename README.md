# Gills_Quant
A personal project focused on statistical and quantitative trading 

## Setup:
1. Clone the repo to home directory
2. run `source setup.bash` to setup python evniroment
3. run `python -m tool` and `s-d` to setup file structure
4. install node:
```
sudo apt update
sudo apt install nodejs npm
```

5. Add in secrets folder a `keys.json` file:
```
{
    "schwab": {
        "app_key": "",
        "app_secret": "",
    },
}
```

6. Install nfty to android phone and subscribe to `gills_quant_trading`

7. run `docker compose up -d --build` to start api service to recive tokens.

7. run `tailscale serve --bg 8000` on gill_01. Take note of url.

### DB Setup:
