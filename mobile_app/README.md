# Schwab Token Update Mobile App (PWA)

A mobile-first Progressive Web Application (PWA) built to replicate the Schwab token update and Tailscale upload functionality of `tool.py` directly from an Android phone.

---

## Features
- **Mobile-First Responsive PWA**: Dark mode UI optimized for Android touchscreens.
- **Installable on Android**: Open in Chrome/Firefox on Android and select **"Add to Home Screen"** to install as a standalone app with custom app icon and splash screen.
- **One-Tap Token Refresh & Sync**: Refreshes Schwab API access tokens and posts the updated `tokens.db` SQLite file to `https://gill-01.taileb5b7d.ts.net/update-token`.
- **OAuth Portal Access**: Provides direct access to Schwab OAuth login for 7-day refresh token renewals.
- **Live Status & Countdown**: Shows real-time validity and expiration countdowns for access tokens and refresh tokens.

---

## How to Run

1. Open a terminal on your computer and execute:
   ```bash
   ./app/start.sh
   ```
2. On your Android phone (connected to local Wi-Fi or Tailscale), open your web browser and navigate to:
   ```
   http://<your-computer-ip>:8000
   ```
3. Tap **"Add to Home Screen"** in your phone browser menu to install it like a native Android app!
