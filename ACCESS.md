# Getting access to the Jetson server

The server is not on the public internet. It's reachable only through
**Tailscale**, a private network between approved devices. So there are two
parts: join the network, then log in.

Takes about five minutes, once.

---

## Part 1 — for the server owner (do this first)

You're sharing **one machine**, not your whole network. The other person uses
their own free account and can only see the Jetson.

1. Go to **[login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)**
2. Find the Jetson in the list (hostname `drstreet`)
3. Click the **⋯** menu on its row → **Share…**
4. Copy the share link and send it to them

Then create them their own login key, so it can be revoked without affecting
yours:

1. Open the admin UI → **API keys**
2. Name: their name. Scope: **admin**. Leave Device id empty.
3. Click **Create key** and copy the key it shows — **it is displayed once**
4. Send them the key and the server address

> Send the Tailscale share link and the API key separately if you can
> (e.g. link by email, key by message). Either one alone is useless.

---

## Part 2 — for the new user (Windows)

### 1. Install Tailscale

Download and run the installer:
**[tailscale.com/download/windows](https://tailscale.com/download/windows)**

Or in PowerShell:

```powershell
winget install tailscale.tailscale
```

### 2. Sign in

Tailscale appears in the system tray (bottom-right, next to the clock — you may
need to click the **^** arrow to see it).

- Right-click the tray icon → **Log in…**
- A browser opens. Sign in with Google, Microsoft, or GitHub — any account is
  fine, it's free, and it creates your own private network.

### 3. Accept the shared machine

Open the share link you were sent, and click **Accept**.

The Jetson now appears in your Tailscale device list.

### 4. Open the server

In your browser:

```
http://100.103.105.68:8000/ui
```

Paste the API key you were sent, click **Log in**. That's it — the session
lasts effectively forever, so you won't have to do this again on this machine.

---

## If the page won't load

**"This site can't be reached" / timeout**

Tailscale isn't connected. Check the tray icon — it should say *Connected*. If
it says *Log in*, do that first.

**"HTTP ERROR 503" or "This page isn't working"**

Almost always a **browser extension**, not the server. Ad blockers and privacy
extensions often block `100.x` addresses, because that range is also used by
some ISPs and they treat it as a security risk. Tailscale uses it legitimately.

Test it: open an **incognito window** (`Ctrl+Shift+N`) and try the address
again. Extensions are disabled there by default.

- **Works in incognito?** It's an extension. Go to `chrome://extensions`, turn
  off your ad/privacy blocker, and reload. Better: whitelist
  `100.103.105.68` in that extension rather than disabling it entirely.
- **Still fails in incognito?** Clear your cache
  (`chrome://settings/clearBrowserData` → Cached images and files), then retry.

**"Not a valid admin key"**

The key is case-sensitive and must be pasted whole. Ask for a fresh one — keys
are stored hashed and cannot be looked up, only reissued.

---

## What you can do once you're in

| Page | What it's for |
|---|---|
| **Dashboard** | Totals, which model is live, recent errors |
| **Models** | Upload a model, activate it, and **Try a model** — drop in an image and see the prediction |
| **Performance** | CPU / GPU / memory / temperature, and per-model inference latency |
| **Logs** | Live server log |
| **How to** | In-app guide to everything above |

Start with **Models → Try a model**. Pick an image, press Predict, and you'll
see the grade with the full probability distribution.

---

## Notes

- Nothing here is exposed to the public internet, and no ports are forwarded.
  Access is only via Tailscale, and only for devices explicitly shared with.
- The server can be reached from anywhere — home, office, mobile data — as long
  as Tailscale is connected on your device.
- Tailscale also has iOS and Android apps; the same share link and URL work
  there.
