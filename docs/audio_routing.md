# Audio Routing Setup (macOS & Windows)

This system can capture **two audio sources** during a call:

1. **Your microphone** (the interpreter's voice) — always an input device.
2. **The client/agent audio** (call audio from the CRM, phone app, or browser) — requires a
   **virtual audio device** so the system can "listen" to what is played through the speakers
   while you still hear it through your headphones.

The application itself is platform-neutral: it simply enumerates input devices and lets you
pick one in the dropdown. What differs between operating systems is **how you create the
virtual audio device** that redirects call audio into the system.

---

## macOS — BlackHole (original setup)

1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (macOS utility).
3. Click `+` (bottom-left) and create a **Multi-Output Device**.
4. Select your main headphones/speakers **and** BlackHole 2ch.
5. **CRITICAL:** enable **Drift Correction** next to "BlackHole 2ch" to avoid audio
   desynchronization (glitches) in long sessions.
6. In macOS sound settings, select the **Multi-Output Device** as the audio output.
7. In Interprete, select **"BlackHole 2ch"** as the input device.

---

## Windows — virtual audio device options

There is no BlackHole on Windows. The equivalent is a **loopback/virtual audio cable**:
audio played by the CRM goes to the virtual device, and Interprete records from that same
virtual device as an input — while you still hear it through headphones.

### Option 1 (Recommended): VB-CABLE (free, universal)

[VB-CABLE](https://vb-audio.com/Cable/) is a free virtual audio cable that works with any
sound card and any application.

**Setup:**

1. Download and install **VB-CABLE** from https://vb-audio.com/Cable/
   (the free "CABLE" entry; the paid "VB-CABLE A+B" also works).
2. After install, two new devices appear in Windows Sound settings:
   - **CABLE Input** (playback device)
   - **CABLE Output** (recording device)
3. Restart any application that was already running so it sees the new devices.

**Routing during a call:**

1. In your CRM / phone app / browser, set the **speaker/output** to **"CABLE Input"**.
   (In most web apps this is the browser's output device or the app's speaker setting.)
2. Keep your headphones connected to your normal output (e.g., "Headphones (Realtek)").
   If your CRM allows only one output, use **Option 2** below instead.
3. In Interprete, select **"CABLE Output"** in the device dropdown — the call audio now
   flows into the interpreter in real time.

**Hearing both (call audio + Interprete):** the call audio is routed to CABLE Input, so to
still hear it you must also send it to your headphones:

- Preferred: in the CRM app, select **both** "Headphones" and "CABLE Input" if the app
  supports multiple outputs (some browsers do, via the media hub).
- Alternative: use **Windows "Stereo Mix"** or **"Listen to this device"** (Option 3) to
  duplicate CABLE Input to your headphones.

### Option 2: Windows "Listen to this device" (built-in, no extra software)

Windows can route any input device's audio to your speakers — the reverse of what we need,
but combined with your mic it covers the "hear everything" requirement:

**Setup:**

1. Open **Sound settings → Input** and pick the microphone you will use for the call
   (e.g., your headset mic).
2. That mic is already an input device for Interprete — no routing needed for your voice.
3. For the **call audio**: most CRM/browser apps let you choose the output. Leave the
   output at your headphones. To also feed call audio into Interprete you still need a
   virtual cable — this option alone cannot capture speaker audio without one.

> **Verdict:** "Listen to this device" helps with monitoring but is **not** a full
> BlackHole replacement by itself. Use it as a complement to VB-CABLE when your CRM
> cannot select two outputs: set the CRM output to **CABLE Input**, then in Windows
> **Sound → Recording → CABLE Output → Properties → Listen → "Listen to this device"**
> so the call audio is simultaneously played through your headphones.

### Option 3: "Stereo Mix" (built-in on many Realtek/Creative sound cards)

Stereo Mix is a built-in recording device that captures **everything your sound card
plays**, which is exactly the call audio.

**Setup:**

1. Right-click the speaker icon → **Sound settings → More sound settings**.
2. Go to the **Recording** tab. Right-click in the list and enable
   **"Show Disabled Devices"** and **"Show Disconnected Devices"**.
3. If **"Stereo Mix"** appears, right-click it → **Enable**.
4. (Optional) Right-click → **Properties → Listen** if you also want to hear it.
5. In Interprete, select **"Stereo Mix"** as the input device — it captures the combined
   audio of your system (call audio + anything else playing).

**Limitations:**

- Not all sound cards expose Stereo Mix (missing on many USB DACs and laptops).
- It captures **all** system audio, including Interprete's own (none, it plays no audio)
  and any notifications — mute/stop other playback to keep the feed clean.

### Option 4: Virtual Audio Cable (VAC, paid)

[Virtual Audio Cable (VAC)](https://vac.muzychenko.net/en/) is a commercial alternative
to VB-CABLE with multiple simultaneous cables and lower latency for professional use.
Setup is analogous to VB-CABLE: it installs one or more virtual playback/recording pairs;
route the CRM output to the VAC playback endpoint and select the VAC recording endpoint
in Interprete.

---

## Verifying the routing before a live call

1. Start Interprete and confirm the status label shows **"System Ready"**.
2. In the device dropdown, select the virtual input (e.g., **CABLE Output**).
3. Click **Start Recording** and play a test audio file through the CRM/CRM test call.
4. You should see the **partial transcript** appear within a few seconds and the
   **translation** bubble appear after you stop recording.

> Tip: Interprete remembers the last device used (`<repo>/.config/preferences.json`),
> so the virtual device is selected automatically on the next launch.
