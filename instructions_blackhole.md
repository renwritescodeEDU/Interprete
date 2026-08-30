# Configuración de BlackHole en macOS

1. Instalar BlackHole: `brew install blackhole-2ch`
2. Abrir **Configuración de Audio MIDI** (Audio MIDI Setup) en macOS.
3. Hacer clic en el `+` (abajo a la izquierda) y crear un **Dispositivo de Salida Múltiple** (Multi-Output Device).
4. Seleccionar tus auriculares/altavoces principales y **BlackHole 2ch**.
5. **CRÍTICO:** Activa la casilla **"Corrección de deriva" (Drift Correction)** junto a "BlackHole 2ch" para evitar desincronización de audio (glitches) en sesiones largas.
6. En la configuración de sonido de macOS, selecciona el **Dispositivo de Salida Múltiple** como salida de audio.
7. El sistema de interpretación capturará el audio seleccionando "BlackHole 2ch" como dispositivo de entrada.

> **Windows:** Consulta [`docs/audio_routing.md`](docs/audio_routing.md) para alternativas
> a BlackHole (VB-CABLE, Stereo Mix, etc.).