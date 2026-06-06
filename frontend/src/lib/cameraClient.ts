export class CameraClient {
  private ws:         WebSocket;
  private stream:     MediaStream | null = null;
  private canvas:     HTMLCanvasElement;
  private ctx:        CanvasRenderingContext2D;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private video:      HTMLVideoElement;

  constructor(private sessionId: string, private wsUrl: string) {
    this.canvas = document.createElement('canvas');
    this.ctx    = this.canvas.getContext('2d')!;
    this.video  = document.createElement('video');
    this.video.muted = true;
    this.ws = new WebSocket(`${wsUrl}/ws/video/${sessionId}`);
  }

  async start(fps: number = 5): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'environment' },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();

    this.canvas.width  = 640;
    this.canvas.height = 360;

    this.intervalId = setInterval(() => {
      if (this.ws.readyState !== WebSocket.OPEN) return;
      this.ctx.drawImage(this.video, 0, 0, 640, 360);
      this.canvas.toBlob(
        (blob) => {
          if (blob) this.ws.send(blob);
        },
        'image/jpeg',
        0.8,
      );
    }, 1000 / fps);
  }

  getVideoElement(): HTMLVideoElement {
    return this.video;
  }

  stop(): void {
    if (this.intervalId) clearInterval(this.intervalId);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ws.close();
  }
}
