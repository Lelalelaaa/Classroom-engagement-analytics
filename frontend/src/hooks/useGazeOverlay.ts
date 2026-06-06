'use client';
import { useEffect, useRef } from 'react';

interface FaceOverlay {
  bbox:    [number, number, number, number]; // normalised x,y,w,h
  yaw:     number;   // radians
  pitch:   number;   // radians
  emotion: string;
  gaze:    string;
  engaged: boolean;
  score:   number;   // 0–1
}

const EMOTION_EMOJI: Record<string, string> = {
  happiness: '😊', surprise: '😮', neutral: '😐',
  sadness:   '😢', anger:   '😠', disgust: '🤢',
  fear:      '😨', contempt:'😒',
};

function drawArrow(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  yaw: number, pitch: number,
  len: number, color: string,
) {
  const ex = cx + Math.sin(yaw) * len;
  const ey = cy + Math.sin(pitch) * len;

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2.5;
  ctx.stroke();

  // Arrow head
  const angle  = Math.atan2(ey - cy, ex - cx);
  const hLen   = 10;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - hLen * Math.cos(angle - 0.4), ey - hLen * Math.sin(angle - 0.4));
  ctx.lineTo(ex - hLen * Math.cos(angle + 0.4), ey - hLen * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

export function useGazeOverlay(
  canvasRef: React.RefObject<HTMLCanvasElement>,
  videoRef:  React.RefObject<HTMLVideoElement>,
  faces:     FaceOverlay[],
) {
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      // Match canvas to video size
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width  = video.videoWidth  || canvas.offsetWidth;
        canvas.height = video.videoHeight || canvas.offsetHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const W = canvas.width;
      const H = canvas.height;

      faces.forEach((face) => {
        const [nx, ny, nw, nh] = face.bbox;
        const x  = nx * W;
        const y  = ny * H;
        const bw = nw * W;
        const bh = nh * H;
        const cx = x + bw / 2;
        const cy = y + bh / 2;

        // Box color: green = engaged, red = not
        const boxColor = face.engaged
          ? 'rgba(74, 222, 128, 0.85)'   // green
          : 'rgba(248, 113, 113, 0.85)'; // red

        // ── Bounding box ──────────────────────────────────────────────────────
        ctx.strokeStyle = boxColor;
        ctx.lineWidth   = 2;
        ctx.strokeRect(x, y, bw, bh);

        // Corner accents
        const clen = 12;
        ctx.lineWidth = 3;
        [[x, y, 1, 1], [x + bw, y, -1, 1], [x, y + bh, 1, -1], [x + bw, y + bh, -1, -1]].forEach(
          ([cx2, cy2, dx, dy]) => {
            ctx.beginPath();
            ctx.moveTo(cx2 as number, cy2 as number);
            ctx.lineTo((cx2 as number) + (dx as number) * clen, cy2 as number);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx2 as number, cy2 as number);
            ctx.lineTo(cx2 as number, (cy2 as number) + (dy as number) * clen);
            ctx.stroke();
          }
        );

        // ── Gaze arrow ────────────────────────────────────────────────────────
        const arrowColor = face.engaged ? '#4ade80' : '#f87171';
        drawArrow(ctx, cx, cy, face.yaw, face.pitch, Math.min(bw, bh) * 0.6, arrowColor);

        // ── Labels ────────────────────────────────────────────────────────────
        const emoji    = EMOTION_EMOJI[face.emotion] ?? '😐';
        const scoreStr = `${Math.round(face.score * 100)}%`;
        const label    = `${emoji} ${scoreStr}`;

        // Label background
        ctx.font       = 'bold 13px Inter, system-ui, sans-serif';
        const tw       = ctx.measureText(label).width + 10;
        const lx       = x;
        const ly       = y - 24;
        ctx.fillStyle  = face.engaged ? 'rgba(0,0,0,0.6)' : 'rgba(80,0,0,0.7)';
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw, 20, 4);
        ctx.fill();

        ctx.fillStyle  = '#ffffff';
        ctx.fillText(label, lx + 5, ly + 14);

        // Engagement score bar at bottom of face box
        const barH  = 4;
        const barY  = y + bh + 4;
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.fillRect(x, barY, bw, barH);
        ctx.fillStyle = arrowColor;
        ctx.fillRect(x, barY, bw * face.score, barH);
      });

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [canvasRef, videoRef, faces]);
}
